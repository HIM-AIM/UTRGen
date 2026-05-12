import copy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Config
import wandb
from downstream_regressor import get_regressor, build_gpt2
from transformers import GPT2Config, GPT2LMHeadModel
from trl.trainer.utils import selective_log_softmax, entropy_from_logits
from pytorch_lightning.loggers import WandbLogger
import RNA


class LitUTRPretrain(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=100),
            'interval': 'step'
        }
        return [optimizer], [scheduler]


#### rewards function

class DownstreamOracle(pl.LightningModule):
    def __init__(self, tokenizer, ckpt: str, n_ctx=256, n_embd=384, n_layer=8, n_head=16, regressor=1, input_len=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.regressor = regressor
        self.input_len = input_len
        self.oracle = self._load_ckpt(ckpt)

    def _load_ckpt(self, ckpt):
        model, _ = build_gpt2(self.tokenizer, self.n_embd, self.n_layer, self.n_head, self.n_ctx)
        lit_model = get_regressor(self.regressor)(pretrained_model=model, input_len=self.input_len)
        ckpt = torch.load(ckpt, map_location='cpu',)
        lit_model.load_state_dict(ckpt['state_dict'], strict=False)
        lit_model.eval()
        for param in lit_model.parameters():
            param.requires_grad = False
        return lit_model

    def forward(self, input_ids, attention_mask, *args):
        if self.oracle is None:
            raise ValueError("No oracle model has been loaded. Please provide a checkpoint path in the constructor.")
        reward = self.oracle(input_ids, attention_mask).squeeze(-1)
        return reward  # shape: (batch_size,)


class RegressionReward(nn.Module):
    def __init__(self, tokenizer, paths):
        super().__init__()
        self.tokenizer = tokenizer
        self.oracle = nn.ModuleList([DownstreamOracle(tokenizer=tokenizer, ckpt=path, input_len=128) for path in paths])

    def __call__(self, input_ids, attention_mask, *args):
        if not self.oracle or len(self.oracle) == 0:
            raise ValueError("No oracle models provided.")
        rewards = []
        with torch.no_grad():
            for model in self.oracle:
                rewards.append(model(input_ids, attention_mask))
            rewards = torch.stack(rewards, dim=1).mean(dim=1)
        return rewards  # shape: (batch_size,)


class ClassificationReward(nn.Module):
    def __init__(self, tokenizer, paths, target_class=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.oracle = nn.ModuleList([DownstreamOracle(tokenizer=tokenizer, ckpt=path, regressor=4, input_len=256) for path in paths])
        self.target_class = target_class

    def __call__(self, input_ids, attention_mask, *args):
        if not self.oracle or len(self.oracle) == 0:
            raise ValueError("No oracle models provided.")
        rewards = []
        with torch.no_grad():
            for model in self.oracle:
                logits = model(input_ids, attention_mask)
                prob = torch.softmax(logits, dim=-1)
                rewards.append(prob[:, self.target_class])
            rewards = torch.stack(rewards, dim=1).mean(dim=1)
        return rewards  # shape: (batch_size,)


def gc_penalty(input_ids, attention_mask, tokenizer, gc_min=0.4, gc_max=0.6):
    g_id = tokenizer.convert_tokens_to_ids('G')
    c_id = tokenizer.convert_tokens_to_ids('C')
    gc_counts = ((input_ids == g_id) | (input_ids == c_id)).float().sum(dim=1)
    lengths = attention_mask.sum(dim=1).float()
    gc_content = gc_counts / lengths
    penalty = torch.zeros_like(gc_content)
    penalty += (gc_content < gc_min).float() * (gc_min - gc_content)
    penalty += (gc_content > gc_max).float() * (gc_content - gc_max)
    return -penalty


def gc_penalty_mse(input_ids, attention_mask, tokenizer, gc_min=0.4, gc_max=0.6):
    g_id = tokenizer.convert_tokens_to_ids('G')
    c_id = tokenizer.convert_tokens_to_ids('C')
    gc_counts = ((input_ids == g_id) | (input_ids == c_id)).float().sum(dim=1)
    lengths = attention_mask.sum(dim=1).float()
    gc_content = gc_counts / lengths
    lower_violation = torch.relu(gc_min - gc_content)
    upper_violation = torch.relu(gc_content - gc_max)
    penalty = (lower_violation ** 2) + (upper_violation ** 2)
    return -penalty


def gc_penalty_target_mse(input_ids, attention_mask, tokenizer, target_gc=0.5):
    g_id = tokenizer.convert_tokens_to_ids('G')
    c_id = tokenizer.convert_tokens_to_ids('C')
    gc_counts = ((input_ids == g_id) | (input_ids == c_id)).float().sum(dim=1)
    lengths = attention_mask.sum(dim=1).float()
    gc_content = gc_counts / lengths
    penalty = (gc_content - target_gc) ** 2
    return -penalty


def mfe_reward(input_ids, attention_mask, tokenizer):
    seqs = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
    seqs = [seq.replace(" ", "").strip() for seq in seqs]
    lens = attention_mask.sum(dim=1).float()
    rewards = []
    for seq in seqs:
        _, mfe = RNA.fold(seq)
        rewards.append(-mfe)
    rewards = torch.tensor(rewards, device=input_ids.device)
    rewards = rewards / lens
    return rewards  # shape: (batch_size,)


class LitUTRGRPO(pl.LightningModule):
    def __init__(self, model, tokenizer, reward_funcs=[], reward_weights=[], max_gen_len=200, min_gen_len=150, rl_batch_size=32,
                 lr=1e-4, num_generations=4, kl_beta: float = 0.1,
                 top_k=50, top_p=0.9, temperature=1.0, repetition_penalty=1.1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        assert len(self.reward_funcs) == len(self.reward_weights), "Number of reward functions must match number of weights."
        for f in reward_funcs:
            if isinstance(f, nn.Module):
                f.eval()
                for param in f.parameters():
                    param.requires_grad = False

        self.max_length = max_gen_len
        self.min_length = min_gen_len
        self.rl_batch_size = rl_batch_size
        self.lr = lr
        self.num_generations = num_generations
        self.kl_beta = kl_beta
        if self.kl_beta > 0:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.rewards_over_epoch = []

    def _sample(self):
        nuc_tokens = ['A', 'T', 'C', 'G']
        nuc_ids = self.tokenizer.convert_tokens_to_ids(nuc_tokens)
        nuc_ids_tensor = torch.tensor(nuc_ids, device=self.device)
        random_indices = torch.randint(0, len(nuc_ids_tensor), (self.rl_batch_size,), device=self.device)
        start_token_ids = nuc_ids_tensor[random_indices]
        input_ids = start_token_ids.unsqueeze(1)

        prompts_repeated = input_ids.repeat_interleave(self.num_generations, dim=0)

        generated_ids = self.model.generate(
            input_ids=prompts_repeated,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=True,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.sep_token_id
        )
        is_eos = (generated_ids == self.tokenizer.sep_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        attention_mask = (sequence_indices <= eos_idx.unsqueeze(1)).long()

        return generated_ids, attention_mask, prompts_repeated

    def _decode(self, generated_ids):
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned = [seq.replace(" ", "").strip() for seq in decoded]
        return cleaned

    def _get_reward(self, generated_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if not self.reward_funcs or len(self.reward_funcs) == 0:
            raise ValueError("No reward functions provided.")
        split_mean_reward_dict = {}
        for i, func in enumerate(self.reward_funcs):
            with torch.no_grad():
                rewards = func(generated_ids, attention_mask, self.tokenizer)
                split_mean_reward_dict[f'reward_{i}_mean'] = rewards.mean().item()
                weighted_rewards = rewards * self.reward_weights[i]
            if i == 0:
                total_rewards = weighted_rewards
            else:
                total_rewards += weighted_rewards
        return total_rewards, split_mean_reward_dict

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, compute_entropy=False):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :]
        logits = logits / self.temperature
        logps = selective_log_softmax(logits, input_ids[:, 1:])
        if compute_entropy:
            with torch.no_grad():
                entropies = entropy_from_logits(logits)
        else:
            entropies = None
        return logps, entropies

    def training_step(self, batch, batch_idx):
        generated_ids, attention_mask, _ = self._sample()
        completion_mask = attention_mask[:, 1:]
        completion_lengths = completion_mask.sum(-1)

        rewards, seplit_mean_reward_dict = self._get_reward(generated_ids, attention_mask)

        # --- advantage ---
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        std_rewards = rewards.std().expand_as(rewards)
        advantages = advantages / (std_rewards + 1e-4)

        # --- policy forward ---
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(self.model, generated_ids, attention_mask, compute_entropy=True)

        # on-policy REINFORCE-style: loss_token = -A * logpi(a_t|s_t)
        per_token_loss = -advantages.unsqueeze(1) * per_token_logps

        # --- for logging ---
        rl_only = ((per_token_loss.detach() * completion_mask).sum(-1) / completion_lengths).mean().item()
        entropy = ((entropies.detach() * completion_mask).sum(-1) / completion_lengths).mean().item()

        mean_kl = 0
        if self.kl_beta > 0.0:
            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(self.ref_model, generated_ids, attention_mask)
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.kl_beta * per_token_kl
            mean_kl = ((per_token_kl.detach() * completion_mask).sum(-1) / completion_lengths).mean().item()

        total_loss = ((per_token_loss * completion_mask).sum(-1) / completion_lengths).mean()

        log_dict = {
            "train_loss_total": total_loss.item(),
            "train_loss_rl": rl_only,
            "train_loss_kl": mean_kl,
            "train_mean_length": completion_lengths.float().mean().item(),
            'train_policy_entropy': entropy,
            "train_reward_weighted_mean": rewards.mean().item(),
            "train_reward_weighted_std": rewards.std().item(),
        }
        log_dict.update(seplit_mean_reward_dict)
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        self.rewards_over_epoch.append(seplit_mean_reward_dict)
        return total_loss

    def validation_step(self, batch, batch_idx):
        generated_ids, attention_mask, _ = self._sample()
        rewards, _ = self._get_reward(generated_ids, attention_mask)
        avg_reward = rewards.mean()
        self.log("val_reward", avg_reward, on_step=False, on_epoch=True, prog_bar=True)

        if (self.current_epoch % 50 == 0) and isinstance(self.logger, WandbLogger):
            decoded_seqs = self._decode(generated_ids)
            decoded_seqs = np.random.choice(decoded_seqs, min(10, len(decoded_seqs)), replace=False)
            table = wandb.Table(columns=["epoch", "length", "sequence"])
            for seq in decoded_seqs:
                table.add_data(self.current_epoch, len(seq), seq)
            self.logger.experiment.log({"Generated Sequences": table})
        return avg_reward

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = [k for k in checkpoint["state_dict"].keys() if
                          k.startswith("reward_funcs.") or k.startswith("ref_model.")]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]
