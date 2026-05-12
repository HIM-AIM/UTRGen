import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import GPT2Config
from .downstream_regressor import get_regressor
from transformers import GPT2Config, GPT2LMHeadModel

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
        
        self.log("train_loss", loss, prog_bar=True,on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=100),
            'interval': 'step'
        }
        return [optimizer], [scheduler]


def build_gpt2(tokenizer, n_embd, n_layer, n_head, n_positions):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_positions,
        pad_token_id=tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(config)
    return model, config


#### rewards function 
class DownstreamOracle(pl.LightningModule):
    def __init__(self, tokenizer, ckpt:str, n_ctx=256, n_embd=384, n_layer=8, n_head=16, regressor=1, input_len=256):
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
        return reward # shape: (batch_size,)

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
        # self.threshold = (len(paths) // 2) + 1
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
