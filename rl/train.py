import argparse
import json
import os
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from lightning_module import LitUTRGRPO, RegressionReward, ClassificationReward, gc_penalty, gc_penalty_mse, gc_penalty_target_mse, mfe_reward, build_gpt2
from nucleotide_tokenizer import NucleotideTokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrained_model(checkpoint_path, tokenizer, n_ctx=256, n_embd=384, n_layer=8, n_head=16):
    model, _ = build_gpt2(tokenizer, n_embd, n_layer, n_head, n_ctx)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    return model


class DummyRLDataset(Dataset):
    def __init__(self, data_len):
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return idx


def main():
    parser = argparse.ArgumentParser(description="RL fine-tuning with GRPO")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vocab_file', type=str, required=True, help='Tokenizer vocab file')
    parser.add_argument('--pretrained_ckpt', type=str, required=True, help='Path to pretrained Lightning checkpoint')

    parser.add_argument('--reward_func', type=str, nargs='*', default=[], help='Reward functions: [mrl|te|ires|gc|gc_mse|gc_tmse|mfe]')
    parser.add_argument('--reward_func_weights', type=float, nargs='*', default=[], help='Weights for reward functions')

    parser.add_argument('--mrl_ckpt_paths', type=str, nargs='*', default=[], help='MRL oracle checkpoint path(s)')
    parser.add_argument('--te_ckpt_paths', type=str, nargs='*', default=[], help='TE oracle checkpoint path(s)')
    parser.add_argument('--ires_ckpt_paths', type=str, nargs='*', default=[], help='IRES oracle checkpoint path(s)')

    parser.add_argument('--n_ctx', type=int, default=256, help='Max sequence length')
    parser.add_argument('--n_embd', type=int, default=384, help='GPT2 embedding dimension')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of Transformer layers')
    parser.add_argument('--n_head', type=int, default=16, help='Number of attention heads')

    parser.add_argument('--min_gen_length', type=int, default=50, help='Min generation length')
    parser.add_argument('--max_gen_length', type=int, default=100, help='Max generation length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    parser.add_argument('--num_generations', type=int, default=4, help='Number of sequences per prompt')
    parser.add_argument('--kl_beta', type=float, default=0.1, help='KL divergence beta')

    parser.add_argument('--task_name', type=str, required=True, help='Task name for logging and checkpointing')
    parser.add_argument('--wandb_project', type=str, default='UTRGEN_RL')
    parser.add_argument('--output_dir', type=str, default='checkpoints_rl', help='Output directory')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)
    tokenizer = NucleotideTokenizer(args.vocab_file)
    assert len(args.reward_func) == len(args.reward_func_weights), \
        "Length of reward_func and reward_func_weights must be the same."

    reward_funcs = []
    reward_weights = args.reward_func_weights
    for func_name in args.reward_func:
        if func_name == 'mrl':
            assert len(args.mrl_ckpt_paths) > 0, "MRL checkpoint paths must be provided for MRL reward function."
            reward_func = RegressionReward(tokenizer, args.mrl_ckpt_paths).cuda().eval()
            reward_funcs.append(reward_func)
        elif func_name == 'te':
            assert len(args.te_ckpt_paths) > 0, "TE checkpoint paths must be provided for TE reward function."
            reward_func = RegressionReward(tokenizer, args.te_ckpt_paths).cuda().eval()
            reward_funcs.append(reward_func)
        elif func_name == 'ires':
            assert len(args.ires_ckpt_paths) > 0, "IRES checkpoint paths must be provided for IRES reward function."
            reward_func = ClassificationReward(tokenizer, args.ires_ckpt_paths).cuda().eval()
            reward_funcs.append(reward_func)
        elif func_name == 'gc':
            reward_func = gc_penalty
            reward_funcs.append(reward_func)
        elif func_name == 'gc_mse':
            reward_func = gc_penalty_mse
            reward_funcs.append(reward_func)
        elif func_name == 'gc_tmse':
            reward_func = gc_penalty_target_mse
            reward_funcs.append(reward_func)
        elif func_name == 'mfe':
            reward_func = mfe_reward
            reward_funcs.append(reward_func)
        else:
            raise ValueError(f"Unknown reward function name: {func_name}")

    hf_model = load_pretrained_model(args.pretrained_ckpt, tokenizer, n_ctx=args.n_ctx, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head)
    lit_rl_model = LitUTRGRPO(
        model=hf_model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        min_gen_len=args.min_gen_length,
        max_gen_len=args.max_gen_length,
        rl_batch_size=args.batch_size,
        lr=args.lr,
        num_generations=args.num_generations,
        kl_beta=args.kl_beta
    ).cuda()
    print(f"Successfully loaded model from '{args.pretrained_ckpt}'")

    save_name = 'debug' if args.debug else args.task_name
    split_dir = os.path.join(args.output_dir, save_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=split_dir,
        filename=f"UTRGen-RL-{save_name}" + "-{epoch:02d}-{val_reward:.2f}",
        save_top_k=1,
        monitor="val_reward",
        mode="max"
    )

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=save_name,
        job_type='finetune',
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, TQDMProgressBar()],
        accelerator="gpu",
        logger=None if args.debug else wandb_logger,
        log_every_n_steps=10
    )

    train_dataset = DummyRLDataset(data_len=args.batch_size)
    val_dataset = DummyRLDataset(data_len=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    print("Starting Reinforcement Learning Fine-tuning...")
    trainer.fit(lit_rl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    with open(os.path.join(split_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    rewards = []
    rewards_weighted = []
    ids = list(lit_rl_model.rewards_over_epoch[0].keys())
    for reward in lit_rl_model.rewards_over_epoch:
        row = []
        w_row = []
        for i, id in enumerate(ids):
            row.append(reward[id])
            w_row.append(reward[id] * reward_weights[i])
        rewards.append(row)
        w_row.append(sum(w_row))
        rewards_weighted.append(w_row)
    reward_df = pd.DataFrame(rewards, columns=ids)
    reward_df.to_csv(os.path.join(split_dir, "rewards_over_epochs.csv"), index=False)
    ids.append('reward_weighted_total')
    reward_weighted_df = pd.DataFrame(rewards_weighted, columns=ids)
    reward_weighted_df.to_csv(os.path.join(split_dir, "reward_weighted_over_epochs.csv"), index=False)


if __name__ == "__main__":
    main()
