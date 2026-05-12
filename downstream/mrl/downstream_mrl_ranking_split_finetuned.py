import os
import random
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import GPT2Config, GPT2LMHeadModel

from downstream_regressor import get_regressor
from nucleotide_tokenizer import NucleotideTokenizer
from regression_dataset import mRNA_RegressionDataset
from data_utils import read_5putr_from_csv_regression_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tokenize_sequences(sequences, tokenizer, max_length):
    tokenized_data = []
    for seq in sequences:
        if not seq:
            continue
        encoded = tokenizer(
            seq,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True
        )
        tokenized_data.append(encoded)
    return tokenized_data


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


def main():
    parser = argparse.ArgumentParser(description="MRL downstream fine-tuning with ranking split")
    parser.add_argument('--train_path', type=str, required=True, help='Training CSV path')
    parser.add_argument('--test_path', type=str, required=True, help='Test/validation CSV path')
    parser.add_argument('--label_column', type=str, required=True, help='Label column name')
    parser.add_argument('--data_column', type=str, required=True, help='Sequence column name')
    parser.add_argument('--clip_direction', type=str, default='right', help="Truncation direction: 'left' or 'right'")

    parser.add_argument('--vocab_file', type=str, required=True, help='Nucleotide vocab file path')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--n_embd', type=int, default=384, help='GPT2 embedding dimension')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of Transformer layers')
    parser.add_argument('--n_head', type=int, default=16, help='Number of attention heads')

    parser.add_argument('--pretrain_ckpt', type=str, default=None, help='Optional: pretrained checkpoint path')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder parameters')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--regressor', type=int, default=1, help='Regressor head type (1=CNN, 2=ResNet, 3=MLP)')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device_tag', type=str, default='gpu', help='Device tag for naming')
    parser.add_argument('--wandb_project', type=str, default='MRL_Ranking', help='W&B project name')
    parser.add_argument('--output_dir', type=str, default='checkpoints_downstream', help='Output directory')
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = NucleotideTokenizer(vocab_file=args.vocab_file)

    train_sequences, train_labels = read_5putr_from_csv_regression_data(
        file_path=args.train_path,
        label_style=args.label_column,
        data_style=args.data_column,
        max_length=args.max_length,
        clip_direction=args.clip_direction
    )
    test_sequences, test_labels = read_5putr_from_csv_regression_data(
        file_path=args.test_path,
        label_style=args.label_column,
        data_style=args.data_column,
        max_length=args.max_length,
        clip_direction=args.clip_direction
    )

    train_tokenized = tokenize_sequences(train_sequences, tokenizer, args.max_length)
    test_tokenized = tokenize_sequences(test_sequences, tokenizer, args.max_length)

    train_dataset = mRNA_RegressionDataset(train_tokenized, train_labels)
    test_dataset = mRNA_RegressionDataset(test_tokenized, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    max_data_length = max(len(seq) for seq in train_sequences + test_sequences)
    max_data_length = min(max_data_length, args.max_length)

    model, _ = build_gpt2(tokenizer, args.n_embd, args.n_layer, args.n_head, args.max_length)
    if args.pretrain_ckpt is not None:
        pretrained_state = torch.load(args.pretrain_ckpt, map_location='cpu')
        model.load_state_dict(pretrained_state, strict=False)
        print(f"Loaded pretrained weights: {args.pretrain_ckpt}")
    regressor_class = get_regressor(args.regressor)
    lit_model = regressor_class(freeze_encoder=args.freeze_encoder, pretrained_model=model, lr=args.lr, log_name=None, input_len=max_data_length)
    train_base = os.path.splitext(os.path.basename(args.train_path))[0]
    file_name = train_base.split('_')[0]
    pretrain_tag = 'no_pretrain' if args.pretrain_ckpt is None else 'pretrained'
    freeze_tag = 'freeze' if args.freeze_encoder else 'unfreeze'
    output_name = f"{file_name}_{args.label_column}_{args.data_column}_seed{args.seed}_{pretrain_tag}_{freeze_tag}_epoch={args.epochs}_bs={args.batch_size}_lr={args.lr}_regressor={args.regressor}_{args.device_tag}"

    logger = WandbLogger(
        project=args.wandb_project,
        name=f"{output_name}_split",
        group=output_name,
        config={
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'label_column': args.label_column,
            'data_column': args.data_column,
            'file': file_name,
            'lr': args.lr,
            'n_embd': args.n_embd,
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'max_length': args.max_length,
            'pretrain': pretrain_tag,
            'regressor': args.regressor,
            'freeze_encoder': args.freeze_encoder
        },
        job_type='finetune',
        reinit=True
    )

    progress_bar = TQDMProgressBar(leave=True)

    split_dir = os.path.join(args.output_dir, output_name)
    os.makedirs(split_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=split_dir,
        filename=(
            f"gpt2-reg-{file_name}-{args.label_column}-{args.data_column}-{pretrain_tag}-"
            + "{epoch:02d}-{all_val_spearman:.4f}-{all_val_pearson:.4f}-{all_val_r2:.4f}-{all_val_rmse:.4f}-{all_val_mae:.4f}"
        ),
        monitor='all_val_spearman',
        mode='max',
        save_top_k=3
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=10
    )

    trainer.fit(lit_model, train_loader, test_loader)

    fold_metrics = {
        'r2': float(lit_model.best_r2),
        'pearson': float(lit_model.best_pearson),
        'spearman': float(lit_model.best_spearman),
        'rmse': float(lit_model.best_rmse),
        'mae': float(lit_model.best_mae)
    }
    logger.experiment.finish()

    df_folds = pd.DataFrame([fold_metrics])
    metrics_names = ['r2', 'pearson', 'spearman', 'rmse', 'mae']
    avg_metrics = {m: np.mean(df_folds[m].astype(float)) for m in metrics_names}
    std_metrics = {m: np.std(df_folds[m].astype(float)) for m in metrics_names}
    df_folds.loc['mean'] = pd.Series(avg_metrics)
    df_folds.loc['std'] = pd.Series(std_metrics)

    csv_path = os.path.join(args.output_dir, f"{output_name}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df_folds.to_csv(csv_path, index=True)

    print("Training finished. Metrics:", fold_metrics)
    print(f"CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
