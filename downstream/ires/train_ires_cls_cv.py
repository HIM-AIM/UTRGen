import os
import re
import random
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import GPT2Config, GPT2LMHeadModel

from downstream_classifier import LitDownstreamClassifier
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


def tokenize_sequences(sequences, tokenizer, max_length=100):
    tokenized_data = []
    for seq in sequences:
        if not seq:
            continue
        encoded = tokenizer(seq, max_length=max_length, padding='max_length', truncation=True, add_special_tokens=True)
        tokenized_data.append(encoded)
    return tokenized_data


def build_gpt2(tokenizer, n_ctx=256, n_embd=384, n_layer=8, n_head=16):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_ctx=n_ctx,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_ctx,
        pad_token_id=tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(config)
    return model, config


class IRESClassificationDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        assert len(tokenized_data) == len(labels)
        self.data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": label_tensor
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRES classification 10-fold CV training")
    parser.add_argument('--data_path', type=str, required=True, help='IRES CSV data path')
    parser.add_argument('--sequence_column', type=str, default='sequence', help='Sequence column name')
    parser.add_argument('--label_column', type=str, default='label', help='Label column name')
    parser.add_argument('--vocab_file', type=str, required=True, help='Nucleotide vocab file path')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_ratio', type=float, default=1.0, help='Ratio of training data to use (0.0-1.0)')
    parser.add_argument('--n_embd', type=int, default=384, help='GPT2 embedding dimension')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of Transformer layers')
    parser.add_argument('--n_head', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--n_ctx', type=int, default=256, help='Max sequence length')
    parser.add_argument('--wandb_project', type=str, default='IRES_CV_cls', help='W&B project name')
    parser.add_argument('--output_dir', type=str, default='checkpoints_ires', help='Output directory')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = NucleotideTokenizer(vocab_file=args.vocab_file)

    df = pd.read_csv(args.data_path)
    sequences = df[args.sequence_column].astype(str).tolist()
    labels = df[args.label_column].astype(int).tolist()

    tokenized_data = tokenize_sequences(sequences, tokenizer, max_length=args.n_ctx)
    dataset = IRESClassificationDataset(tokenized_data, labels)

    file_name = os.path.splitext(os.path.basename(args.data_path))[0]
    output_name = f"{file_name}_{args.label_column}_classification_cv_epoch={args.epochs}_bs={args.batch_size}_ratio={args.train_ratio}"

    kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nFold {fold+1}/10 CV started...")

        train_indices = train_idx.tolist()
        val_indices = val_idx.tolist()

        if args.train_ratio < 1.0:
            sample_size = int(len(train_indices) * args.train_ratio)
            rng = np.random.RandomState(args.seed + fold)
            train_indices = rng.choice(train_indices, size=sample_size, replace=False).tolist()
            print(f"Fold {fold+1}: Downsampled training set from {len(train_idx)} to {len(train_indices)} samples")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

        model, config = build_gpt2(tokenizer, n_ctx=args.n_ctx, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head)
        if args.pretrained and os.path.exists(args.pretrained):
            state = torch.load(args.pretrained, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            new_sd = {}
            for k, v in state.items():
                nk = k
                if nk.startswith('model.'):
                    nk = nk[6:]
                if nk.startswith('pretrained_model.'):
                    nk = nk[len('pretrained_model.'):]
                new_sd[nk] = v
            mk, uk = model.load_state_dict(new_sd, strict=False)
            print(f"Loaded pretrained weights. missing_keys={len(mk)}, unexpected_keys={len(uk)}")

        lit_model = LitDownstreamClassifier(pretrained_model=model, num_classes=2, freeze_encoder=False, lr=args.lr)

        fold_logger = WandbLogger(
            project=args.wandb_project,
            name=f"{output_name}_fold{fold + 1}",
            group=output_name,
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "train_ratio": args.train_ratio,
                "fold": fold + 1,
                "n_splits": 10,
                "label_column": args.label_column,
                "file": file_name,
            },
            job_type="finetune",
            reinit=True
        )

        progress_bar = TQDMProgressBar(leave=True)

        fold_dir = os.path.join(args.output_dir, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            filename=f"gpt2-cls-{file_name}-{args.label_column}-cv-{{epoch:02d}}-{{val_acc:.4f}}-{{val_auroc:.4f}}-{{val_aupr:.4f}}",
            monitor="val_acc",
            mode="max",
            save_top_k=1
        )

        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator="gpu",
            logger=fold_logger,
            callbacks=[checkpoint_callback, progress_bar],
            log_every_n_steps=10,
        )

        trainer.fit(lit_model, train_loader, val_loader)

        best_path = checkpoint_callback.best_model_path
        acc_match = re.search(r'val_acc=(\d+\.?\d*)', best_path)
        auroc_match = re.search(r'val_auroc=(\d+\.?\d*)', best_path)
        aupr_match = re.search(r'val_aupr=(\d+\.?\d*)', best_path)

        fold_acc = float(acc_match.group(1)) if acc_match else 0.0
        fold_auroc = float(auroc_match.group(1)) if auroc_match else 0.0
        fold_aupr = float(aupr_match.group(1)) if aupr_match else 0.0

        fold_metrics = {
            "acc": fold_acc,
            "auroc": fold_auroc,
            "aupr": fold_aupr
        }
        print(f"Fold {fold+1} best metrics: {fold_metrics}")
        fold_metrics_list.append(fold_metrics)

        fold_logger.experiment.finish()

    # Compute average and std across folds
    metrics_names = ["acc", "auroc", "aupr"]
    avg_metrics = {}
    std_metrics = {}

    for m in metrics_names:
        values = [fold[m] for fold in fold_metrics_list]
        avg_metrics[m] = np.mean(values)
        std_metrics[m] = np.std(values)

    print("\n10-fold CV final average metrics:")
    print(f"Mean ACC: {avg_metrics['acc']:.4f}   Std: {std_metrics['acc']:.4f}")
    print(f"Mean AUROC: {avg_metrics['auroc']:.4f}   Std: {std_metrics['auroc']:.4f}")
    print(f"Mean AUPR: {avg_metrics['aupr']:.4f}   Std: {std_metrics['aupr']:.4f}")

    df_folds = pd.DataFrame(fold_metrics_list)
    df_folds.loc['mean'] = pd.Series(avg_metrics)
    df_folds.loc['std'] = pd.Series(std_metrics)
    csv_path = os.path.join(args.output_dir, f"{output_name}.csv")
    df_folds.to_csv(csv_path, index=True)
    print(f"Results saved to: {csv_path}")
