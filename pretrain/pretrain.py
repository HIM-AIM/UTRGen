import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from transformers import GPT2LMHeadModel, GPT2Config
from nucleotide_tokenizer import NucleotideTokenizer
from dataset import mRNA_Dataset
from pytorch_lightning.loggers import WandbLogger
import argparse
import wandb


class LitUTRGenPretrain(pl.LightningModule):
    def __init__(self, model, learning_rate, pad_token_id, t_max):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.learning_rate = learning_rate
        self.t_max = t_max

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        val_loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.t_max),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]


def create_hf_model(args, tokenizer):
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.n_ctx,
        n_embd=args.n_embeding,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(model_config)
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    return model


def main():
    parser = argparse.ArgumentParser(description="UTRGen Pre-training Script")
    # Model Config
    parser.add_argument('--n_embeding', type=int, default=384, help='Embedding dimension (n_embd)')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of Transformer layers')
    parser.add_argument('--n_head', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--n_ctx', type=int, default=256, help='Context length (n_positions)')
    # Training Config
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--lr_t_max', type=int, default=100, help='T_max for CosineAnnealingLR')
    # Paths and Logging
    parser.add_argument('--vocab_file', type=str, required=True, help='Path to vocab.txt')
    parser.add_argument('--train_files', nargs='+', required=True, help='Training FASTA file(s)')
    parser.add_argument('--test_files', nargs='+', required=True, help='Validation FASTA file(s)')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints_pretrain", help='Checkpoint output directory')
    # Resume functionality
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID to resume logging')

    args = parser.parse_args()

    tokenizer = NucleotideTokenizer(vocab_file=args.vocab_file)

    train_dataset = mRNA_Dataset(file_paths=args.train_files, tokenizer=tokenizer, max_length=args.n_ctx)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    test_dataset = mRNA_Dataset(file_paths=args.test_files, tokenizer=tokenizer, max_length=args.n_ctx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16, pin_memory=True)

    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
    wandb_logger = None

    if is_main_process:
        wandb_run_name = f"UTRGen_{args.n_layer}layer_{args.n_head}head_{args.n_embeding}embed_length{args.n_ctx}_{args.lr}lr"

        config_for_wandb = vars(args).copy()
        config_for_wandb.pop('resume_from_checkpoint', None)
        config_for_wandb.pop('wandb_run_id', None)

        init_args = {
            "project": "gpt2-pretrain",
            "name": wandb_run_name,
            "config": config_for_wandb
        }
        if args.wandb_run_id:
            init_args["id"] = args.wandb_run_id
            init_args["resume"] = "must"

        wandb.init(**init_args)
        wandb_logger = WandbLogger(log_model=True)

    model = create_hf_model(args, tokenizer)
    lit_model = LitUTRGenPretrain(model, learning_rate=args.lr, pad_token_id=tokenizer.pad_token_id, t_max=args.lr_t_max)

    checkpoint_filename = f"UTRGen-{args.n_layer}layer-{args.n_head}head-{args.n_embeding}embd-length{args.n_ctx}-{args.lr}lr-{{epoch:02d}}-{{val_loss:.4f}}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=checkpoint_filename,
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(
        max_epochs=args.epoch,
        callbacks=[checkpoint_callback, TQDMProgressBar()],
        accelerator="gpu",
        devices=args.n_gpu,
        logger=wandb_logger,
        log_every_n_steps=50
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=args.resume_from_checkpoint)

    if is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
