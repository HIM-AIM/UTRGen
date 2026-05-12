import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel
from torchmetrics import AUROC, AveragePrecision
from torchmetrics.classification import Accuracy

class LitDownstreamClassifier(pl.LightningModule):
    def __init__(self, pretrained_model: GPT2LMHeadModel, num_classes: int = 2, freeze_encoder: bool = False, lr: float = 1e-3):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.lr = lr
        self.input_len = getattr(self.pretrained_model.config, "n_ctx", 128)
        self.criterion = nn.CrossEntropyLoss()
        if freeze_encoder:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False
        self.hidden_size = self.pretrained_model.config.n_embd
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        self.filter_len = 8
        self.nbr_filters = 64
        self.dropout1 = 0.1
        self.dropout2 = 0.1
        self.conv1 = nn.Conv1d(in_channels=self.pretrained_model.config.hidden_size, out_channels=self.nbr_filters, kernel_size=self.filter_len, padding='same')
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters, out_channels=self.nbr_filters, kernel_size=self.filter_len, padding='same')
        self.relu = nn.ReLU()
        self.dropout1_layer = nn.Dropout(self.dropout1)
        self.dropout2_layer = nn.Dropout(self.dropout2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.nbr_filters * (self.input_len - 2), num_classes)
        self.auroc = AUROC(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.aupr = AveragePrecision(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_preds = []
        self.val_labels = []

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states[:, 1:self.input_len - 1, :]
        hidden_states = hidden_states.permute(0, 2, 1)
        x_cnn1 = self.conv1(hidden_states)
        x_relu1 = self.relu(x_cnn1)
        x_dropout1 = self.dropout1_layer(x_relu1)
        x_cnn2 = self.conv2(x_dropout1)
        x_relu2 = self.relu(x_cnn2)
        x_dropout2 = self.dropout2_layer(x_relu2)
        x_flattened = self.flatten(x_dropout2)
        output = self.fc(x_flattened)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.acc(preds, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.auroc(probs, labels)
        self.aupr(probs, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.acc, prog_bar=True, on_epoch=True)
        self.log("val_auroc", self.auroc, prog_bar=True, on_epoch=True)
        self.log("val_aupr", self.aupr, prog_bar=True, on_epoch=True)
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0:
            all_preds = torch.cat(self.val_preds)
            all_labels = torch.cat(self.val_labels)
            acc_global = (all_preds == all_labels).float().mean()
            self.log("val_acc_global", acc_global, prog_bar=True)
            self.val_preds.clear()
            self.val_labels.clear()
            # Lightning 会自动管理 metric 的 reset，无需手动调用，否则会导致 Sanity Check 报错
            # self.auroc.reset()
            # self.aupr.reset()
            # self.acc.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params': self.pretrained_model.parameters(), 'lr': 3e-5},
            {'params': self.regressor.parameters(), 'lr': self.lr}
        ], weight_decay=0.01)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
