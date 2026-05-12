# downstream_regressor.py
import os  
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from scipy.stats import spearmanr,pearsonr
import numpy as np
import json
from torchmetrics.regression import R2Score


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


def get_regressor(type: int):
    if type == 1:
        return LitDownstreamRegressor
    elif type == 3:
        return RegressorMLP
    elif type == 2:
        return RegressorResNet
    elif type == 4:
        return LitDownstreamClassifier
    else:
        raise ValueError("Invalid regressor type. Choose 1, 2, 3, or 4.")

class BaseRegressor(pl.LightningModule):
    """Base class that implements shared training/validation/logging logic."""

    def __init__(self, pretrained_model=None, freeze_encoder: bool = False, config=None, lr: float = 2e-4,
                 log_name: str | None = None, input_len: int = 256):
        super().__init__()
        if pretrained_model is None:
            assert config is not None, "Config must be provided when pre trained_model is not passed in"
            pretrained_model = GPT2LMHeadModel(config)

        self.pretrained_model = pretrained_model
        self.lr = lr
        self.input_len = input_len

        self.criterion = torch.nn.HuberLoss()

        self.train_losses = []
        self.val_preds = []
        self.val_labels = []
        self.val_r2 = R2Score()

        self.best_r2 = -float('inf')
        self.best_pearson = -float('inf')
        self.best_spearman = -float('inf')
        self.best_rmse = float('inf')
        self.best_mae = float('inf')
        self.log_name = (log_name or "run").split("epoch=")[0]

        if freeze_encoder:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False

    # ---------- hooks shared by all regressors ----------
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, labels)

        self.train_losses.append(loss.detach().cpu())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        preds = self(input_ids, attention_mask)
        loss = self.criterion(preds, labels)

        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if len(self.train_losses) > 0:
            avg_train_loss = torch.stack(self.train_losses).mean().detach().cpu().item()
            self.log("train_loss_epoch_avg", avg_train_loss, prog_bar=True, sync_dist=True)
            self.avg_train_loss_this_epoch = avg_train_loss
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0)).item()
        avg_train_loss = getattr(self, "avg_train_loss_this_epoch", 0.0)
        self._finalize_validation_epoch(avg_val_loss=avg_val_loss, avg_train_loss=avg_train_loss)

    def _finalize_validation_epoch(self, avg_val_loss: float, avg_train_loss: float):
        """Shared metrics computation and logging, called by on_validation_epoch_end of subclasses."""
        all_preds = torch.cat(self.val_preds)
        all_labels = torch.cat(self.val_labels)

        preds_np = all_preds.numpy().flatten()
        labels_np = all_labels.numpy().flatten()

        mae = F.l1_loss(all_preds, all_labels)
        mse = F.mse_loss(all_preds, all_labels)
        rmse_epoch = torch.sqrt(mse)

        try:
            spearman_corr = spearmanr(preds_np, labels_np)[0]
        except Exception:
            spearman_corr = 0.0
        try:
            pearson_corr = pearsonr(preds_np, labels_np)[0]
        except Exception:
            pearson_corr = 0.0

        self.val_r2.update(all_preds, all_labels)
        r2_score = self.val_r2.compute()

        if spearman_corr > self.best_spearman:
            self.best_epoch = self.current_epoch
            self.best_spearman = spearman_corr
            self.best_pearson = pearson_corr
            self.best_r2 = r2_score
            self.best_rmse = rmse_epoch
            self.best_mae = mae

        self.log("train_loss_epoch_avg", avg_train_loss, prog_bar=False)
        self.log("val_loss_epoch_avg", avg_val_loss, prog_bar=True)
        self.log("all_val_spearman", spearman_corr, prog_bar=True)
        self.log("all_val_pearson", pearson_corr, prog_bar=True)
        self.log("all_val_r2", r2_score, prog_bar=True)
        self.log("all_val_rmse", rmse_epoch, prog_bar=False)
        self.log("all_val_mae", mae, prog_bar=False)

        log_path = f"./log/{self.log_name}_epoch.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_line = (
            f"Epoch {self.current_epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Spearman: {spearman_corr:.4f} | "
            f"Pearson: {pearson_corr:.4f} | "
            f"R2: {r2_score:.4f} | "
            f"RMSE: {rmse_epoch:.4f} | "
            f"MAE: {mae:.4f}\n"
        )

        if self.current_epoch == 0 and not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("Epoch | Train Loss | Val Loss | Spearman | Pearson | R2 | RMSE | MAE\n")
        with open(log_path, "a") as f:
            f.write(log_line)

        self.val_preds.clear()
        self.val_labels.clear()
        self.val_r2.reset()

    def on_fit_end(self):
        print("\n best verification performance is as follows：")
        print(f"  best Spearman: {self.best_spearman:.4f}")
        print(f"  best Pearson : {self.best_pearson:.4f}")
        print(f"  best R²      : {self.best_r2:.4f}")
        print(f"  best RMSE    : {self.best_rmse:.4f}")
        print(f"  best MAE     : {self.best_mae:.4f}")
        best_epoch = getattr(self, "best_epoch", "unknow")
        print(f"  best Epoch   : {best_epoch}")

        result = {
            "best_spearman": float(self.best_spearman),
            "best_pearson": float(self.best_pearson),
            "best_r2": float(self.best_r2),
            "best_rmse": float(self.best_rmse),
            "best_mae": float(self.best_mae),
            "best_epoch": best_epoch
        }
        best_metric = f"./log/{self.log_name}_best_metrics.json"
        os.makedirs(os.path.dirname(best_metric), exist_ok=True)
        with open(best_metric, "w") as f:
            json.dump(result, f, indent=4)
        print("✅ best verification performance is save to final_metrics.json")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]


class LitDownstreamRegressor(BaseRegressor):
    def __init__(self, pretrained_model=None, freeze_encoder: bool = False, config=None, lr: float = 2e-4,
                 log_name: str | None = None, input_len: int = 128):
        """CNN-based regressor head."""
        super().__init__(pretrained_model=pretrained_model, freeze_encoder=freeze_encoder, config=config,
                         lr=lr, log_name=log_name, input_len=input_len)
        ##################################### CNN  ###################################
        
        self.filter_len = 8 
        self.nbr_filters = 64 
        self.dropout1 = 0.1  
        self.dropout2 = 0.1  
        
        
        self.conv1 = nn.Conv1d(in_channels=self.pretrained_model.config.hidden_size, 
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding='same')
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters, 
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding='same')
        
        self.relu = nn.ReLU()
        self.dropout1_layer = nn.Dropout(self.dropout1)
        self.dropout2_layer = nn.Dropout(self.dropout2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.nbr_filters * (self.input_len - 2), 1)

        ##################################### CNN  ###################################

        # ## 无用
        # self.regressor = nn.Sequential(
        #                 nn.Linear(pretrained_model.config.hidden_size, 256),
        #                 nn.ReLU(),
        #                 nn.Linear(256, 1)
        #             )

    def forward(self, input_ids, attention_mask):
        target_len = int(self.input_len)
        seq_len = int(input_ids.size(1))
        if seq_len < target_len:
            pad_length = target_len - seq_len
            padding = torch.full((input_ids.size(0), pad_length), self.pretrained_model.config.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros((attention_mask.size(0), pad_length), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        outputs = self.pretrained_model(
            input_ids=input_ids, # [B,L]
            attention_mask=attention_mask,# [B,L]
            output_hidden_states=True,
            # return_dict=True
        )
        ################################## extrac feature by CNN ####################################
        
        hidden_states = outputs.hidden_states[-1] 
        hidden_states = hidden_states[:,1:self.input_len-1,:]
        hidden_states = hidden_states.permute(0, 2, 1)

        # forward by CNN
        x_cnn1 = self.conv1(hidden_states)  
        x_relu1 = self.relu(x_cnn1)  
        x_dropout1 = self.dropout1_layer(x_relu1)  
        
        x_cnn2 = self.conv2(x_dropout1)  
        x_relu2 = self.relu(x_cnn2)  
        x_dropout2 = self.dropout2_layer(x_relu2)  
        
        
        x_flattened = self.flatten(x_dropout2)  
        output = self.fc(x_flattened)
        return output.squeeze(-1)

class LitDownstreamClassifier(BaseRegressor):
    def __init__(self, pretrained_model=None, num_classes=2, freeze_encoder: bool = False, config=None, lr: float = 2e-4,
                 log_name: str | None = None, input_len: int = 256):
        """CNN-based regressor head."""
        super().__init__(pretrained_model=pretrained_model, freeze_encoder=freeze_encoder, config=config,
                         lr=lr, log_name=log_name, input_len=input_len)
        ##################################### CNN  ###################################
        
        self.filter_len = 8 
        self.nbr_filters = 64 
        self.dropout1 = 0.1  
        self.dropout2 = 0.1  
        
        
        self.conv1 = nn.Conv1d(in_channels=self.pretrained_model.config.hidden_size, 
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding='same')
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters, 
                               out_channels=self.nbr_filters, kernel_size=self.filter_len, padding='same')
        
        self.relu = nn.ReLU()
        self.dropout1_layer = nn.Dropout(self.dropout1)
        self.dropout2_layer = nn.Dropout(self.dropout2)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.nbr_filters * (self.input_len - 2), num_classes)

        ##################################### CNN  ###################################

    def forward(self, input_ids, attention_mask):
        target_len = int(self.input_len)
        seq_len = int(input_ids.size(1))
        if seq_len < target_len:
            pad_length = target_len - seq_len
            padding = torch.full((input_ids.size(0), pad_length), self.pretrained_model.config.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, padding], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros((attention_mask.size(0), pad_length), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        outputs = self.pretrained_model(
            input_ids=input_ids, # [B,L]
            attention_mask=attention_mask,# [B,L]
            output_hidden_states=True,
            # return_dict=True
        )

        hidden_states = outputs.hidden_states[-1] 
        hidden_states = hidden_states[:,1:self.input_len-1,:]
        hidden_states = hidden_states.permute(0, 2, 1)

        x_cnn1 = self.conv1(hidden_states)
        x_relu1 = self.relu(x_cnn1)
        x_dropout1 = self.dropout1_layer(x_relu1)
        x_cnn2 = self.conv2(x_dropout1)
        x_relu2 = self.relu(x_cnn2)
        x_dropout2 = self.dropout2_layer(x_relu2)
        x_flattened = self.flatten(x_dropout2)
        fc = self.fc(x_flattened)
        return fc


class RegressorMLP(BaseRegressor):
    def __init__(self, pretrained_model=None, freeze_encoder: bool = False, config=None, lr: float = 2e-4,
                 log_name: str | None = None, input_len: int = 256):
        super().__init__(pretrained_model=pretrained_model, freeze_encoder=freeze_encoder, config=config,
                         lr=lr, log_name=log_name, input_len=input_len)
        ##################################### MLP  ###################################
        
        self.hidden_size = self.pretrained_model.config.n_embd
        
        self.regressor = nn.Sequential(
                        nn.Linear(self.hidden_size, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )
        ##################################### MLP  ###################################


    def forward(self, input_ids, attention_mask):
        
        
        outputs = self.pretrained_model(
            input_ids=input_ids, # [B,L]
            attention_mask=attention_mask,# [B,L]
            output_hidden_states=True,
            # return_dict=True
        )
        ################################## extrac feature by MLP ####################################
        
        last_hidden = outputs.hidden_states[-1]
        pooled_output = last_hidden.mean(dim=1)
        prediction = self.regressor(pooled_output)  # shape: [batch_size, 1]
        return prediction.squeeze(-1)


class RegressorResNet(BaseRegressor):
    def __init__(self, pretrained_model=None, freeze_encoder: bool = False, config=None, lr: float = 2e-4,
                 log_name: str | None = None, input_len: int = 256):
        super().__init__(pretrained_model=pretrained_model, freeze_encoder=freeze_encoder, config=config,
                         lr=lr, log_name=log_name, input_len=input_len)

        # ---------- ResNet18-style 1D head ----------
        in_channels = self.pretrained_model.config.hidden_size
        base_channels = 64

        class BasicBlock1D(nn.Module):
            expansion = 1
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,
                                       padding=1, bias=False)
                self.bn1 = nn.BatchNorm1d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                                       padding=1, bias=False)
                self.bn2 = nn.BatchNorm1d(planes)

                self.downsample = None
                if stride != 1 or in_planes != planes * self.expansion:
                    self.downsample = nn.Sequential(
                        nn.Conv1d(in_planes, planes * self.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(planes * self.expansion)
                    )

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)
                return out

        def make_layer(in_planes, planes, num_blocks, stride):
            layers = []
            layers.append(BasicBlock1D(in_planes, planes, stride=stride))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock1D(planes, planes, stride=1))
            return nn.Sequential(*layers)

        self.resnet_conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_bn1 = nn.BatchNorm1d(base_channels)
        self.resnet_relu = nn.ReLU(inplace=True)
        self.resnet_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet18: [2,2,2,2] blocks
        self.layer1 = make_layer(base_channels, 64, 2, stride=1)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(512, 1)
        # ---------- ResNet18 head end ----------

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # [B, T, C]
        hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states[:, 1:self.input_len-1, :]   # [B, T-2, C]
        hidden_states = hidden_states.permute(0, 2, 1)            # [B, C, T-2]

        x = self.resnet_conv1(hidden_states)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)   # [B, 512, 1]
        x = x.squeeze(-1)         # [B, 512]
        out = self.fc_out(x)      # [B, 1]
        return out.squeeze(-1)
