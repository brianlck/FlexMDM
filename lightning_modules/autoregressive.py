import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.casual_transformer import CausalDiT


class AutoregressiveModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config.training.learning_rate

        # Initialize model (causal transformer)
        self.model = CausalDiT(config)

    def forward(self, x):
        return self.model(x)

    def training_loss(self, x1):
        # next token prediction loss
        input_ids = x1[:, :-1]
        logits = self.model(input_ids)
        target_ids = x1[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=self.config.interpolant.pad_token,
        )
        return loss

    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        loss = self.training_loss(x1)

        self.log("train/total_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        loss = self.training_loss(x1)

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint):
        self.config = checkpoint["config"]
