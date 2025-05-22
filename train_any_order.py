"""
Training script for Semi-Autoregressive Interpolant
"""

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.transformer import AnyOrderMaskInsertionFlow
from interpolant import AnyOrderMaskInsertionInterpolant, ReparametrizedRate
from parenthesis import BracketDataset
from bregman import mse
import os

from schedule import GeometricSchedule

# Configuration
config = OmegaConf.create(
    {
        "tokens": 3,  # 0, 1, 2
        "mask_token": 0,
        "pad_token": 3,
        "max_length": 64,
        "model": {
            "hidden_size": 256,
            "n_heads": 4,
            "cond_dim": 64,
            "dropout": 0.1,
            "n_blocks": 4,
        },
    }
)

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
CHECKPOINT_DIR = "checkpoints/bracket-flow"


class BracketFlowModule(pl.LightningModule):
    def __init__(self, config, learning_rate=LEARNING_RATE):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate

        # Initialize model
        self.model = AnyOrderMaskInsertionFlow(config)

        # Initialize Masking Schedule
        mask_schedule = GeometricSchedule(min=5, max=0.01)

        # Initialize interpolant
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            mask_schedule=mask_schedule,
            vocab_size=config.tokens,
            mask_token=config.mask_token,
            pad_token=config.pad_token,
            max_length=config.max_length,
        )

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, t) -> ReparametrizedRate:
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        x1 = batch
        batch_size = x1.shape[0]

        # Sample random time steps
        t = torch.rand(batch_size, device=x1.device)

        # Sample interpolant
        xt, st = self.interpolant.sample_interpolant(t, x1)

        # Get true conditional rate
        true_rate = self.interpolant.reparametrised_conditional_rate(xt, st, t, x1)

        # Get model predictions
        pred_rate: ReparametrizedRate = self(xt, t)

        # Compute losses
        unmask_loss = (
            mse(pred_rate.per_token_posterior, true_rate.per_token_posterior)
            / self.config.max_length
        )
        unmask_loss = unmask_loss.mean()
        len_loss = mse(pred_rate.length_posterior, true_rate.length_posterior)
        len_loss = len_loss.mean()
        loss = unmask_loss + len_loss
        loss = loss.mean()

        # Log losses
        self.log("train/unmask_loss", unmask_loss, prog_bar=True)
        self.log("train/len_loss", len_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Save additional information to checkpoint"""
        checkpoint["config"] = self.config
        checkpoint["interpolant_state"] = {
            "vocab_size": self.interpolant.vocab_size,
            "mask_token": self.interpolant.mask_token,
            "pad_token": self.interpolant.pad_token,
            "max_length": self.interpolant.max_length,
        }

    def on_load_checkpoint(self, checkpoint):
        """Load additional information from checkpoint"""
        self.config = checkpoint["config"]
        interpolant_state = checkpoint["interpolant_state"]
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            mask_schedule=GeometricSchedule(min=5.0, max=0.01),
            vocab_size=interpolant_state["vocab_size"],
            mask_token=interpolant_state["mask_token"],
            pad_token=interpolant_state["pad_token"],
            max_length=interpolant_state["max_length"],
        )


def train(resume_from_checkpoint=None):
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = BracketDataset(10000, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = BracketFlowModule(config)

    # Initialize trainer with enhanced checkpointing
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                monitor="train/total_loss",
                mode="min",
                save_top_k=3,
                filename="bracket-any-order-mask-flow-{epoch:02d}-{train/total_loss:.4f}",
                save_last=True,  # Always save the last checkpoint
                every_n_epochs=5,  # Save every epoch
            )
        ],
        log_every_n_steps=100,
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, dataloader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(resume_from_checkpoint=args.resume)
