import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import argparse
from omegaconf import OmegaConf

from model.transformer import SemiAutoregressiveFlow, AnyOrderMaskInsertionFlow
from interpolant import (
    SemiAutoregressiveInterpolant,
    AnyOrderMaskInsertionInterpolant,
    ReparametrizedRate,
)
from data.parenthesis import BracketDataset
from bregman import mse
from schedule import GeometricSchedule


class BracketFlowModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.interpolant.type
        self.learning_rate = config.training.learning_rate

        # Initialize model based on type
        match config.interpolant.type:
            case "semi-auto":
                self.model = SemiAutoregressiveFlow(config.model)
                interpolant_class = SemiAutoregressiveInterpolant
            case "any-order":
                self.model = AnyOrderMaskInsertionFlow(config.model)
                interpolant_class = AnyOrderMaskInsertionInterpolant

        # Initialize Masking Schedule
        mask_schedule = GeometricSchedule(
            min=config.interpolant.mask_schedule.min,
            max=config.interpolant.mask_schedule.max,
        )

        # Initialize interpolant
        self.interpolant = interpolant_class(
            mask_schedule=mask_schedule,
            vocab_size=config.interpolant.tokens,
            mask_token=config.interpolant.mask_token,
            pad_token=config.interpolant.pad_token,
            max_length=config.interpolant.max_length,
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
            / self.config.interpolant.max_length
        )
        unmask_loss = unmask_loss.mean()
        len_loss = mse(pred_rate.length_posterior, true_rate.length_posterior)
        len_loss = len_loss.mean()
        loss = unmask_loss + len_loss

        # Log losses
        self.log("train/unmask_loss", unmask_loss, prog_bar=True)
        self.log("train/len_loss", len_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config
        checkpoint["interpolant_state"] = {
            "vocab_size": self.interpolant.vocab_size,
            "mask_token": self.interpolant.mask_token,
            "pad_token": self.interpolant.pad_token,
            "max_length": self.interpolant.max_length,
        }

    def on_load_checkpoint(self, checkpoint):
        self.config = checkpoint["config"]
        interpolant_state = checkpoint["interpolant_state"]

        interpolant_class = (
            SemiAutoregressiveInterpolant
            if self.config.interpolant.type == "semi-auto"
            else AnyOrderMaskInsertionInterpolant
        )

        self.interpolant = interpolant_class(
            mask_schedule=GeometricSchedule(min=5.0, max=0.01),
            vocab_size=interpolant_state["vocab_size"],
            mask_token=interpolant_state["mask_token"],
            pad_token=interpolant_state["pad_token"],
            max_length=interpolant_state["max_length"],
        )


def train(config_path, resume_from_checkpoint=None):
    # Load config
    config = OmegaConf.load(config_path)

    # Create checkpoint directory
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = BracketDataset(10000, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1})
    dataloader = DataLoader(
        dataset, batch_size=config.training.batch_size, shuffle=True
    )

    # Initialize model
    model = BracketFlowModule(config)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=config.training.checkpoint_dir,
                monitor="train/total_loss",
                mode="min",
                save_top_k=3,
                filename=f"bracket-{config.interpolant.type}"
                + "-{epoch:02d}-{train / total_loss:.4f}",
                save_last=True,
                every_n_epochs=5,
            )
        ],
        log_every_n_steps=100,
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, dataloader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args.config, args.resume)
