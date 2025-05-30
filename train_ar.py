import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os
import argparse
from omegaconf import OmegaConf
import wandb
import torch.nn.functional as F
from model.casual_transformer import CausalDiT
from data.text import TEXT_DATASETS, setup_tokeniser, get_text_dataset
from data.parenthesis import BracketDataset
from datetime import datetime


class Autoregressive(pl.LightningModule):
    def __init__(self, config, args):
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
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=self.config.interpolant.pad_token)
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


def train(args):
    # set the random seed
    pl.seed_everything(42)
    torch.manual_seed(42)

    # Load config
    config = OmegaConf.load(args.config)
    if args.wandb:
        wandb.init(project="interpretable-flow", entity=args.wandb_entity, config=OmegaConf.to_container(config, resolve=True),
            name=os.path.basename(args.config),
        )
        wandb_logger = WandbLogger(
            project=wandb.run.project,
            name=wandb.run.name,
            log_model=True,
        )
    else:
        wandb_logger = None

    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")

    config.training.checkpoint_dir = os.path.join(
        config.training.checkpoint_dir, time_string
    )

    # Create checkpoint directory
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Initialize dataset and dataloader

    if config.dataset in TEXT_DATASETS:
        tokeniser = setup_tokeniser()
        train_set = get_text_dataset(config.dataset, split="train", max_length=config.interpolant.max_length)
        val_set = get_text_dataset(config.dataset, split="validation", max_length=config.interpolant.max_length)
        train_loader = DataLoader(
            train_set, batch_size=config.training.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=config.training.batch_size, shuffle=False
        )
        # there is no interpolant for AR, but just match the notation
        config.interpolant.tokens = len(tokeniser)
        config.interpolant.pad_token = tokeniser.pad_token_id

    if config.dataset in ["bracket"]:
        train_loader = DataLoader(
            BracketDataset(10000, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1}),
            batch_size=config.training.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            BracketDataset(300, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1}),
            batch_size=config.training.batch_size,
            shuffle=False,
        )

    # Initialize model
    model = Autoregressive(config, args)

    # Initialize trainer
    # TODO: add gradient clipping / learning rate scheduler / gradient accumulation
    trainer_kwargs = dict(
        max_epochs=config.training.num_epochs,
        accelerator="gpu",
        devices=config.training.devices,
        log_every_n_steps=100,
        enable_checkpointing=True,
    )
    if wandb_logger is not None:
        trainer_kwargs["logger"] = wandb_logger
    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    if args.wandb:
        wandb.finish()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="config/wikitext2/autoregressive.yaml"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--wandb_entity", type=str, help="wandb entity", default = "jaeyeon_kim-harvard-university")

    args = parser.parse_args()
    train(args)