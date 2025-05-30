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
from model.MDM_transformer import DDiTNoLengthModel
from interpolant import vanilla_MDM_Interpolant, ReparametrizedRate
from data.text import TEXT_DATASETS, setup_tokeniser, get_text_dataset
from data.parenthesis import BracketDataset
from schedule import GeometricSchedule, LinearSchedule
from datetime import datetime


class MDM(pl.LightningModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.learning_rate = config.training.learning_rate

        # Initialize model (no length head)
        self.model = DDiTNoLengthModel(config)

        # Initialize Masking Schedule
        if args.mask_schedule_type == "geometric":
            mask_schedule = GeometricSchedule(min=5.0, max=0.01)
        elif args.mask_schedule_type == "linear":
            mask_schedule = LinearSchedule()
        else:
            raise ValueError(f"Invalid mask schedule type: {args.mask_schedule_type}")

        # Initialize interpolant
        self.interpolant = vanilla_MDM_Interpolant(
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

    def training_loss(self, x1, t):
        # sample interpolant and elbo weight
        xt, _ , xt_mask_indices, _, _ = self.interpolant.new_sample_interpolant(t, x1)
        weight_unmask = self.interpolant.elbo_weight(t, x1)

        # model prediction
        pred_rate = self(xt, t)

        # compute unmask loss
        unmask_logits = pred_rate.per_token_posterior
        if pred_rate.length_posterior is not None:
            raise ValueError("Length posterior should be None for the vanilla MDM")
        
        # compute unmask loss
        loss = F.cross_entropy(unmask_logits[xt_mask_indices] , x1[xt_mask_indices], reduction='none') * weight_unmask[xt_mask_indices]
        loss = loss.sum() / (x1.shape[0] * self.config.interpolant.max_length)        
        return loss

    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)
        loss = self.training_loss(x1, t)

        self.log("train/total_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        batch_size = x1.shape[0]

        t = torch.rand(batch_size, device=x1.device)
        loss = self.training_loss(x1, t)

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
        config.interpolant.tokens = len(tokeniser)
        config.interpolant.pad_token = tokeniser.pad_token_id
        config.interpolant.mask_token = tokeniser.mask_token_id

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
    model = MDM(config, args)

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
        default="config/wikitext2/vanilla_MDM.yaml"
    )
    parser.add_argument("--mask_schedule_type", type=str, help="which mask schedule to use: currently supports geometric and linear", default = "geometric")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--wandb_entity", type=str, help="wandb entity", default = "jaeyeon_kim-harvard-university")

    args = parser.parse_args()
    train(args)