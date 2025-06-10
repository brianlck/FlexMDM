import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import os
import argparse
from omegaconf import OmegaConf
import wandb
import torch.nn.functional as F
from model.transformer import AnyOrderMaskInsertionFlow
from interpolant import AnyOrderMaskInsertionInterpolant, ModelPrediction
from data.text import TEXT_DATASETS, setup_tokeniser, get_text_dataset
from data.parenthesis import BracketDataset
from bregman import jump_kernel_elbo, mse
from schedule import GeometricSchedule, LinearSchedule
from datetime import datetime


class TransdimensionalFlowModule(pl.LightningModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.model_type = config.interpolant.type
        self.learning_rate = config.training.learning_rate
        self.unmask_loss_type = args.unmask_loss_type
        self.len_predict_type = args.len_predict_type
        self.len_loss_scheduler = args.len_loss_scheduler

        # Initialize model based on type
        self.model = AnyOrderMaskInsertionFlow(config, self.len_predict_type)

        # Initialize Masking Schedule
        if args.mask_schedule_type == "geometric":
            mask_schedule = GeometricSchedule(min_val=5.0, max_val=0.01)
        elif args.mask_schedule_type == "linear":
            mask_schedule = LinearSchedule()
        else:
            raise ValueError(f"Invalid mask schedule type: {args.mask_schedule_type}")

        # Initialize interpolant
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            mask_schedule=mask_schedule,
            vocab_size=config.interpolant.tokens,
            mask_token=config.interpolant.mask_token,
            pad_token=config.interpolant.pad_token,
            max_length=config.interpolant.max_length,
        )

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, t) -> ModelPrediction:
        return self.model(x, t)

    def training_loss(self, x1, t):
        interpolant_sample = self.interpolant.sample_interpolant(t, x1)
        unmask_weight, insert_weight = self.interpolant.elbo_weight(t, x1)

        prediction: ModelPrediction = self(interpolant_sample.xt, t)

        match self.unmask_loss_type:
            case "elbo":
                mask_indices = interpolant_sample.mask_indices
                unmask_loss = unmask_weight * F.cross_entropy(
                    prediction.token_posterior[mask_indices],
                    interpolant_sample.unmasked[mask_indices],
                    reduction="none",
                )
                unmask_loss = unmask_loss.mean()

            case _:
                raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_type}")

        match self.len_predict_type:
            case "expectation":
                insertion_loss = insert_weight * jump_kernel_elbo(
                    prediction.expected_gaps, interpolant_sample.gaps
                )
                insertion_loss = insertion_loss.mean()

            case "distribution":
                gap_one_hot = F.one_hot(
                    interpolant_sample.gaps,
                    num_classes=self.config.interpolant.max_length + 1,
                ).to(prediction.length_posterior.dtype)
                insertion_loss = insert_weight * mse(
                    prediction.length_posterior, gap_one_hot
                )
                insertion_loss = insertion_loss.mean()

        unmask_loss + insertion_loss
        return unmask_loss, insertion_loss

    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)

        unmask_loss, len_loss = self.training_loss(x1, t)

        if self.len_loss_scheduler:
            warmup_steps = self.config.training.warmup_steps
            alpha = float(self.global_step) / float(warmup_steps)
            alpha = max(0.0, min(1.0, alpha))
        else:
            alpha = 1.0
        
        loss = unmask_loss + alpha * len_loss

        self.log("train/unmask_loss", unmask_loss, prog_bar=True)
        self.log("train/len_loss", len_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        batch_size = x1.shape[0]

        t = torch.rand(batch_size, device=x1.device)
        unmask_loss, len_loss = self.training_loss(x1, t)

        # no scheduler for validation
        loss = unmask_loss + len_loss

        self.log("val/unmask_loss", unmask_loss, prog_bar=True)
        self.log("val/len_loss", len_loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        warmup_steps = self.config.training.warmup_steps
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor = 0.01, end_factor = 1.0, total_iters = warmup_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config
        checkpoint["interpolant_state"] = {
            "vocab_size": self.interpolant.vocab_size,
            "mask_token": self.interpolant.mask_token,
            "pad_token": self.interpolant.pad_token,
            "max_length": self.interpolant.max_length,
        }

    def on_load_checkpoint(self, checkpoint):
        # TODO: work with general mask schedule
        self.config = checkpoint["config"]
        interpolant_state = checkpoint["interpolant_state"]

        self.interpolant = AnyOrderMaskInsertionInterpolant(
            mask_schedule=GeometricSchedule(min=5.0, max=0.01),
            vocab_size=interpolant_state["vocab_size"],
            mask_token=interpolant_state["mask_token"],
            pad_token=interpolant_state["pad_token"],
            max_length=interpolant_state["max_length"],
        )


def train(args):
    # set the random seed
    pl.seed_everything(42)
    # Load config
    config = OmegaConf.load(args.config)
    wandb_logger = None
    
    @rank_zero_only
    def _init_wandb():
        nonlocal wandb_logger
        if args.wandb:
            wandb.init(project="interpretable-flow", entity=args.wandb_entity, config=OmegaConf.to_container(config, resolve=True),
                name=f"VLMDM_{args.unmask_loss_type}_{args.len_predict_type}_{args.mask_schedule_type}_{args.len_loss_scheduler}",
            )
            wandb_logger = WandbLogger(
                project=wandb.run.project,
                name=wandb.run.name,
                log_model=True,
            )
        else:
            wandb_logger = None

    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    _init_wandb()

    config.training.checkpoint_dir = os.path.join(
        config.training.checkpoint_dir, time_string
    )

    # Create checkpoint directory
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # Initialize dataset and dataloader
    if config.dataset in TEXT_DATASETS:
        tokeniser = setup_tokeniser()
        train_set = get_text_dataset(
            config.dataset, split="train", max_length=config.interpolant.max_length
        )
        val_set = get_text_dataset(
            config.dataset, split="validation", max_length=config.interpolant.max_length
        )
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

    # calculate max steps
    print(len(train_loader))
    total_iters = config.training.num_epochs * len(train_loader)
    config.training.warmup_steps = int(total_iters * 0.01)
    
    # Initialize model
    model = TransdimensionalFlowModule(config, args)

    # Initialize trainer
    # TODO: add gradient clipping / learning rate scheduler
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
        @rank_zero_only
        def _finish_wandb():
            wandb.finish()
        _finish_wandb()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="config/wikitext2/any_order.yaml",
    )
    parser.add_argument(
        "--interpolant_type",
        type=str,
        help="which interpolant to use: currently just supports any-order",
        default="any-order",
    )
    parser.add_argument(
        "--mask_schedule_type",
        type=str,
        help="which mask schedule to use: currently supports geometric and linear",
        default="geometric",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--unmask_loss_type",
        type=str,
        help="which unmask loss to use: cross entropy or mse",
        default="ce",
    )
    parser.add_argument(
        "--len_predict_type",
        type=str,
        help="which length prediction to use: distribution (p(s_t | x_t, t)) or expectation (E[s_t | x_t, t])",
        default="distribution",
    )
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--len_loss_scheduler", action = "store_true", help = "whether to use len loss scheduler")
    parser.add_argument("--wandb_entity", type=str, help="wandb entity", default = "jaeyeon_kim-harvard-university")

    args = parser.parse_args()
    train(args)
