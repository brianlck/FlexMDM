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
from model.transformer import SemiAutoregressiveFlow, AnyOrderMaskInsertionFlow
from interpolant import (
    SemiAutoregressiveInterpolant,
    AnyOrderMaskInsertionInterpolant,
    ReparametrizedRate,
)
from data.text import TEXT_DATASETS, setup_tokeniser, get_text_dataset
from data.parenthesis import BracketDataset
from bregman import mse, scalar_bregman
from schedule import GeometricSchedule, LinearSchedule
from datetime import datetime


class TransdimensionalFlowModule(pl.LightningModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.model_type = config.interpolant.type
        self.learning_rate = config.training.learning_rate
        self.loss_brian = args.loss_brian
        self.unmask_loss_type = args.unmask_loss_type
        self.len_predict_type = args.len_predict_type

        # Initialize model based on type
        match config.interpolant.type:
            case "semi-auto":
                self.model = SemiAutoregressiveFlow(config)
                interpolant_class = SemiAutoregressiveInterpolant
            case "any-order":
                self.model = AnyOrderMaskInsertionFlow(config, self.len_predict_type)
                interpolant_class = AnyOrderMaskInsertionInterpolant

        # Initialize Masking Schedule
        if args.mask_schedule_type == "geometric":
            mask_schedule = GeometricSchedule(min=5.0, max=0.01)
        elif args.mask_schedule_type == "linear":
            mask_schedule = LinearSchedule()
        else:
            raise ValueError(f"Invalid mask schedule type: {args.mask_schedule_type}")

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

    def brian_training_loss(self, x1, t):
        xt, st = self.interpolant.sample_interpolant(t, x1)
        print(x1.shape, xt.shape)
        true_rate = self.interpolant.reparametrised_conditional_rate(xt, st, t, x1)

        # model prediction
        pred_rate = self(xt, t)

        # compute unmask loss
        unmask_loss = (
            mse(pred_rate.per_token_posterior.softmax(dim=-1), true_rate.per_token_posterior)
            / self.config.interpolant.max_length
        ).mean()
        len_loss = mse(pred_rate.length_posterior, true_rate.length_posterior).mean()

        return unmask_loss, len_loss, unmask_loss + len_loss
    
    def training_loss(self, x1, t):
        # forward pass
        batch_size = x1.shape[0]
        xt, st, xt_mask_indices, x1_remained, gap_counts = self.interpolant.new_sample_interpolant(t, x1)
        weight_unmask, weight_delete = self.interpolant.elbo_weight(t, x1)

        # model prediction
        pred_rate = self(xt, t)

        # compute unmask loss
        unmask_logits = pred_rate.per_token_posterior
        if self.unmask_loss_type == 'ce':
            unmask_loss = F.cross_entropy(unmask_logits[xt_mask_indices], x1_remained[xt_mask_indices], reduction='none') * weight_unmask[xt_mask_indices]
            unmask_loss = unmask_loss.sum() / (batch_size * self.config.interpolant.max_length)
        elif self.unmask_loss_type == 'mse':
            # make x1_remained one-hot
            x1_one_hot = F.one_hot(x1_remained, num_classes=self.config.interpolant.tokens)
            x1_one_hot = x1_one_hot.to(unmask_logits.dtype)
            unmask_prob = unmask_logits.softmax(dim=-1)
            # calculates the loss only for the masked indices
            unmask_loss = mse(unmask_prob[xt_mask_indices], x1_one_hot[xt_mask_indices]).mean()

            # # calculates the loss for all indices
            # x1_one_hot = x1_one_hot * xt_mask_indices.unsqueeze(-1)
            # unmask_loss = ( mse(unmask_prob , x1_one_hot) / self.config.interpolant.max_length ).mean()
        else:
            raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_type}")

        # compute length loss
        len_posterior = pred_rate.length_posterior
        if self.len_predict_type == "distribution":
            gap_one_hot = F.one_hot(gap_counts, num_classes=self.config.interpolant.max_length + 1)
            gap_one_hot = gap_one_hot.to(len_posterior.dtype)
            len_loss = mse(len_posterior, gap_one_hot).mean()
        elif self.len_predict_type == "expectation":
            # gap_counts: B X (L+1), len_posterior: B X (L+1), weight_delete: B X (L+1)
            len_loss = scalar_bregman(gap_counts.to(len_posterior.dtype), len_posterior) * weight_delete
            len_loss = len_loss.mean()
        else:
            raise ValueError(f"Invalid length prediction type: {self.len_predict_type}")
        
        return unmask_loss, len_loss, unmask_loss + len_loss

    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)
        if self.loss_brian:
            unmask_loss, len_loss, loss = self.brian_training_loss(x1, t)
        else:
            unmask_loss, len_loss, loss = self.training_loss(x1, t)

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
        if self.loss_brian:
            unmask_loss, len_loss, loss = self.brian_training_loss(x1, t)
        else:
            unmask_loss, len_loss, loss = self.training_loss(x1, t)

        self.log("val/unmask_loss", unmask_loss, prog_bar=True)
        self.log("val/len_loss", len_loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

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
        # TODO: work with general mask schedule
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
        wandb.finish()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        default="config/wikitext2/any_order.yaml"
    )
    parser.add_argument("--interpolant_type", type=str, help="which interpolant to use: currently just supports any-order", default = "any-order")
    parser.add_argument("--mask_schedule_type", type=str, help="which mask schedule to use: currently supports geometric and linear", default = "geometric")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--loss_brian", type=bool, help="whether to use brian's loss", default = False)
    parser.add_argument("--unmask_loss_type", type=str, help="which unmask loss to use: cross entropy or mse", default = "ce")
    parser.add_argument("--len_predict_type", type=str, help="which length prediction to use: distribution (p(s_t | x_t, t)) or expectation (E[s_t | x_t, t])", default = "distribution")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--wandb_entity", type=str, help="wandb entity", default = "jaeyeon_kim-harvard-university")

    args = parser.parse_args()
    train(args)
