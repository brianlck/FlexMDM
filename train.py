import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import wandb
import torch.nn.functional as F
from model.transformer import AnyOrderMaskInsertionFlow
from interpolant import AnyOrderMaskInsertionInterpolant, ModelPrediction
from bregman import jump_kernel_elbo, mse
from schedule import get_schedule_from_config
import data


class TransdimensionalFlowModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model_type = config.interpolant.type
        self.learning_rate = config.training.learning_rate
        self.unmask_loss_fn = config.training.loss_fn.unmask
        self.insert_loss_fn = config.training.loss_fn.insert

        # Initialize model based on type
        self.model = AnyOrderMaskInsertionFlow(config, self.insert_loss_fn)

        schedule = get_schedule_from_config(config.interpolant.mask_schedule)

        # Initialize interpolant
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            insertion_schedule=schedule,
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

        match self.unmask_loss_fn:
            case "elbo":
                mask_indices = interpolant_sample.mask_indices
                unmask_loss = unmask_weight[mask_indices] * F.cross_entropy(
                    prediction.token_posterior[mask_indices],
                    interpolant_sample.unmasked[mask_indices],
                    reduction="none",
                )
                unmask_loss = unmask_loss.mean()

            case _:
                raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_fn}")

        match self.insert_loss_fn:
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

        total_loss = unmask_loss + insertion_loss
        return unmask_loss, insertion_loss, total_loss

    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)
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
        unmask_loss, len_loss, loss = self.training_loss(x1, t)

        self.log("val/unmask_loss", unmask_loss, prog_bar=True)
        self.log("val/len_loss", len_loss, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config

    def on_load_checkpoint(self, checkpoint):
        self.config = checkpoint["config"]

        self.interpolant = AnyOrderMaskInsertionInterpolant(
            mask_schedule=get_schedule_from_config(
                self.config.interpolant.mask_schedule
            ),
            mask_token=self.config.interpolant.mask_token,
            pad_token=self.config.interpolant.pad_token,
            max_length=self.config.interpolant.max_length,
        )


def train(config: DictConfig):
    # set the random seed
    pl.seed_everything(42)
    torch.manual_seed(42)

    if "wandb" in config:
        wandb.init(
            project="interpretable-flow",
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config, resolve=True),
            name=config.wandb.name,
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

    dataset_bundle = data.setup_data_and_update_config(config)
    model = TransdimensionalFlowModule(config)

    # Initialize trainer
    # TODO: add gradient clipping / learning rate scheduler
    trainer_kwargs = dict(
        max_epochs=config.training.num_epochs,
        accelerator="gpu",
        devices=config.training.devices,
        log_every_n_steps=100,
        enable_checkpointing=True,
        default_root_dir="checkpoints",
    )

    # Add ModelCheckpoint callback to save the checkpoint when validation loss is at a new low
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-val-loss-{epoch:02d}-{val_loss:.2f}",
    )

    trainer_kwargs["callbacks"] = [checkpoint_callback]

    if wandb_logger is not None:
        trainer_kwargs["logger"] = wandb_logger

    trainer = pl.Trainer(**trainer_kwargs)

    # Train the model
    ckpt_path = None
    if "resume_path" in config.training:
        ckpt_path = config.training.resume_path

    trainer.fit(
        model,
        train_dataloaders=dataset_bundle.train_loader,
        val_dataloaders=dataset_bundle.val_loader,
        ckpt_path=ckpt_path,
    )

    if "wandb" in config:
        wandb.finish()


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
