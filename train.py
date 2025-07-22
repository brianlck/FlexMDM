import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import wandb
import data
from lightning_modules import (
    MaskedDiffusionModule,
    AutoregressiveModule,
    AnyOrderInsertionFlowModule,
)
from pytorch_lightning.utilities import rank_zero_only


torch.set_printoptions(threshold=10_000)
torch.set_float32_matmul_precision("high")


def train(config: DictConfig):
    # set the random seed
    pl.seed_everything(42)
    torch.manual_seed(42)

    if "wandb" in config and rank_zero_only.rank == 0:
        init_kwargs = dict(
            project="interpretable-flow",
            entity=config.wandb.entity,
            config=OmegaConf.to_container(config, resolve=True),
            name=config.wandb.name,
        )
        # resume wandb run if we're resuming from a checkpoint
        if "resume_path" in config.training:
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)
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

    match config.trainer:
        case "mdm":
            module = MaskedDiffusionModule(config)
        case "autoregressive":
            module = AutoregressiveModule(config)
        case "any-order-flow":
            module = AnyOrderInsertionFlowModule(config)
        case _:
            raise NotImplementedError(f"Trainer {config.trainer} is not supported")

    # Initialize trainer

    # Configure trainer arguments
    trainer_kwargs = dict(
        num_nodes=config.training.nodes,
        accelerator="gpu",
        devices=config.training.devices,
        strategy="ddp",
        accumulate_grad_batches=(
            config.training.batch_size
            // (
                config.training.per_gpu_batch_size
                * config.training.nodes
                * config.training.devices
            )
        ),
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=config.training.checkpoint_dir,
        gradient_clip_val=1.0,
    )
    # Only one of max_steps or max_epochs will be used
    if config.training.max_steps is not None:
        trainer_kwargs["max_steps"] = config.training.max_steps
    elif config.training.num_epochs is not None:
        trainer_kwargs["max_epochs"] = config.training.num_epochs
        config.training.max_steps = config.training.num_epochs * len(
            dataset_bundle.train_loader
        )
    else:
        raise ValueError(
            "Either max_steps or num_epochs must be specified in the config"
        )

    if config.training.warmup_steps is None:
        config.training.warmup_steps = int(config.training.max_steps * 0.01)

    # Add ModelCheckpoint callback to save the checkpoint when validation loss is at a new low
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=config.training.save_top_k,
        save_last=True,
        filename="epoch-{epoch:02d}-val_loss-{val_loss:.4f}",
        dirpath=config.training.checkpoint_dir,
        every_n_train_steps=10000,
        # every_n_epochs=config.training.save_every_n_epochs,
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
        module,
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
