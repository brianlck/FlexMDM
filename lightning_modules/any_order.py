import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch.nn.functional as F
from model.transformer import AnyOrderMaskInsertionFlow
from interpolant import AnyOrderMaskInsertionInterpolant, ModelPrediction
from bregman import jump_kernel_elbo, mse
from schedule import get_schedule_from_config


class AnyOrderInsertionFlowModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model_type = config.interpolant.type
        self.learning_rate = config.training.learning_rate
        self.unmask_loss_fn = config.training.loss_fn.unmask
        self.insert_loss_fn = config.training.loss_fn.insert

        # Initialize model based on type
        self.model = AnyOrderMaskInsertionFlow(config)

        insert_schedule = get_schedule_from_config(config.interpolant.insert_schedule)
        unmask_schedule = get_schedule_from_config(config.interpolant.unmask_schedule)

        # Initialize interpolant
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            insertion_schedule=insert_schedule,
            unmask_schedule=unmask_schedule,
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
                    prediction.token_logits[mask_indices],
                    interpolant_sample.unmasked[mask_indices],
                    reduction="none",
                )
                unmask_loss = unmask_loss.mean()

            case _:
                raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_fn}")

        match self.insert_loss_fn:
            case "expectation":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss = insert_weight[gaps_mask] * jump_kernel_elbo(
                    prediction.expected_gaps[gaps_mask], gaps[gaps_mask]
                )
                print(prediction.expected_gaps[gaps_mask], gaps[gaps_mask])
                insertion_loss = insertion_loss.mean()

            case "distribution":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                gap_one_hot = F.one_hot(
                    gaps,
                    num_classes=self.config.interpolant.max_length + 1,
                ).to(prediction.length_posterior.dtype)
                insertion_loss = insert_weight[gaps_mask] * mse(
                    prediction.length_posterior[gaps_mask], gap_one_hot[gaps_mask]
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

        insert_schedule = get_schedule_from_config(
            self.config.interpolant.insert_schedule
        )
        unmask_schedule = get_schedule_from_config(
            self.config.interpolant.unmask_schedule
        )

        self.interpolant = AnyOrderMaskInsertionInterpolant(
            insertion_schedule=insert_schedule,
            unmask_schedule=unmask_schedule,
            vocab_size=self.config.interpolant.tokens,
            mask_token=self.config.interpolant.mask_token,
            pad_token=self.config.interpolant.pad_token,
            max_length=self.config.interpolant.max_length,
        )
