import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model.MDM_transformer import DDiTNoLengthModel
from interpolant import MDMInterpolant  # replaced relative import
from schedule import get_schedule_from_config


class MaskedDiffusionModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = config.training.learning_rate

        # Initialize model (no length head)
        self.model = DDiTNoLengthModel(config)

        unmask_schedule = get_schedule_from_config(config.interpolant.unmask_schedule)

        # Initialize interpolant
        self.interpolant = MDMInterpolant(
            unmask_schedule=unmask_schedule,
            vocab_size=config.interpolant.tokens,
            mask_token=config.interpolant.mask_token,
            pad_token=config.interpolant.pad_token,
            max_length=config.interpolant.max_length,
        )

        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x, t) -> torch.Tensor:
        return self.model(x, t)

    def training_loss(self, x1, t):
        # sample interpolant and elbo weight

        interpolant_result = self.interpolant.sample_interpolant(t, x1)
        unmask_weight = self.interpolant.elbo_weight(t, x1)

        # model prediction
        predicted_logits = self(interpolant_result.xt, t)
        mask_indices = interpolant_result.mask_indices

        # compute unmask loss
        loss = unmask_weight[mask_indices] * F.cross_entropy(
            predicted_logits[mask_indices],
            interpolant_result.unmasked[mask_indices],
            reduction="none",
        )

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
