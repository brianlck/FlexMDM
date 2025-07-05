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
        self.model = torch.compile(self.model)

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

        self.ema_decay = config.training.ema_decay or 0.0
        self.use_ema = self.ema_decay > 0
        self._orig_params = {}

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps

        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        post_warmup = max_steps - warmup_steps
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=post_warmup // 10,
            T_mult=1,
            eta_min=0.0,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
    ):
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure
        )
        # log learning rate and gradient norm
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=True)
        grad_norm = torch.sqrt(
            sum(p.grad.norm(2) ** 2 for p in self.parameters() if p.grad is not None)
        )
        self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=True)

        # update EMA
        if self.use_ema:
            for n, p in self.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(
                    p.data.clone().detach(), alpha=1 - self.ema_decay
                )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config
        # save EMA state
        if self.use_ema:
            checkpoint["ema_params"] = {n: v.cpu() for n, v in self.ema_params.items()}

    def on_load_checkpoint(self, checkpoint):
        self.config = checkpoint["config"]

        unmask_schedule = get_schedule_from_config(
            self.config.interpolant.unmask_schedule
        )

        self.interpolant = MDMInterpolant(
            unmask_schedule=unmask_schedule,
            vocab_size=self.config.interpolant.tokens,
            mask_token=self.config.interpolant.mask_token,
            pad_token=self.config.interpolant.pad_token,
            max_length=self.config.interpolant.max_length,
        )

        self.ema_params = checkpoint["ema_params"] if self.use_ema else {}

    def swap_to_ema(self):
        for name, p in self.named_parameters():
            self._orig_params[name] = p.data.clone()
            p.data.copy_(self.ema_params[name].to(p.device))

    def restore_original(self):
        for name, p in self.named_parameters():
            p.data.copy_(self._orig_params[name])
        self._orig_params.clear()

    def on_train_start(self):
        # initialize and move EMA buffers once model is on correct device
        if self.use_ema:
            self.ema_params = {
                name: param.clone().detach().to(self.device)
                for name, param in self.named_parameters()
            }
            for buf in self.ema_params.values():
                buf.requires_grad = False
