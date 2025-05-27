"""
Training script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.transformer import AnyOrderMaskInsertionFlow
from interpolant import stoppingtime_interpolant, mdmstyle_interpolant, ReparametrizedRate
from parenthesis import BracketDataset
from bregman import mse, scalar_bregman
import os
from pathlib import Path
from schedule import GeometricSchedule, LinearSchedule

# Configuration
config = OmegaConf.create({
    'tokens': 3,  # 0, 1, 2
    'mask_token': 0,
    'pad_token': 3,
    'max_length': 64,
    'model': {
        'hidden_size': 256,
        'n_heads': 4,
        'cond_dim': 64,
        'dropout': 0.1,
        'n_blocks': 4,
    }
})

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
CHECKPOINT_DIR = "checkpoints/bracket-flow"

class BracketFlowModule(pl.LightningModule):
    def __init__(self, config, learning_rate=LEARNING_RATE):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = AnyOrderMaskInsertionFlow(config, len_predict_type=args.len_predict_type)

        # Initialize Masking Schedule
        mask_schedule = GeometricSchedule(min=5, max=0.01)

        # Initialize interpolant
        if args.interpolant == 'stoppingtime':
            self.interpolant = stoppingtime_interpolant(
            vocab_size=config.tokens,
            mask_token=config.mask_token,
            pad_token=config.pad_token,
            max_length=config.max_length
        )
        elif args.interpolant == 'mdmstyle':
            self.interpolant = mdmstyle_interpolant(
                vocab_size=config.tokens,
                mask_token=config.mask_token,
                pad_token=config.pad_token,
                max_length=config.max_length,
                beta=args.beta
            )
        else:
            raise ValueError(f"Invalid interpolant: {args.interpolant}")

        # Save hyperparameters
        self.save_hyperparameters()
    
    def forward(self, x, t) -> ReparametrizedRate:
        return self.model(x, t)
    
    def training_step(self, batch, batch_idx):
        x1 = batch
        batch_size = x1.shape[0]
        
        # Sample random time steps
        t = torch.rand(batch_size, device=x1.device)

        # Sample interpolant and load elbo weight
        xt, st, xt_mask_indices, x1_remained, gap_counts = self.interpolant.sample_interpolant(t, x1)
        weight_unmask, weight_delete = self.interpolant.elbo_weight(t, x1)
        
        # Get model predictions
        pred_rate: ReparametrizedRate = self(xt, t)

        # compute unmask loss
        # TODO: check the dtype issue carefully
        unmask_logits = pred_rate.per_token_posterior
        if args.unmask_loss_type == 'ce':
            unmask_loss = F.cross_entropy(unmask_logits[xt_mask_indices], x1_remained[xt_mask_indices], reduction='none') * weight_unmask[xt_mask_indices]
            unmask_loss = unmask_loss.sum() / (batch_size * config.max_length)
        elif args.unmask_loss_type == 'mse':
            # make x1_remained one-hot
            x1_one_hot = F.one_hot(x1_remained, num_classes=self.config.tokens + 1)
            x1_one_hot = x1_one_hot[:, :, :-1].to(unmask_logits.dtype) # remove the padding token index to match the size
            unmask_prob = unmask_logits.softmax(dim=-1)
            unmask_loss = F.mse_loss(unmask_prob[xt_mask_indices], x1_one_hot[xt_mask_indices], reduction='none').mean()
            ## TODO-training: check if we need to use the weight_unmask
        else:
            raise ValueError(f"Invalid unmask loss type: {args.unmask_loss_type}")

        # compute length loss
        len_posterior = pred_rate.length_posterior
        if args.len_predict_type == 'distribution':
            gap_one_hot = F.one_hot(gap_counts, num_classes=self.config.max_length + 1)
            gap_one_hot = gap_one_hot.to(len_posterior.dtype)
            len_loss = F.mse_loss(len_posterior, gap_one_hot, reduction='none').mean()
            ## TODO-training: check if we need to use the weight_delete
            ## TODO: check the scale issue of the loss
        elif args.len_predict_type == 'expectation':
            # gap_counts: B X (L+1), len_posterior: B X (L+1), weight_delete: B X (L+1)
            len_loss = scalar_bregman(gap_counts.to(len_posterior.dtype), len_posterior) * weight_delete
            len_loss = len_loss.mean()
        else:
            raise ValueError(f"Invalid len predict type: {args.len_predict_type}")
        
        loss = unmask_loss + args.len_loss_constant * len_loss

        
        # Log losses
        self.log('train/unmask_loss', unmask_loss.item(), prog_bar=True)
        self.log('train/len_loss', len_loss.item(), prog_bar=True)
        self.log('train/total_loss', loss.item(), prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Save additional information to checkpoint"""
        checkpoint['config'] = self.config
        checkpoint['interpolant_state'] = {
            'vocab_size': self.interpolant.vocab_size,
            'mask_token': self.interpolant.mask_token,
            'pad_token': self.interpolant.pad_token,
            'max_length': self.interpolant.max_length
        }

    def on_load_checkpoint(self, checkpoint):
        """Load additional information from checkpoint"""
        self.config = checkpoint['config']
        interpolant_state = checkpoint['interpolant_state']
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            mask_schedule=GeometricSchedule(min=5., max=0.01),
            vocab_size=interpolant_state['vocab_size'],
            mask_token=interpolant_state['mask_token'],
            pad_token=interpolant_state['pad_token'],
            max_length=interpolant_state['max_length']
        )

def train(resume_from_checkpoint=None):
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize dataset and dataloader
    dataset = BracketDataset(10000, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = BracketFlowModule(config)
    
    # Initialize trainer with enhanced checkpointing
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu',
        devices=1,
        gradient_clip_val=1.0, # to avoid gradient explosion
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                monitor='train/total_loss',
                mode='min',
                save_top_k=3,
                filename='bracket-any-order-mask-flow-{epoch:02d}-{train/total_loss:.4f}',
                save_last=True,  # Always save the last checkpoint
                every_n_epochs=5,  # Save every epoch
            )
        ],
        log_every_n_steps=100,
        enable_checkpointing=True,
    )
    
    # Train the model
    trainer.fit(
        model, 
        dataloader,
        ckpt_path=resume_from_checkpoint
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--interpolant', type=str, help='Interpolant to use', default='stoppingtime')
    parser.add_argument('--beta', type=float, help='Beta for mdmstyle interpolant', default=1.0)
    parser.add_argument('--unmask_loss_type', type=str, help='Unmask loss type', default='ce')
    parser.add_argument('--len_predict_type', type = str, help = 'Length prediction type', default = 'distribution')
    parser.add_argument('--len_loss_constant', type=float, help='Length loss constant', default=1.0)
    args = parser.parse_args()
    
    train(resume_from_checkpoint=args.resume)
