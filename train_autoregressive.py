"""
Training script for CausalDiT (Autoregressive Transformer)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model.causal_dit import CausalDiT
from parenthesis import BracketDataset
import os
from pathlib import Path

# Configuration
config = OmegaConf.create({
    'tokens': 3,  # 0, 1, 2 -- will be 2 after removing mask token
    'pad_token': 2,
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
CHECKPOINT_DIR = "checkpoints/causal-dit"

class CausalDiTModule(pl.LightningModule):
    def __init__(self, config, learning_rate=LEARNING_RATE):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.model = CausalDiT(config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch  # (B, L)
        x = (x - 1.0).long() # for autoregressive training, remove mask token
        # Predict next token: input is x[:, :-1], target is x[:, 1:]
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]
        logits = self(input_seq)  # (B, L-1, vocab_size)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def train(resume_from_checkpoint=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    dataset = BracketDataset(10000, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CausalDiTModule(config)
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu',
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR,
                monitor='train/loss',
                mode='min',
                save_top_k=3,
                filename='causal-dit-{epoch:02d}-{train/loss:.4f}',
                save_last=True,
                every_n_epochs=5,
            )
        ],
        log_every_n_steps=100,
        enable_checkpointing=True,
    )
    trainer.fit(
        model,
        dataloader,
        ckpt_path=resume_from_checkpoint
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(resume_from_checkpoint=args.resume) 