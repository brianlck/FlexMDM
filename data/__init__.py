from omegaconf import DictConfig
from .parenthesis import BracketDataset
from .text import get_text_dataset, setup_tokeniser_from_dataset, TEXT_DATASETS
from typing import Optional
from transformers import AutoTokenizer
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    tokeniser: Optional[AutoTokenizer] = None


def setup_data_and_update_config(config: DictConfig) -> DatasetBundle:
    """
    Get the dataset and update the config with token information for text datasets.
    """
    tokeniser = None
    if config.dataset in TEXT_DATASETS:
        tokeniser = setup_tokeniser_from_dataset(config.dataset)
        train_set = get_text_dataset(
            config.dataset,
            split="train",
            max_length=config.interpolant.max_length,
            filter_max_length=config.training.filter_max_length,
        )
        val_set = get_text_dataset(
            config.dataset,
            split="validation",
            max_length=config.interpolant.max_length,
            filter_max_length=config.training.filter_max_length,
        )
        config.interpolant.tokens = len(tokeniser)
        config.interpolant.pad_token = tokeniser.pad_token_id
        config.interpolant.mask_token = tokeniser.mask_token_id

    if config.dataset == "bracket":
        train_set = BracketDataset(2048, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1})
        val_set = BracketDataset(300, {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1})

    train_loader = DataLoader(
        train_set,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.training.cpus,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=config.training.cpus,
        pin_memory=True,
        persistent_workers=True,
    )

    return DatasetBundle(
        train_loader=train_loader, val_loader=val_loader, tokeniser=tokeniser
    )
