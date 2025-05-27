import abc
from typing import Any
import torch
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass
from schedule import Schedule


@dataclass
class Rate:
    unmask_rate: Tensor  # Shape [Batch, Length, Vocab]
    length_rate: Tensor  # Shape [Batch]

class ReparametrizedRate:
    def __init__(self, per_token_posterior: Tensor, length_posterior: Tensor):
        self.per_token_posterior = per_token_posterior
        self.length_posterior = length_posterior

class Interpolant(abc.ABC):
    def __init__(self, vocab_size: int, mask_token: int, pad_token: int, max_length: int):
        assert 0 <= mask_token < vocab_size
        assert pad_token >= vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size

    @abc.abstractmethod
    def stopping_rate(self, t: Tensor, x1: Tensor) -> Tensor:
        """
        Return the stopping time foreach position
        Shape:
            x1: [B, L]
        Returns:
            rate[B, L, 2]
            rate[:, :, 0] is the stopping time for (deletion, mask)
            rate[:, :, 1] is the stopping time for (mask, clean)
            thus rate[:, :, 0] <= rate[:, :, 1] <= 1
            (rate[:, :, 0] can be smaller than 0)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_actual_rate(self, rate: ReparametrizedRate, t: Tensor) -> Rate:
        """
        Return the actual rate for the sampling
        """
        return rate

    @abc.abstractmethod
    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        Shape:
            t: [B]
        Returns:
            weight_unmask: [B, L]
            weight_delete: [B, L+1]
        """
        raise NotImplementedError

    def sample_interpolant(self, t: Tensor, x1: Tensor) -> tuple[Any, Any, Any, Any, Any]:
        """
        Shapes:
            x1: [B, L]
            t: [B]
        Returns:
            xt: [B, L]
            st: [B, L] boolean mask of positions that corresponds to xt
            xt_mask_indices: [B, L] boolean mask of positions that are masked at xt
            x1_remained: [B, L] tokens that are not deleted, used for the training target
            gap_counts: [B, L+1] the number of deleted tokens between xt slots
        """
        # sample the stopping time (B, L, 2)
        stopping_time = self.stopping_rate(t, x1)
        t1, t2 = stopping_time[:, :, 0], stopping_time[:, :, 1]
        t_expand = t.unsqueeze(1).expand_as(t1)

        # consider clean tokens for now
        clean_pos    = x1.ne(self.pad_token)            # (B, L) True for clean tokens

        # decide for each position whether to delete/mask/clean
        delete_indices = clean_pos & (t_expand <  t1)
        mask_indices   = clean_pos & (t_expand >  t1) & (t_expand <  t2)

        # sample the intermediate state
        values = torch.where(delete_indices,
            self.pad_token,  # for deletion, change to pad token
            torch.where(
                mask_indices,
                self.mask_token,  # for masking, change to mask token
                x1                                   
            )
        )
        # pack all non-deleted positions to the front and sample st
        st = values.ne(self.pad_token)       # (B, L) bool: indices that's not a pad token
        keep_idx  = st.argsort(dim=1, descending=True)
        xt        = torch.gather(values, 1, keep_idx)

        # get the masked indices of xt
        xt_mask_indices = (xt == self.mask_token)

        # get the tokens that are not deleted (and also sorted)
        x1_tokens = torch.where(delete_indices, self.pad_token, x1)
        x1_remained = torch.gather(x1_tokens, 1, keep_idx)

        # calculating the gap counts
        B, L = x1.shape
        pos = torch.arange(L , device = x1.device)
        sentinel = L
        st_idx = torch.where(st, pos, sentinel)
        sorted_st , _ = torch.sort(st_idx, dim=1)
        x1_len = (x1 != self.pad_token).sum(dim=1)
        sorted_clamped = torch.minimum(sorted_st, x1_len.unsqueeze(1))
        pad_front = x1.new_zeros((B, 1)) - 1
        pad_back  = x1_len.unsqueeze(1)
        padded    = torch.cat([pad_front, sorted_clamped, pad_back], dim=1)  # (B, L+2)
        gap_counts = padded[:,1:] - padded[:,:-1] - 1                         # (B, L+1)
        gap_counts = gap_counts.clamp(min=0)

        assert xt.shape == st.shape == x1.shape == xt_mask_indices.shape == x1_remained.shape
        assert (xt == self.pad_token).sum() == (x1_remained == self.pad_token).sum()
        assert gap_counts.shape == (B, L+1)

        return xt, st, xt_mask_indices, x1_remained, gap_counts


class vanilla_mdm_interpolant(Interpolant):
    def __init__(self, vocab_size: int, mask_token: int, pad_token: int, max_length: int):
        super().__init__(vocab_size, mask_token, pad_token, max_length)

    def stopping_rate(self, t: Tensor, x1: Tensor) -> Tensor:
        """
        t2 is sampled from a uniform distribution over [0, 1]
        t1 is always 0
        """
        B, L = x1.shape
        t2 = torch.rand((B, L), device=x1.device)
        t1 = torch.zeros((B, L), device=x1.device)
        return torch.stack([t1, t2], dim=2)

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        """
        eps = 1e-6
        weight_unmask = 1.0 / (1 - t + eps)
        weight_unmask_expanded = weight_unmask.unsqueeze(1).expand(-1, x1.shape[1]) # (B,L)
        return weight_unmask_expanded

    def to_actual_rate(self, rate: ReparametrizedRate, t: Tensor) -> Rate:
        """
        Return the actual rate for the sampling
        """
        ## TODO: Implement the actual rate ##
        return rate

class mdmstyle_interpolant(Interpolant):
    def __init__(self, vocab_size: int, mask_token: int, pad_token: int, max_length: int, beta: float):
        """
        beta: the beta parameter the mdmstyle interpolant 
        """
        super().__init__(vocab_size, mask_token, pad_token, max_length)
        assert 0 <= beta <= 1
        self.beta = beta

    def stopping_rate(self, t: Tensor, x1: Tensor) -> Tensor:
        """
        t2 is sampled from a uniform distribution over [0, 1]
        t1 is sampled from a uniform distribution over [t2 - 1/beta, t2]
        """
        B, L = x1.shape
        t2 = torch.rand((B, L), device=x1.device)
        t1 = t2 - torch.rand((B, L), device=x1.device) / self.beta
        return torch.stack([t1, t2], dim=2)

    def elbo_weight(self, t: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        Shape:
            t: [B]
        Returns:
            weight_unmask: [B, L]
            weight_delete: [B, L+1]
        """
        eps = 1e-6
        weight_unmask = 1.0 / (1 - t - self.beta * ((1-t) ** 2) / 2 + eps)
        weight_delete = 2.0 / (1 - t + eps)
        weight_unmask_expanded = weight_unmask.unsqueeze(1).expand(-1, x1.shape[1]) # (B,L)
        weight_delete_expanded = weight_delete.unsqueeze(1).expand(-1, x1.shape[1]+1) # (B,L+1)
        return weight_unmask_expanded, weight_delete_expanded

    def to_actual_rate(self, rate: ReparametrizedRate, t: Tensor) -> Rate:
        """
        Return the actual rate for the sampling
        """
        ## TODO: Implement the actual rate ##
        return rate


class stoppingtime_interpolant(Interpolant):
    def __init__(self, vocab_size: int, mask_token: int, pad_token: int, max_length: int):
        super().__init__(vocab_size, mask_token, pad_token, max_length)

    def stopping_rate(self, t: Tensor, x1: Tensor) -> Tensor:
        """
        t1 is sampled from a uniform distribution over [0, 1]
        t2 is sampled from a uniform distribution over [t1, 1]
        """
        B, L = x1.shape
        t1 = torch.rand((B, L), device=x1.device)
        t2 = t1 + torch.rand((B, L), device=x1.device) * (1-t1)
        return torch.stack([t1, t2], dim=2)

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        """
        eps = 1e-6
        weight_unmask = 1.0 / (1 - t + eps)
        weight_unmask_expanded = weight_unmask.unsqueeze(1).expand(-1, x1.shape[1]) # (B,L)
        weight_delete = 1.0 / (1 - t + eps)
        weight_delete_expanded = weight_delete.unsqueeze(1).expand(-1, x1.shape[1]+1) # (B,L+1)
        return weight_unmask_expanded, weight_delete_expanded

    def to_actual_rate(self, rate: ReparametrizedRate, t: Tensor) -> Rate:
        """
        Return the actual rate for the sampling
        """
        # TODO: Implement the actual rate ##
        return rate
