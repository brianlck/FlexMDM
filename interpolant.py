import abc
from typing import Optional
import torch
from torch import Tensor
from dataclasses import dataclass
from schedule import Schedule


@dataclass
class ModelPrediction:
    token_posterior: Tensor
    length_posterior: Optional[Tensor]
    expected_gaps: Tensor

    def __init__(
        self,
        token_posterior: Tensor,
        length_posterior: Optional[Tensor] = None,
        expected_gaps: Optional[Tensor] = None,
    ):
        assert length_posterior is not None or expected_gaps is not None
        self.token_posterior = token_posterior
        self.length_posterior = length_posterior
        self.expected_gaps = expected_gaps
        if self.expected_gaps is None:
            _, _, L = self.length_posterior.shape
            index = torch.arange(0, L, device=token_posterior.device).view(1, 1, -1)
            self.expected_gaps = (self.length_posterior * index).sum(dim=-1)


@dataclass
class Rate:
    unmask_rate: Tensor  # Shape [Batch, Length, Vocab]
    length_rate: Tensor  # Shape [Batch]


@dataclass
class HittingTime:
    insertion_time: Tensor  # Shape [Batch, Length]
    unmasking_time: Tensor  # Shape [Batch, Length]

    def __iter__(self):
        yield from [self.insertion_time, self.unmasking_time]


@dataclass
class JointInterpolantResult:
    # Joint Interpolant
    xt: Tensor  # Shape [Batch, Length]
    st: Tensor  # Shape [Batch, Length]
    _x1: Tensor
    _pad_token: int
    _mask_token: int

    @property
    def mask_indices(self) -> Tensor:
        return self.xt != self._mask_token

    @property
    def unmasked(self) -> Tensor:
        return torch.gather(self._x1, 1, self.st)

    @property
    def xt_length(self) -> Tensor:
        # Calculate length of xt
        return (self.xt != self._pad_token).sum(dim=1)

    @property
    def x1_length(self) -> Tensor:
        # Calculate length of x1
        return (self._x1 != self._pad_token).sum(dim=1)

    @property
    def gaps_and_mask(self) -> tuple[Tensor, Tensor]:
        x1_len = self.x1_length
        gaps = self.st.clone()

        pad_front = gaps.new_zeros((gaps.shape[0], 1)) - 1  # -1 for the front padding
        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([pad_front, gaps, pad_back], dim=1)  # Add a leading zero

        gaps.scatter_(
            1, self.xt_length.unsqueeze(1) + 1, x1_len.unsqueeze(1)
        )  # Fill the last position with x1_len

        gaps = gaps[:, 1:] - gaps[:, :-1] - 1

        idx = torch.arange(gaps.size(1), device=self.xt.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx <= self.xt_length.unsqueeze(1)

        return gaps, mask


class JointInterpolant(abc.ABC):
    def __init__(
        self,
        insertion_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        """
        TODO: Add knobs
        """
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.insertion_schedule = insertion_schedule

    @abc.abstractmethod
    def hitting_time(self, t: Tensor, x1: Tensor) -> HittingTime:
        """
        Return the stopping time for each token in x1
        """
        raise NotImplementedError

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

    @abc.abstractmethod
    def to_actual_rate(self, prediction: ModelPrediction, t: Tensor) -> Rate:
        raise NotImplementedError

    def sample_interpolant(self, t: Tensor, x1: Tensor) -> JointInterpolantResult:
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
        insertion_time, unmasking_time = self.hitting_time(t, x1)

        clean_tokens = x1.ne(self.pad_token)
        deleted_tokens = clean_tokens & (
            self.insertion_schedule.at(t)[:, None] < insertion_time
        )
        masked_tokens = (
            clean_tokens
            & (self.insertion_schedule.at(t)[:, None] >= insertion_time)
            & (t[:, None] < unmasking_time)
        )

        xt = torch.where(
            deleted_tokens,
            self.pad_token,  # for deletion, change to pad token
            torch.where(
                masked_tokens,
                self.mask_token,  # for masking, change to mask token
                x1,
            ),
        )

        st = xt.ne(self.pad_token).argsort(dim=1, descending=True)
        xt = torch.gather(xt, 1, st)

        return JointInterpolantResult(
            xt=xt, st=st, _x1=x1, _pad_token=self.pad_token, _mask_token=self.mask_token
        )


class AnyOrderMaskInsertionInterpolant(JointInterpolant):
    def __init__(
        self,
        insertion_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        super().__init__(
            insertion_schedule, vocab_size, mask_token, pad_token, max_length
        )

    def hitting_time(self, t: Tensor, x1: Tensor) -> HittingTime:
        """
        t1 is sampled from a uniform distribution over [0, 1]. when t1 < self.mask_schedule.at(t)
        t2 is sampled from a uniform distribution over [t1, 1]
        """
        B, L = x1.shape
        eps = 1e-6
        t1 = eps + torch.rand((B, L), device=x1.device) * (
            1 - eps
        )  # address the issue of t1 = 0
        t2 = t1 + torch.rand((B, L), device=x1.device) * (1 - t1)
        return HittingTime(
            insertion_time=t1,  # (B, L)
            unmasking_time=t2,  # (B, L)
        )

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        """
        eps = 1e-6
        insert_weight = self.insertion_schedule.rate_scale_factor(t)
        insert_weight = insert_weight[:, None].expand(-1, x1.shape[1] + 1)

        unmask_weight = 1.0 / (1 - t + eps)
        unmask_weight = unmask_weight.unsqueeze(1).expand(-1, x1.shape[1])

        return unmask_weight, insert_weight

    def to_actual_rate(
        self, xt: Tensor, prediction: ModelPrediction, t: Tensor
    ) -> Rate:
        """
        Return the actual rate for the sampling
        Args:
            xt: [B, L] the sampled tokens
            prediction: ModelPrediction object containing token_posterior and expected_gaps
            t: [B] the time parameter
        """
        unmask_rate = prediction.token_posterior / (1 - t).view(-1, 1, 1)
        length_rate = (
            prediction.expected_gaps
            * self.insertion_schedule.rate_scale_factor(t).view(-1, 1)
        )

        return Rate(
            unmask_rate=unmask_rate,  # (B, L, V)
            length_rate=length_rate,  # (B, L+1)
        )


class VanillaMDMInterpolant(JointInterpolant):
    def __init__(
        self,
        mask_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        super().__init__(mask_schedule, vocab_size, mask_token, pad_token, max_length)

    def hitting_time(self, t: Tensor, x1: Tensor) -> Tensor:
        """
        t2 is sampled from a uniform distribution over [0, 1]
        t1 is always 0
        """
        B, L = x1.shape
        eps = 1e-6
        t2 = eps + torch.rand((B, L), device=x1.device) * (
            1 - eps
        )  # address the issue of t2 = 0
        t1 = torch.zeros((B, L), device=x1.device)
        return torch.stack([t1, t2], dim=2)

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        there's no weight_delete for the vanilla MDM
        """
        weight_unmask = self.mask_schedule.rate_scale_factor(t)
        weight_unmask_expanded = weight_unmask.unsqueeze(1).expand(
            -1, x1.shape[1]
        )  # (B,L)
        return weight_unmask_expanded

    def to_actual_rate(
        self, xt: Tensor, prediction: ModelPrediction, t: Tensor
    ) -> Rate:
        """
        Return the actual rate for the sampling
        """
        raise NotImplementedError
