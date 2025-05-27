import abc
from typing import Any
import torch
from torch import Tensor


class Interpolant(abc.ABC):
    def __init__(self, vocab_size: int, mask_token: int, pad_token: int, max_length: int):
        assert 0 <= mask_token < vocab_size
        assert pad_token >= vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size


    @abc.abstractmethod
    def stopping_rate(self, t: Tensor, x_1: Tensor) -> Tensor:
        """
        Return the stopping time foreach position
        Shape:
            t: [B]
            x1: [B, L]
        Returns:
            rate[B, L, 2]
            rate[:, :, 0] is the stopping time for (deletion, mask)
            rate[:, :, 1] is the stopping time for (mask, clean)
            thus 0 <= rate[:, :, 0] <= rate[:, :, 1] <= 1
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
        print("t1:\n", t1)
        print("t2:\n", t2)
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


class ToyInterpolant(Interpolant):
    def stopping_rate(self, t: Tensor, x1: Tensor) -> Tensor:
        B, L = x1.shape
        t1   = torch.rand((B, L), device=x1.device)
        t2   = torch.rand((B, L), device=x1.device)
        t2 = t2 + t1
        return torch.stack([t1, t2], dim=-1) 



def run_test():
    torch.manual_seed(0)

    vocab_size  = 20
    mask_token  = 10
    pad_token   = 99
    L           = 20
    B           = 1

    interp = ToyInterpolant(vocab_size, mask_token, pad_token, max_length=L)

    # batch with existing pads (99)
    x1 = torch.tensor([
        [ 4,  5,  6,  7, 3, 8, 2, 1, 1, 1, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
    ])

    # random times in (0,1)
    t  = torch.rand(B)

    xt, st, xt_mask, x1_rem, insertion = interp.sample_interpolant(t, x1)

    print("t:", t)
    print("x1:\n", x1)
    print("xt:\n", xt)
    print("st (kept):\n", st.int())
    print("xt_mask_indices:\n", xt_mask.int())
    print("x1_remained:\n", x1_rem)
    print("insertion_lens:\n", insertion)

    # ---- reconstruction unit-test ----
    # # rebuild x1 from (xt, insertion_lens, deletes=pads) to be extra sure
    # rebuilt = torch.full_like(x1, pad_token)
    # for b in range(B):
    #     cursor = 0
    #     for gap, tok in zip(insertion[b], xt[b]):
    #         cursor += gap.item()          # skip deletions
    #         if tok != pad_token:
    #             rebuilt[b, cursor] = tok
    #             cursor += 1

    # print("rebuilt:\n", rebuilt)

if __name__ == "__main__":
    run_test()