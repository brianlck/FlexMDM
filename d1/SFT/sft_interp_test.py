import torch
from torch import Tensor
from sft_interpolant import calc_len_loss_indices
from sft_interpolant import JointInterpolant, JointInterpolantResult
import numpy as np
import matplotlib.pyplot as plt

# test the len_loss_indices
# clean_tokens = torch.Tensor([[False, True, True, True, True, False, False, False],
#                              [False, False, True, True, True, True, False, False]]).bool()  
# print(calc_len_loss_indices(clean_tokens))



# test the joint interpolant
test_interpolant = JointInterpolant(
    mask_token = 11,
    pad_token = 30,
    max_length = 14,
    beta = 0.5)


num_batches = 500
batch_size = 32


def length(xt: torch.Tensor, pad: int, mask: int) -> torch.Tensor:
    batch_size = xt.shape[0]
    seq_len = (xt != pad).sum() / batch_size
    clean_tokens = ((xt != mask) & (xt != pad)).sum() / batch_size
    return seq_len, clean_tokens

batch_size = 2
t = torch.Tensor([0.5, 0.1])
x1 = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 30, 30, 30],
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 30, 30, 30, 30, 30]])

prompt_indices = torch.zeros(2, 14, dtype=torch.bool)
result = test_interpolant.sample_interpolant(t, x1, prompt_indices)


print(result.xt)
print(result.masked_indices)
print(result.x1_remained)
print(result.gap_counts)
print(result.len_loss_indices)

assert torch.all(result.xt[result.masked_indices] == test_interpolant.mask_token)
assert torch.all(result.x1_remained[~result.masked_indices] == result.xt[~result.masked_indices])


# seq_len_mean = []
# clean_tokens_mean = []

# for timesteps in torch.linspace(0, 1, 20):
#     seq_len_list = []
#     clean_tokens_list = []
#     for _ in range(num_batches):
#         t = torch.ones(batch_size) * timesteps
#         x1 = torch.randint(0, 3, (batch_size, 10))
#         prompt_indices = torch.zeros(batch_size, 10, dtype=torch.bool)
        
#         result = test_interpolant.sample_interpolant(t, x1, prompt_indices)

#         seq_len, clean_tokens = length(result.xt, pad = 20, mask = 10)
#         seq_len_list.append(seq_len.item())
#         clean_tokens_list.append(clean_tokens.item())

#         assert torch.all(result.xt[result.masked_indices] == test_interpolant.mask_token)
#         assert torch.all(result.x1_remained[~result.masked_indices] == result.xt[~result.masked_indices])

#     seq_len_mean.append(np.mean(seq_len_list))
#     clean_tokens_mean.append(np.mean(clean_tokens_list))

# print(seq_len_mean)
# print(clean_tokens_mean)

# plt.plot(torch.linspace(0, 1, 20), np.array(seq_len_mean))
# plt.plot(torch.linspace(0, 1, 20), np.array(clean_tokens_mean))
# plt.savefig("interp_test.png")






