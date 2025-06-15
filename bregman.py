# A file of bregman divergences
import torch


def mse(x, y):
    sq_diff = (x - y) ** 2
    if x.shape != y.shape:
        assert False, "x and y must have the same shape"
    return sq_diff.reshape(sq_diff.size(0), -1).sum(dim=-1)


# TODO: check if this formulation is correct
def jump_kernel_elbo(x, y, eps=1e-6):
    # x_safe: true length
    # y_safe: predicted length
    x_safe = torch.clamp(x, min=eps)
    y_safe = torch.clamp(y, min=eps)

    return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))
