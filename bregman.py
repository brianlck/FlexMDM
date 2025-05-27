# A file of bregman divergences


def mse(x, y):
    sq_diff = (x - y) ** 2
    return sq_diff.reshape(sq_diff.size(0), -1).sum(dim=-1)
<<<<<<< HEAD


# TODO: check if this formulation is correct
def scalar_bregman(x, y, eps=1e-6):
    x_safe = torch.clamp(x, min=eps)
    y_safe = torch.clamp(y, min=eps)

    return y_safe - x_safe + x_safe * ( torch.log(x_safe) - torch.log(y_safe))
    
=======
>>>>>>> origin/main
