# A file of bregman divergences


def mse(x, y):
    sq_diff = (x - y) ** 2
    return sq_diff.reshape(sq_diff.size(0), -1).sum(dim=-1)
