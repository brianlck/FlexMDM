from torch.utils.data import Dataset
import numpy as np
import torch


def generate_bracket(length: int, seq: str =  ""):
    import random
    if length == 0:
        return seq
    p = random.randint(0, 1)
    if p == 0 or seq == "":
        return generate_bracket(length-2, "(" + seq + ")")
    else:
        return seq + generate_bracket(length, "")


class BracketDataset(Dataset):
    def __init__(self, n, length_probs):
        lengths = list(length_probs.keys())
        probs = [length_probs[k] for k in lengths]
        self.data = []
        for _ in range(n):
            L = int(np.random.choice(lengths, p=probs))
            seq = generate_bracket(L)
            mapped = [1 if c == "(" else 2 for c in seq]
            mapped += [3] * (64 - len(mapped))
            self.data.append(torch.tensor(mapped, dtype=torch.long))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    ds = BracketDataset(1000, {4: 0.1, 8: 0.2, 32: 0.3, 64: 0.4})

    for i in range(len(ds)):
        print(ds[i])