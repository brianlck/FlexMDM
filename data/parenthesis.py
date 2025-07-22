from torch.utils.data import Dataset
import numpy as np
import torch


def generate_bracket(length: int, seq: str = ""):
    import random

    if length == 0:
        return seq
    p = random.randint(0, 1)
    if p == 0 or seq == "":
        return generate_bracket(length - 2, "(" + seq + "(")
    else:
        return seq + generate_bracket(length, "")


class BracketDataset(Dataset):
    def __init__(self, n, length_probs):
        lengths = list(length_probs.keys())
        probs = [length_probs[k] for k in lengths]
        self.data = []

        # Track actual length distribution
        length_counts = {length: 0 for length in lengths}

        for _ in range(n):
            L = int(np.random.choice(lengths, p=probs))
            seq = generate_bracket(L)
            mapped = [1 if c == "(" else 2 for c in seq]
            mapped += [3] * (64 - len(mapped))
            self.data.append(torch.tensor(mapped, dtype=torch.long))

            # Count actual sequence length
            actual_length = len(seq)
            if actual_length in length_counts:
                length_counts[actual_length] += 1
            else:
                length_counts[actual_length] = 1

        # Print length distribution
        print("Length distribution in dataset:")
        for length, count in sorted(length_counts.items()):
            print(f"  Length {length}: {count} sequences ({count/n:.2%})")

    @staticmethod
    def parse_tensor(tensor: torch.Tensor):
        if tensor.dim() == 1:
            result = ""
            mapping = {0: "m", 1: "(", 2: ")", 3: ""}
            for i in range(tensor.size(0)):
                result += mapping[int(tensor[i].item())]
            return result
        elif tensor.dim() == 2:
            return [
                BracketDataset.parse_tensor(tensor[i]) for i in range(tensor.size(0))
            ]
        else:
            raise ValueError("input cannot have dimension more than 2")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    ds = BracketDataset(1000, {4: 0.1, 8: 0.2, 32: 0.3, 64: 0.4})

    for i in range(len(ds)):
        print(ds[i])
