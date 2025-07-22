import re
from transformers import GPT2TokenizerFast, GPTNeoXTokenizerFast
from datasets import Dataset, load_dataset
from typing import Literal, List

TEXT_DATASETS = ["wikitext2", "openwebtext"]
MIN_LEN = 50


def wt_detokeniser(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def setup_tokeniser(tokeniser_name: str) -> GPT2TokenizerFast:
    match tokeniser_name:
        case "gpt2":
            tokeniser = GPT2TokenizerFast.from_pretrained("gpt2")
        case "gpt-neo":
            tokeniser = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        case _:
            raise ValueError(f"Tokeniser {tokeniser_name} not supported")
    tokeniser.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
    )
    return tokeniser


def find_delimiter_positions(tokens, delimiter_tokens):
    """Return the start indices where the delimiter occurs in the token sequence."""
    positions = []
    n = len(delimiter_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == delimiter_tokens:
            positions.append(i)
    return positions


def recursive_split(tokens, max_length, delimiter_tokens):
    if len(tokens) <= max_length:
        return [tokens]

    # Find all positions where the delimiter sequence occurs
    split_candidates = find_delimiter_positions(tokens, delimiter_tokens)
    if not split_candidates:
        # Safe fallback: naive split
        return [
            tokens[i : min(i + max_length, len(tokens))]
            for i in range(0, len(tokens), max_length)
        ]

    # Find delimiter closest to the midpoint
    midpoint = len(tokens) // 2
    split_point = min(split_candidates, key=lambda x: abs(x - midpoint))

    # Recurse on both sides, skipping the delimiter
    dlen = len(delimiter_tokens)
    left = recursive_split(tokens[:split_point], max_length, delimiter_tokens)
    right = recursive_split(tokens[split_point + dlen :], max_length, delimiter_tokens)

    return left + right


def preprocess_batch(batch, pad_token, max_length, delimiter, detokeniser, tokeniser):
    all_input_ids = []
    all_lengths = []

    for text in batch["text"]:
        if detokeniser is not None:
            text = detokeniser(text)

        tokens = tokeniser.encode(text, add_special_tokens=False)
        chunks = recursive_split(tokens, max_length, delimiter)

        all_input_ids.extend(
            [
                c + [pad_token] * (max_length - len(c)) if len(c) < max_length else c
                for c in chunks
            ]
        )
        all_lengths.extend([len(chunk) for chunk in chunks])

    return {
        "input_ids": all_input_ids,
        "length": all_lengths,
    }


def setup_tokeniser_from_dataset(dataset_name: str):
    tokeniser = None
    match dataset_name:
        case "wikitext2" | "openwebtext":
            tokeniser = setup_tokeniser("gpt2")
        case "dclm":
            tokeniser = setup_tokeniser("gpt-neo")
        case _:
            raise ValueError(f"Tokeniser for dataset {dataset_name} not supported")

    return tokeniser


def decode_sequence_with_mask(
    seqs: List[List[int]], tokeniser: GPT2TokenizerFast, pad_token: int, mask_token: int
) -> List[str]:
    """
    Decode a sequence with visible mask tokens.
    """
    decoded = []
    for seq in seqs:
        tokens = tokeniser.convert_ids_to_tokens(seq)
        filtered = []
        for tok, tok_id in zip(tokens, seq):
            if tok_id == pad_token:
                continue
            if tok_id == mask_token:
                filtered.append("[MASK]")
            else:
                filtered.append(tok)
        text = tokeniser.convert_tokens_to_string(filtered)
        decoded.append(text)
    return decoded


def get_text_dataset(
    name: str,
    split: Literal["train", "validation", "test"],
    cache_dir=None,
    max_length=1024,
    num_proc=64,
    filter_max_length=True,
) -> Dataset:
    match name:
        case "wikitext2":
            dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir, split=split
            )
        case "openwebtext":
            ds_all = load_dataset(name, cache_dir=cache_dir)
            train_ds = ds_all["train"]
            if split in ["train", "validation"]:
                split_data = train_ds.train_test_split(test_size=0.02, seed=42)
                dataset = (
                    split_data["train"] if split == "train" else split_data["test"]
                )
            else:
                raise ValueError(f"Dataset {name} does not support split {split}")
        case _:
            raise ValueError(f"Dataset {name} not supported")

    match name:
        case "wikitext2":
            detokeniser = wt_detokeniser
        case "openwebtext":
            detokeniser = None
        case "dclm":
            detokeniser = None
        case _:
            raise ValueError(f"Dataset {name} not supported")

    tokeniser = setup_tokeniser_from_dataset(name)
    pad_token = tokeniser.pad_token_id

    if filter_max_length:

        def preprocess(sample):
            text = sample["text"]
            if detokeniser is not None:
                text = detokeniser(text)
            text = tokeniser(text, return_attention_mask=False)
            if len(text["input_ids"]) < MIN_LEN:
                return {"input_ids": []}
            text["input_ids"] += max(0, max_length - len(text["input_ids"])) * [
                pad_token
            ]
            return text

        tokenised_dataset = dataset.map(
            preprocess,
            num_proc=num_proc,
            load_from_cache_file=True,
            remove_columns=["text"],
        )
        tokenised_dataset = tokenised_dataset.filter(
            lambda x: 0 < len(x["input_ids"]) <= max_length,
            num_proc=num_proc,
            load_from_cache_file=True,
        )
        tokenised_dataset = tokenised_dataset.with_format("torch")

        return tokenised_dataset
    else:
        tokenised_dataset = dataset.map(
            lambda batch: preprocess_batch(
                batch,
                pad_token=pad_token,
                max_length=max_length,
                detokeniser=detokeniser,
                tokeniser=tokeniser,
                delimiter=[198, 198],
            ),
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"],
        )

        tokenised_dataset = tokenised_dataset.with_format("torch")

        return tokenised_dataset
