import re
from transformers import GPT2TokenizerFast
from datasets import Dataset, load_dataset
from typing import Literal, List

TEXT_DATASETS = ["wikitext2"]
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


def setup_tokeniser() -> GPT2TokenizerFast:
    tokeniser = GPT2TokenizerFast.from_pretrained("gpt2")
    tokeniser.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
        }
    )

    return tokeniser

def decode_sequence_with_mask(
    seqs: List[List[int]],
    tokeniser: GPT2TokenizerFast,
    pad_token: int,
    mask_token: int
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
    num_proc=8,
) -> Dataset:
    match name:
        case "wikitext2":
            dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir, split=split
            )
        case _:
            raise ValueError(f"Dataset {name} not supported")

    detokeniser = wt_detokeniser
    tokeniser = setup_tokeniser()
    pad_token = tokeniser.pad_token_id


    # raw wikitext dataset also includes blanks
    def preprocess(sample):
        text = sample["text"]
        text = detokeniser(text)
        text = tokeniser(text, return_attention_mask=False)
        if len(text["input_ids"]) < MIN_LEN:
            return {"input_ids": []}
        text["input_ids"] += max(0, max_length - len(text["input_ids"])) * [pad_token]
        return text

    tokenised_dataset = dataset.map(
        preprocess,
        num_proc=num_proc,
        load_from_cache_file=True,
        remove_columns=["text"]
    )
    tokenised_dataset = tokenised_dataset.filter(
        lambda x: 0 < len(x["input_ids"]) <= max_length,
        num_proc=num_proc,
        load_from_cache_file=True,
    )
    tokenised_dataset = tokenised_dataset.with_format("torch")

    return tokenised_dataset
