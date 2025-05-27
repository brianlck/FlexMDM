import re
from transformers import GPT2TokenizerFast
from datasets import Dataset, load_dataset
from typing import Literal


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


def get_dataset(
    name: str,
    split: Literal["train", "validation", "test"],
    cache_dir=None,
    max_length=1024,
    num_proc=8,
) -> tuple[GPT2TokenizerFast, Dataset]:
    match name:
        case "wikitext2":
            dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir, split=split
            )
        case _:
            raise ValueError(f"Dataset {name} not supported")

    detokeniser = wt_detokeniser
    tokeniser: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    tokeniser.add_special_tokens({"pad_token": "[PAD]"})
    pad_token = tokeniser.pad_token_id

    def preprocess(sample):
        text = sample["text"]
        text = detokeniser(text)
        text = tokeniser(text, return_attention_mask=False)
        text["input_ids"] += max(0, max_length - len(text["input_ids"])) * [pad_token]
        return text

    tokenised_dataset = dataset.map(
        preprocess,
        num_proc=num_proc,
        load_from_cache_file=True,
        remove_columns=["text"],
    )
    tokenised_dataset = tokenised_dataset.filter(
        lambda x: len(x["input_ids"]) <= max_length
    )
    tokenised_dataset = tokenised_dataset.with_format("torch")

    return tokeniser, tokenised_dataset
