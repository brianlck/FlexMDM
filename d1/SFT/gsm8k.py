import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import time
import random
import re
from datasets import load_dataset
from parsers import Parser, is_equiv
import torch.distributed as dist
from torch.utils.data import DataLoader

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def extract_rationale_and_answer(answer_field: str):
    """Split raw GSM8K `answer` into (reasoning_text, final_answer_str)."""
    if "####" in answer_field:
        rationale, final_ans = answer_field.split("####", 1)
        return rationale.strip(), final_ans.strip()
    # fallback – dataset might already be split elsewhere
    return "", answer_field.strip()

# -----------------------------------------------------------------------------
# data preprocessing code
# -----------------------------------------------------------------------------

def preprocess_gsm8k(
    split: str, tokenizer: AutoTokenizer, max_length: int
):
    ds = load_dataset("gsm8k", "main", split=split)
    prompt_builder = GSM8KDataset(tokenizer, num_examples=0, add_reasoning=True)
    preprocessed = []

    for ex in tqdm(ds, desc = "Preprocessing"):
        q = ex["question"].strip()
        rat, ans = extract_rationale_and_answer(ex["answer"])
        prompt_text = prompt_builder.create_prompt(q)
        target_txt = f"{rat}</reasoning>\n<answer>\\boxed{{{ans}}}</answer>"
        full_txt   = prompt_text + target_txt

        enc = tokenizer(full_txt, truncation=True, max_length=max_length,
                  padding="max_length", return_tensors="pt")
        input_ids  = enc.input_ids.squeeze(0)
        prompt_len = len(tokenizer(prompt_text).input_ids)

        preprocessed.append({"input_ids": input_ids, "prompt_lengths": prompt_len})

    # ---- shuffle & split like your original helper ----
    random.shuffle(preprocessed)

    return preprocessed


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GSM_SYSTEM_PROMPT,
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
        self.load_test_dataset()
        self.create_few_shot_prompt()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def create_prompt(self, input_text):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def load_few_shot_examples(self):
        if isinstance(self.dataset, GSM8KDataset):
            train_data = load_dataset("gsm8k", "main", split="train")
            examples = random.sample(range(len(train_data)), self.num_examples)
            return [train_data[example] for example in examples]
        else:
            return []

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
        self.few_shot_prompt = "\n\n".join(formatted_examples)

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["question"]
        answer = Parser.extract_answer_gsm8k(self.dataset[self.subsample[idx].item()]["answer"])
        prompt = self.create_prompt(question)
        return prompt, question, answer

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        ).input_ids
        return {"input_ids": input_ids, "questions": questions, "answers": answers, "prompts": prompts}



# test code
# if __name__ == "__main__":
#     train_data, test_data = preprocess_gsm8k(
#         split="train", model_name="GSAI-ML/LLaDA-8B-Base", max_length=4096, test_split=0.01
#     )

#     def collate_fn(batch):
#         ids = torch.stack([item["input_ids"] for item in batch])
#         plen = torch.tensor([item["prompt_len"] for item in batch])
#         return {"input_ids": ids, "prompt_lengths": plen}

#     loader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)

#     for batch in loader:
#         print(batch["input_ids"].shape)
#         print(batch["prompt_lengths"])
#         break