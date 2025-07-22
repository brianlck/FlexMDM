import json
import re
import tiktoken

def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


count = 0

def parse_gsm_answers(json_path=None, json_data=None):
    global count
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens

        parsed_answer = None

        boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
        if boxed_matches:
            for boxed_content in boxed_matches:
                boxed_content = boxed_content.strip()
                if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                    try:
                        parsed_answer = float(boxed_content)
                        count += 1
                        break
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[0])
                                count += 1
                                break
                            except ValueError:
                                pass

        if parsed_answer is None:
            answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text:
                    try:
                        parsed_answer = float(answer_text)
                    except ValueError:
                        numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                        if numbers:
                            try:
                                parsed_answer = float(numbers[-1])
                            except ValueError:
                                pass

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )


file_format = "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow-tmp/d1/SFT/results/gsm8k-2600/512_ours_random_0.5_{format}_generations.json"
# file_format = "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow-tmp/d1/SFT/results/mdm_test/256_256_low_confidence_{format}_generations.json"

n_partitions = 4

metrics = []
for i in range(n_partitions):
    metrics.extend([l["is_correct"] for l in parse_gsm_answers(json_path=file_format.format(format=i))[2]])

print(count)

print(f"Total correct answers: {sum(1 for m in metrics if m == True)}")
print(f"Total processed answers: {len(metrics)}")