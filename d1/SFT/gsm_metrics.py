import json
import re
import tiktoken
from fractions import Fraction

def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


def safe_eval_arithmetic(expression):
    """Safely evaluate arithmetic expressions including fractions"""
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Replace common fraction patterns like 1/2 with Fraction(1, 2)
        # Handle mixed operations with fractions
        
        # First, try to handle it as a simple fraction
        if '/' in expression and expression.count('/') == 1 and '*' not in expression and '+' not in expression and '-' not in expression:
            parts = expression.split('/')
            if len(parts) == 2:
                try:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    if denominator != 0:
                        return numerator / denominator
                except ValueError:
                    pass
        
        # Handle more complex expressions
        # Only allow safe characters
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return None
            
        # Use eval for complex expressions
        result = eval(expression)
        return float(result)
    except:
        return None

def count_arithmetic_clauses(text):
    """Count arithmetic clauses like <<4/2=2>> and validate them"""
    if not text:
        return 0, 0
    
    # Pattern to match arithmetic clauses like <<expression=result>>
    pattern = r'<<([^>]+?)=([^>]+?)>>'
    matches = re.findall(pattern, text)
    
    total_clauses = len(matches)
    correct_clauses = 0
    
    for expression, claimed_result in matches:
        expression = expression.strip()
        claimed_result = claimed_result.strip()
        
        # Evaluate the expression
        actual_result = safe_eval_arithmetic(expression)
        
        if actual_result is not None:
            try:
                # Try to parse claimed result as float
                claimed_float = float(claimed_result)
                
                # Check if results match (with tolerance for floating point)
                if abs(actual_result - claimed_float) < 0.001:
                    correct_clauses += 1
            except ValueError:
                # If claimed result can't be parsed as float, it's incorrect
                pass
    
    return total_clauses, correct_clauses

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
    total_arithmetic_clauses = 0
    total_correct_arithmetic = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        # Count effective tokens
        effective_tokens = count_effective_tokens(raw_generation)
        total_effective_tokens += effective_tokens
        
        # Count arithmetic clauses
        clause_count, correct_count = count_arithmetic_clauses(raw_generation)
        total_arithmetic_clauses += clause_count
        total_correct_arithmetic += correct_count

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
                "arithmetic_clauses": clause_count,
                "correct_arithmetic": correct_count,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
        total_arithmetic_clauses,
        total_correct_arithmetic,
    )

file_format = "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/d1/SFT/results/top-prob-slide-datamix-gsm8k-5900/256_ours_random_0.5_{format}_generations.json"
# file_format = "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/d1/SFT/results/gsm8k/128_ours_random_0.5_{format}_generations.json"

# file_format = "/n/netscratch/albergo_lab/Lab/brianlck/interpretable-flow/d1/SFT/results/mdm_test/256_256_low_confidence_{format}_generations.json"

n_partitions = 4

metrics = []
total_arithmetic_clauses = 0
total_correct_arithmetic = 0

for i in range(n_partitions):
    result = parse_gsm_answers(json_path=file_format.format(format=i))
    total_correct, total_processed, processed_items, total_effective_tokens, arith_clauses, correct_arith = result
    
    metrics.extend([l["is_correct"] for l in processed_items])
    total_arithmetic_clauses += arith_clauses
    total_correct_arithmetic += correct_arith

print(count)

print(f"Total correct answers: {sum(1 for m in metrics if m == True)}")
print(f"Total processed answers: {len(metrics)}")
print(f"Total arithmetic clauses: {total_arithmetic_clauses}")
print(f"Correct arithmetic clauses: {total_correct_arithmetic}")
if total_arithmetic_clauses > 0:
    print(f"Arithmetic accuracy: {total_correct_arithmetic/total_arithmetic_clauses*100:.1f}%")