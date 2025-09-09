from datasets import load_dataset   

def preprocess_opc_coder(tokenizer, max_length):
    ds = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct")['train']
    
    def process_sample(sample):
        # Tokenize instruction and output separately
        instruction_tokens = tokenizer(sample['instruction'], add_special_tokens=False)['input_ids']
        output_tokens = tokenizer(sample['output'], add_special_tokens=False)['input_ids']
        
        # Combine instruction and output
        input_ids = instruction_tokens + output_tokens
        
        # Pad to max_length
        if len(input_ids) < max_length:
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        elif len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Set prefix_cutoff to the length of the instruction
        prefix_cutoff = len(instruction_tokens)
        
        return {
            'input_ids': input_ids,
            'prefix_cutoff': prefix_cutoff
        }
    
    processed_ds = ds.map(process_sample, remove_columns=ds.column_names)
    return processed_ds


def preprocess_human_eval(tokenizer, max_length):
    ds = load_dataset("openai/openai_humaneval")['test']
    
    def process_sample(sample):
        # Tokenize prompt and canonical_solution separately
        prompt_tokens = tokenizer(sample['prompt'], add_special_tokens=False)['input_ids']
        solution_tokens = tokenizer(sample['canonical_solution'], add_special_tokens=False)['input_ids']
        
        # Combine prompt and solution
        input_ids = prompt_tokens + solution_tokens
        
        # Pad to max_length
        if len(input_ids) < max_length:
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        elif len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Set prefix_cutoff to the length of the prompt
        prefix_cutoff = len(prompt_tokens)
        
        return {
            'input_ids': input_ids,
            'prefix_cutoff': prefix_cutoff
        }
    
    processed_ds = ds.map(process_sample, remove_columns=ds.column_names)
    return processed_ds