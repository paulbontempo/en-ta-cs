import pandas as pd
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings

# Suppress numpy compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning)

# NumPy compatibility fix
import numpy as np
if hasattr(np, '__version__') and int(np.__version__.split('.')[0]) >= 2:
    np.set_printoptions(legacy='1.25')

# Configuration
FEW_SHOT_CSV = "few_shot.csv"
INPUT_CSV = "test_input_data.csv"
OUTPUT_JSON = "test_results.json"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Better at instruction following

# H100 Configuration
DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1  # Low temperature for consistent outputs


def load_few_shot_examples(csv_path):
    """Load few-shot examples from CSV and format them for prompting."""
    # Read CSV with basic Python to avoid pandas/numpy issues
    examples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Skip header
    for line in lines[1:]:
        # Split by comma, handling the trailing comma
        parts = line.strip().rstrip(',').split(',')
        
        if len(parts) >= 3:
            sentence_id = parts[0]
            sentence = parts[1]
            labels = parts[2]
            
            # Skip empty rows
            if sentence and sentence.strip():
                examples.append(f'Input: "{sentence}"\nOutput: {labels}')
    
    return examples


def create_few_shot_prompt(examples, test_sentence):
    """Create the few-shot prompt with instructions."""
    system_instruction = """Task: Label each word in code-switched English-Tamil text.

CRITICAL INSTRUCTIONS:
- Output ONLY space-separated labels for WORDS (not punctuation)
- Completely IGNORE all punctuation marks - do not label them
- Count only whitespace-separated words after removing punctuation
- Labels: "en" (English), "ta" (Tamil/romanized/Tamil script), "na" (numbers/ambiguous)
- Do NOT write code or explanations
- Do NOT use markdown or formatting
- Output format: just the labels separated by spaces (e.g., "en ta en ta")

Examples:"""
    
    # Add few-shot examples (limit to 5 for context)
    prompt = system_instruction + "\n\n" + "\n\n".join(examples[:5])
    
    # Add the test sentence with word count hint
    words_in_sentence = len(extract_words_without_punctuation(test_sentence))
    prompt += f'\n\nInput: "{test_sentence}"\nExpected number of labels: {words_in_sentence}\nOutput:'
    
    return prompt


def extract_words_without_punctuation(sentence):
    """Extract words from sentence, preserving order but tracking which had punctuation."""
    # Split by whitespace first
    tokens = sentence.split()
    words = []
    
    for token in tokens:
        # Remove all punctuation from the token, but keep Unicode word characters
        word = re.sub(r'[^\w]', '', token, flags=re.UNICODE)
        if word:  # Only add if something remains after removing punctuation
            words.append(word)
    
    return words


def parse_model_output(output_text, expected_count):
    """Parse model output to extract labels, handling various formats."""
    # Extract everything after "Output:" if present
    if "Output:" in output_text:
        output_text = output_text.split("Output:")[-1]
    
    # Clean and split
    output_text = output_text.strip()
    labels = output_text.split()
    
    # Take only valid labels
    valid_labels = ['en', 'ta', 'na']
    parsed_labels = [label.lower() for label in labels if label.lower() in valid_labels]
    
    # Validate count
    if len(parsed_labels) != expected_count:
        print(f"Warning: Expected {expected_count} labels, got {len(parsed_labels)}")
        print(f"Raw output: {output_text}")
        # Pad or truncate to match expected count
        if len(parsed_labels) < expected_count:
            parsed_labels.extend(['na'] * (expected_count - len(parsed_labels)))
        else:
            parsed_labels = parsed_labels[:expected_count]
    
    return parsed_labels


def process_sentences(model, tokenizer, few_shot_examples, input_df):
    """Process all sentences and generate language ID labels."""
    results = []
    
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing sentences"):
        sentence_id = row['sentence id']
        codemixed_text = row['codemixed text'].strip()
        
        # Extract words (without punctuation)
        words = extract_words_without_punctuation(codemixed_text)
        
        if not words:
            results.append({
                "sentence_id": sentence_id,
                "sentence": codemixed_text,
                "tokens": [],
                "labels": []
            })
            continue
        
        # Create prompt
        prompt = create_few_shot_prompt(few_shot_examples, codemixed_text)
        
        # Format for chat template with system message
        messages = [
            {"role": "system", "content": "You are a language labeling system. Respond ONLY with space-separated language labels (en/ta/na). Never write code or explanations."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Debug: Print first 3 examples
        if len(results) < 3:
            print(f"\n=== DEBUG: Sentence {len(results)} ===")
            print(f"Input: {codemixed_text}")
            print(f"Model output: {generated_text}")
        
        # Parse labels
        labels = parse_model_output(generated_text, len(words))
        
        # Debug: Print parsed labels
        if len(results) < 3:
            print(f"Parsed labels: {labels}")
            print(f"Tokens: {words}")
            print(f"Token count: {len(words)}, Label count: {len(labels)}")
            print("="*50)
        
        # Store result
        results.append({
            "sentence_id": sentence_id,
            "sentence": codemixed_text,
            "tokens": words,
            "labels": labels
        })
    
    return results


def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto"
    )
    model.eval()
    
    print("Loading few-shot examples...")
    few_shot_examples = load_few_shot_examples(FEW_SHOT_CSV)
    print(f"Loaded {len(few_shot_examples)} few-shot examples")
    
    print("Loading input sentences...")
    # Read input CSV manually to avoid numpy/pandas compatibility issues
    input_data = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse header to find column indices
    header = lines[0].strip().split(',')
    sentence_id_idx = header.index('sentence id')
    codemixed_idx = header.index('codemixed text')
    
    # Parse data rows
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) > max(sentence_id_idx, codemixed_idx):
            input_data.append({
                'sentence id': parts[sentence_id_idx],
                'codemixed text': parts[codemixed_idx]
            })
    
    print(f"Found {len(input_data)} sentences to process")
    
    print("Processing sentences...")
    results = []
    
    for row in tqdm(input_data, desc="Processing sentences"):
        sentence_id = row['sentence id']
        codemixed_text = row['codemixed text'].strip()
        
        # Extract words (without punctuation)
        words = extract_words_without_punctuation(codemixed_text)
        
        if not words:
            results.append({
                "sentence_id": sentence_id,
                "sentence": codemixed_text,
                "tokens": [],
                "labels": []
            })
            continue
        
        # Create prompt
        prompt = create_few_shot_prompt(few_shot_examples, codemixed_text)
        
        # Format for Llama chat template
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse labels
        labels = parse_model_output(generated_text, len(words))
        
        # Store result
        results.append({
            "sentence_id": sentence_id,
            "sentence": codemixed_text,
            "tokens": words,
            "labels": labels
        })
    
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    print(f"\nProcessed {len(results)} sentences")
    print(f"Results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()