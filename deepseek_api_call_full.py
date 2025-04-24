import json
from datetime import datetime
from openai import OpenAI
import time

# Configuration
API_KEY = "sk-2c55f815f11a4bb3a3aacfb7a2f99228"  # Replace with your actual API key
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
TEMPERATURE = 1.3

# Configurable parameters
SYSTEM_PROMPT = '''You are a fluent bilingual speaker of the English and Tamil languages, capable of translation between them. 
You are capable of code-switching, the common linguistic practice of alternating between two languages in the same utterance. 
Your task is to translate a given code-switched English and Tamil input sentence into a monolingual English version of the same sentence.
In your response, only include the monolingual translation, with no extra text or non-English words. All Tamil words should be translated into English. 
Learn from the following example of conversion from Tamil-English code-switching to monolingual English:
1. code-switched: "Trailer pakka super ah iruku but Rajesh neenga edhutha thungavanam padatha patha dha bayama iruku.." monolingual English: "The trailer looks nice but I am getting scared for the movie Thungavanam you've directed Rajesh"
Translate the following code-switched text into monolingual English:
'''
INPUT_FILE = "tamil_sentiment_preprocessed.json"
OUTPUT_FILE = "ENTACS_sentiment_translated.json"

# Line span configuration - ADJUST THESE VALUES TO PROCESS DIFFERENT CHUNKS
START_LINE = 1  # Starting line index (1-based for user convenience)
END_LINE = 10  # Ending line index (inclusive)

RATE_LIMIT_DELAY = 0  # Seconds to wait between API calls to avoid rate limiting

# Initialize the client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_questions_from_json(filename, start_line, end_line):
    """
    Load a specific span of questions from a JSON file.
    
    Args:
        filename (str): Path to the JSON file
        start_line (int): 1-based index of the first line to process
        end_line (int): 1-based index of the last line to process (inclusive)
        
    Returns:
        list: List of selected questions/text lines
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            all_questions = json.load(f)
        
        # Convert to 0-based indexing and get the slice
        start_idx = max(0, start_line - 1)  # Prevent negative index
        end_idx = min(len(all_questions), end_line)  # Prevent out of bounds
        
        selected_questions = all_questions[start_idx:end_idx]
        
        print(f"Successfully loaded {len(selected_questions)} questions from {filename}")
        print(f"Processing lines {start_line} to {min(end_line, len(all_questions))} " +
              f"out of {len(all_questions)} total lines")
        
        return selected_questions
    except Exception as e:
        print(f"Error loading questions from {filename}: {e}")
        return []

def make_api_call(system_prompt, user_text):
    """
    Make an API call to the LLM with the provided system prompt and user text.
    
    Args:
        system_prompt (str): The system prompt to use
        user_text (str): The user text to evaluate
        
    Returns:
        dict: Response data including the model's response and metadata
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=False
        )
        
        result = {
            "original_text": user_text,
            "model_response": response.choices[0].message.content,
            "line_number": START_LINE + texts.index(user_text),  # Store the absolute line number
            "timestamp": datetime.now().isoformat(),
            "finish_reason": response.choices[0].finish_reason
        }
        
        # Include usage stats if available
        if hasattr(response, 'usage') and response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return result
    
    except Exception as e:
        print(f"Error making API call: {e}")
        return {
            "original_text": user_text,
            "error": str(e),
            "line_number": START_LINE + texts.index(user_text),
            "timestamp": datetime.now().isoformat()
        }

def process_texts(system_prompt, texts):
    """
    Process a list of texts with the given system prompt.
    
    Args:
        system_prompt (str): The system prompt to use
        texts (list): List of texts to process
        
    Returns:
        list: All processing results
    """
    results = []
    total_texts = len(texts)
    
    for i, text in enumerate(texts):
        print(f"Processing text {START_LINE+i} (chunk item {i+1}/{total_texts}): \"{text[:10]}...\"")
        
        result = make_api_call(system_prompt, text)
        results.append(result)
        
        # Display progress
        if (i + 1) % 10 == 0 or i == total_texts - 1:
            print(f"Progress: {i+1}/{total_texts} ({(i+1)/total_texts*100:.1f}%)")
        
        # Save intermediate results every 20 items
        if (i + 1) % 20 == 0:
            save_results(results, get_chunk_output_filename())
            print(f"Saved intermediate results to {get_chunk_output_filename()}")
        
        # Add delay to avoid rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    return results

def get_chunk_output_filename():
    """Generate a filename that includes the chunk range"""
    base_name = OUTPUT_FILE.split('.json')[0]
    return f"{base_name}_{START_LINE}_to_{END_LINE}.json"

def save_results(results, filename):
    """Save results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")

def append_to_master_file(chunk_results):
    """
    Append chunk results to the master output file if it exists
    
    Args:
        chunk_results (list): Results from the current chunk
    """
    try:
        # Try to read existing results
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_results = []
            
        # Append new results
        combined_results = existing_results + chunk_results
        
        # Save the combined results
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
        print(f"Appended {len(chunk_results)} results to master file {OUTPUT_FILE}")
        print(f"Master file now contains {len(combined_results)} total results")
    except Exception as e:
        print(f"Error appending to master file: {e}")
        print("Results are still saved in the chunk-specific file")

def main():
    global texts  # Make texts available for line number calculation in make_api_call
    
    print(f"Loading texts from {INPUT_FILE} (lines {START_LINE} to {END_LINE})...")
    texts = load_questions_from_json(INPUT_FILE, START_LINE, END_LINE)
    
    if not texts:
        print("No texts loaded. Exiting.")
        return
    
    #print(f"\nProcessing with system prompt: \"{SYSTEM_PROMPT}\"")
    
    # Get start time for estimating completion
    start_time = time.time()
    
    # Process texts
    results = process_texts(SYSTEM_PROMPT, texts)
    
    # Calculate and display statistics
    elapsed_time = time.time() - start_time
    total_processed = len(results)
    avg_time_per_item = elapsed_time / total_processed if total_processed > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Processed {total_processed} texts in {elapsed_time:.2f} seconds")
    print(f"Average time per text: {avg_time_per_item:.2f} seconds")
    
    # Save final chunk results
    chunk_filename = get_chunk_output_filename()
    save_results(results, chunk_filename)
    
    # Append to master file
    append_to_master_file(results)
    
    # Suggest next chunk to process
    next_start = END_LINE + 1
    next_end = END_LINE + (END_LINE - START_LINE + 1)
    print(f"\nTo process the next chunk, set:")
    print(f"START_LINE = {next_start}")
    print(f"END_LINE = {next_end}")

if __name__ == "__main__":
    main()