import json
from datetime import datetime
from openai import OpenAI
import time

# Configuration
API_KEY = "<DeepSeek API Key>"  # Replace with your actual API key
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# Configurable parameters
SYSTEM_PROMPT = "You are a helpful assistant that analyzes sentiment in Tamil language text"
INPUT_FILE = "tamil_sentiment_preprocessed.json"
OUTPUT_FILE = "ENTACS_sentiment_translated.json"
MAX_LINES_TO_PROCESS = 1000  # Set to None to process all lines
RATE_LIMIT_DELAY = 0.5  # Seconds to wait between API calls to avoid rate limiting

# Initialize the client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_questions_from_json(filename):
    """
    Load evaluation questions from a JSON file.
    
    Args:
        filename (str): Path to the JSON file
        
    Returns:
        list: List of questions/text lines
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        print(f"Successfully loaded {len(questions)} questions from {filename}")
        return questions
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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=False
        )
        
        result = {
            "original_text": user_text,
            "model_response": response.choices[0].message.content,
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
            "timestamp": datetime.now().isoformat()
        }

def process_texts(system_prompt, texts, max_lines=None):
    """
    Process a list of texts with the given system prompt.
    
    Args:
        system_prompt (str): The system prompt to use
        texts (list): List of texts to process
        max_lines (int, optional): Maximum number of lines to process
        
    Returns:
        list: All processing results
    """
    results = []
    
    # Limit the number of texts to process if specified
    if max_lines is not None and max_lines > 0:
        texts = texts[:max_lines]
    
    total_texts = len(texts)
    
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{total_texts}: \"{text[:30]}...\"")
        
        result = make_api_call(system_prompt, text)
        results.append(result)
        
        # Display progress
        if (i + 1) % 10 == 0 or i == total_texts - 1:
            print(f"Progress: {i+1}/{total_texts} ({(i+1)/total_texts*100:.1f}%)")
        
        # Save intermediate results every 100 items
        if (i + 1) % 100 == 0:
            save_results(results, OUTPUT_FILE)
            print(f"Saved intermediate results to {OUTPUT_FILE}")
        
        # Add delay to avoid rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    return results

def save_results(results, filename):
    """Save results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")

def main():
    print(f"Loading texts from {INPUT_FILE}...")
    texts = load_questions_from_json(INPUT_FILE)
    
    if not texts:
        print("No texts loaded. Exiting.")
        return
    
    print(f"\nProcessing with system prompt: \"{SYSTEM_PROMPT}\"")
    print(f"Max lines to process: {MAX_LINES_TO_PROCESS if MAX_LINES_TO_PROCESS is not None else 'All'}")
    
    # Get start time for estimating completion
    start_time = time.time()
    
    # Process texts
    results = process_texts(SYSTEM_PROMPT, texts, MAX_LINES_TO_PROCESS)
    
    # Calculate and display statistics
    elapsed_time = time.time() - start_time
    total_processed = len(results)
    avg_time_per_item = elapsed_time / total_processed if total_processed > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Processed {total_processed} texts in {elapsed_time:.2f} seconds")
    print(f"Average time per text: {avg_time_per_item:.2f} seconds")
    
    # Save final results
    save_results(results, OUTPUT_FILE)

if __name__ == "__main__":
    main()