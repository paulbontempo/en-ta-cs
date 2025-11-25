import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import re
from tqdm import tqdm

# Configuration for running on CURC Blanca cluster
MODEL_DIR = "/projects/pabo8622/entacs/xlmr-langid-model" # Fine-tuned model directory
INPUT_JSON = "/projects/pabo8622/entacs/romanized_sentences.json"  # Filtered romanized sentences
OUTPUT_JSON = "/projects/pabo8622/entacs/langid_predictions.json" 


def extract_words_without_punctuation(sentence):
    """Extract words from sentence, removing all punctuation."""
    tokens = sentence.split()
    words = []
    
    for token in tokens:
        word = re.sub(r'[^\w]', '', token, flags=re.UNICODE)
        if word:
            words.append(word)
    
    return words


def predict_labels(model, tokenizer, words, device):
    """Predict language labels for a list of words."""
    
    # Tokenize
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Align predictions with original words
    word_ids = inputs.word_ids(batch_index=0)
    labels = []
    previous_word_idx = None
    
    for word_idx, pred in zip(word_ids, predictions[0].cpu().numpy()):
        if word_idx is not None and word_idx != previous_word_idx:
            # Get label for first subword of each word
            label = model.config.id2label[pred]
            labels.append(label)
            previous_word_idx = word_idx
    
    return labels


def main():
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer from base model to avoid config issues
    print("Loading tokenizer from xlm-roberta-base...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    print("Loading input data...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} sentences...")
    
    results = []
    for item in tqdm(data):
        sentence_id = item['sentence_id']
        sentiment = item['sentiment']
        sentence = item['sentence']
        
        words = extract_words_without_punctuation(sentence)
        
        if not words:
            results.append({
                "sentence_id": sentence_id,
                "sentiment": sentiment,
                "sentence": sentence,
                "tokens": [],
                "labels": [],
                "en_proportion": 0.0,
                "ta_proportion": 0.0,
                "na_proportion": 0.0
            })
            continue
        
        labels = predict_labels(model, tokenizer, words, device)
        
        # Calculate label proportions
        total = len(labels)
        en_count = labels.count('en')
        ta_count = labels.count('ta')
        na_count = labels.count('na')
        
        results.append({
            "sentence_id": sentence_id,
            "sentiment": sentiment,
            "sentence": sentence,
            "tokens": words,
            "labels": labels,
            "en_proportion": round(en_count / total, 4) if total > 0 else 0.0,
            "ta_proportion": round(ta_count / total, 4) if total > 0 else 0.0,
            "na_proportion": round(na_count / total, 4) if total > 0 else 0.0
        })
    
    print(f"Saving results to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Done!")
    print(f"Processed {len(results)} sentences")
    
    # Print sample results
    print("\nSample predictions:")
    for i, result in enumerate(results[:3]):
        print(f"\nSentence {i+1} [{result['sentiment']}]: {result['sentence']}")
        print(f"Tokens: {' '.join(result['tokens'])}")
        print(f"Labels: {' '.join(result['labels'])}")
        print(f"EN: {result['en_proportion']:.2%}, TA: {result['ta_proportion']:.2%}, NA: {result['na_proportion']:.2%}")


if __name__ == "__main__":
    main()