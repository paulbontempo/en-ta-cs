import csv
import re
import random
import json

def extract_words_without_punctuation(sentence):
    """Extract words from sentence, removing all punctuation."""
    tokens = sentence.split()
    words = []
    
    for token in tokens:
        word = re.sub(r'[^\w]', '', token, flags=re.UNICODE)
        if word:
            words.append(word)
    
    return words


def create_annotation_template(input_json, output_csv, num_sentences=500):
    """Create a CSV template for easy annotation from JSON array."""
    
    # Read input sentences from JSON
    with open(input_json, 'r', encoding='utf-8') as f:
        all_sentences = json.load(f)
    
    print(f"Loaded {len(all_sentences)} sentences from JSON")
    
    # Sample random sentences
    if len(all_sentences) > num_sentences:
        selected_sentences = random.sample(all_sentences, num_sentences)
    else:
        selected_sentences = all_sentences
        print(f"Warning: Only {len(all_sentences)} sentences available, using all of them")
    
    # Create annotation template with generated IDs
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sentence_id', 'sentence', 'tokens', 'num_tokens', 'labels (space-separated)'])
        
        for idx, sentence in enumerate(selected_sentences, start=1):
            words = extract_words_without_punctuation(sentence)
            tokens_str = ' '.join(words)
            num_tokens = len(words)
            # Generate simple sequential ID
            sentence_id = f"sent_{idx:04d}"
            # Empty labels - to be filled by annotator
            writer.writerow([sentence_id, sentence, tokens_str, num_tokens, ''])
    
    print(f"Created annotation template with {len(selected_sentences)} sentences")
    print(f"Saved to: {output_csv}")
    print("\nInstructions:")
    print("1. Open the CSV in Excel/Google Sheets")
    print("2. For each row, fill in the 'labels' column")
    print("3. Use: en (English), ta (Tamil), na (numbers/ambiguous)")
    print("4. Separate labels with spaces, one per token")
    print("5. Example: 'en ta en ta en'")


def convert_csv_to_conll(annotated_csv, output_conll):
    """Convert annotated CSV to CoNLL format."""
    
    with open(annotated_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    skipped = 0
    written = 0
    
    with open(output_conll, 'w', encoding='utf-8') as f:
        for row in rows:
            tokens = row['tokens'].split()
            labels = row['labels (space-separated)'].strip().split()
            
            # Skip if not annotated or mismatch
            if not labels or not row['labels (space-separated)'].strip():
                skipped += 1
                continue
            
            if len(labels) != len(tokens):
                print(f"Warning: Skipping sentence {row['sentence_id']} - {len(tokens)} tokens but {len(labels)} labels")
                skipped += 1
                continue
            
            # Write in CoNLL format
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")  # Blank line between sentences
            written += 1
    
    print(f"Converted {written} sentences to CoNLL format")
    if skipped > 0:
        print(f"Skipped {skipped} sentences (not annotated or label mismatch)")
    print(f"Saved to: {output_conll}")


def split_train_val(conll_file, train_file, val_file, val_ratio=0.1):
    """Split CoNLL file into train and validation sets."""
    
    # Read all sentences
    sentences = []
    current_sentence = []
    
    with open(conll_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
        if current_sentence:
            sentences.append(current_sentence)
    
    # Shuffle and split
    random.shuffle(sentences)
    split_idx = int(len(sentences) * (1 - val_ratio))
    train_sentences = sentences[:split_idx]
    val_sentences = sentences[split_idx:]
    
    # Write train file
    with open(train_file, 'w', encoding='utf-8') as f:
        for sentence in train_sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')
    
    # Write validation file
    with open(val_file, 'w', encoding='utf-8') as f:
        for sentence in val_sentences:
            for line in sentence:
                f.write(line + '\n')
            f.write('\n')
    
    print(f"Split {len(sentences)} sentences:")
    print(f"  Train: {len(train_sentences)} sentences -> {train_file}")
    print(f"  Val: {len(val_sentences)} sentences -> {val_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Step 1 - Create annotation template:")
        print("    python annotation_helper.py create [sentences.json] [annotation_template.csv] [num_sentences]")
        print("  Step 2 - Convert annotated CSV to CoNLL:")
        print("    python annotation_helper.py convert [annotation_template.csv] [output.conll]")
        print("  Step 3 - Split into train/val:")
        print("    python annotation_helper.py split [output.conll] [train.conll] [val.conll]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        input_json = sys.argv[2]
        output_csv = sys.argv[3]
        num_sentences = int(sys.argv[4]) if len(sys.argv) > 4 else 500
        create_annotation_template(input_json, output_csv, num_sentences)
    
    elif command == "convert":
        annotated_csv = sys.argv[2]
        output_conll = sys.argv[3]
        convert_csv_to_conll(annotated_csv, output_conll)
    
    elif command == "split":
        conll_file = sys.argv[2]
        train_file = sys.argv[3]
        val_file = sys.argv[4]
        split_train_val(conll_file, train_file, val_file)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)