import re

def contains_tamil_script(text):
    """Check if text contains Tamil Unicode characters (U+0B80 to U+0BFF)."""
    tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
    return bool(tamil_pattern.search(text))

def clean_conll_file(input_file, output_file):
    """
    Remove sentences containing Tamil script from CoNLL file.
    
    Args:
        input_file: Input CoNLL file with potential Tamil script
        output_file: Output CoNLL file with only romanized text
    """
    
    current_sentence = []
    clean_sentences = []
    removed_sentences = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Empty line = end of sentence
                if current_sentence:
                    # Check if sentence contains Tamil script
                    has_tamil_script = any(contains_tamil_script(word) 
                                          for word, _ in current_sentence)
                    
                    if has_tamil_script:
                        removed_sentences.append(current_sentence)
                    else:
                        clean_sentences.append(current_sentence)
                    
                    current_sentence = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    word, label = parts
                    current_sentence.append((word, label))
        
        # Don't forget last sentence
        if current_sentence:
            has_tamil_script = any(contains_tamil_script(word) 
                                  for word, _ in current_sentence)
            if has_tamil_script:
                removed_sentences.append(current_sentence)
            else:
                clean_sentences.append(current_sentence)
    
    # Write cleaned sentences
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in clean_sentences:
            for word, label in sentence:
                f.write(f"{word}\t{label}\n")
            f.write("\n")  # Blank line between sentences
    
    return len(clean_sentences), len(removed_sentences)

def main():
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python clean_conll.py <input.conll> <output.conll>")
        print("\nExample:")
        print("  python clean_conll.py train.conll train_clean.conll")
        print("  python clean_conll.py val.conll val_clean.conll")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Cleaning {input_file}...")
    kept, removed = clean_conll_file(input_file, output_file)
    
    print(f"\nResults:")
    print(f"  Kept (romanized): {kept} sentences")
    print(f"  Removed (Tamil script): {removed} sentences")
    print(f"  Total: {kept + removed} sentences")
    print(f"  Removal rate: {removed / (kept + removed) * 100:.1f}%")
    print(f"\nCleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()