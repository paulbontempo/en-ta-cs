
''' 
Preprocess the codeswitched text from tamil_sentiment_full.csv, 
part of the Dravidian Code-Mix corpus created by Chakravarthi et al 
'''
import pandas as pd
import re
import json
import emoji
import argparse
from typing import List, Optional

def preprocess_text(
    text: str,
    remove_emojis: bool = True,
    standardize_punctuation: bool = True,
    normalize_whitespace: bool = True
) -> str:
    """
    Preprocess text by standardizing punctuation, removing emojis, and normalizing whitespace.
    
    Args:
        text: The input text to preprocess
        remove_emojis: Whether to remove emojis
        standardize_punctuation: Whether to standardize punctuation
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        # Handle non-string inputs by converting to string or returning empty string
        return str(text) if text is not None else ""
    
    # Remove emojis
    if remove_emojis:
        # Using emoji library to remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Also remove common ASCII emoticons
        # emoticon_pattern = r'[:;=]-?[)(/\\|DPdpO3]|[)(\\|DPdpO0></\\]+|¯\\_\(ツ\)_/¯'
        # text = re.sub(emoticon_pattern, '', text)
    
    # Standardize punctuation
    if standardize_punctuation:
        # Reduce multiple consecutive instances of the same punctuation to a single instance
        # except periods (to preserve ellipses)
        punctuation_pattern = r'([!?#$%&*+,\-/:;<=>@^_`{|}~\"\'])\1+'
        text = re.sub(punctuation_pattern, r'\1', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Replace multiple whitespace characters with a single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
    return text

def process_csv_to_json(
    input_file: str,
    output_file: str,
    column: int,
    has_header: bool = False,
    remove_emojis: bool = True,
    standardize_punctuation: bool = True,
    normalize_whitespace: bool = True
) -> None:
    """
    Process a CSV file, extract and preprocess a single column, and output to JSON as a list of strings.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output JSON file
        column: Column index to extract from the CSV (0-based)
        has_header: Whether the CSV file has a header row
        remove_emojis: Whether to remove emojis
        standardize_punctuation: Whether to standardize punctuation
        normalize_whitespace: Whether to normalize whitespace
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file, header=0 if has_header else None, delimiter='\t')  # for tab-delimited files
        
        # Validate column
        if column >= len(df.columns):
            raise ValueError(f"Column index {column} out of range. CSV has {len(df.columns)} columns.")
        
        # Extract and preprocess the specified column
        result = []
        for _, row in df.iterrows():
            processed_text = preprocess_text(
                row[column], 
                remove_emojis=remove_emojis,
                standardize_punctuation=standardize_punctuation,
                normalize_whitespace=normalize_whitespace
            )
            if processed_text:  # Only add non-empty strings
                result.append(processed_text)
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully processed {len(result)} text items to {output_file}")
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess CSV text data from a single column and output to JSON as a list of strings')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output JSON file')
    parser.add_argument('--column', required=True, type=int, help='Column index to extract and preprocess (0-based)')
    parser.add_argument('--has-header', action='store_true', help='CSV file has a header row')
    parser.add_argument('--keep-emojis', action='store_true', help='Keep emojis in the text')
    parser.add_argument('--keep-punctuation', action='store_true', help='Keep original punctuation')
    parser.add_argument('--keep-whitespace', action='store_true', help='Keep original whitespace')
    
    args = parser.parse_args()
    
    process_csv_to_json(
        args.input_file,
        args.output_file,
        args.column,
        has_header=args.has_header,
        remove_emojis=not args.keep_emojis,
        standardize_punctuation=not args.keep_punctuation,
        normalize_whitespace=not args.keep_whitespace
    )

if __name__ == '__main__':
    main()