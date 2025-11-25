import json
import csv

def count_switches(labels):
    """
    Count the number of codeswitches in an utterance.
    Switch +=1 when consecutive labels differ (excluding 'na').
    """
    if len(labels) < 2:
        return 0
    
    switches = 0
    prev_lang = None
    
    for label in labels:
        # Skip 'na' labels - they don't count as a language
        if label == 'na':
            continue
        
        # If we have a previous language and it's different, count a switch
        if prev_lang is not None and label != prev_lang:
            switches += 1
        
        prev_lang = label
    
    return switches

def combine_unknown_sentiments(sentiment):
    """Combine 'not-Tamil' and 'unknown_state' sentiment labels into 'unknown' label."""
    if sentiment in ['not-Tamil', 'unknown_state']:
        return 'unknown'
    return sentiment

def prepare_dataset(predictions_file, output_csv):
    """
    Load predictions and create R-ready dataset with calculated features.
    """
    
    print("Loading predictions...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} sentences...")
    
    # Prepare rows for CSV
    rows = []
    
    for item in data:
        sentence_id = item['sentence_id']
        sentiment = combine_unknown_sentiments(item['sentiment'])
        tokens = item['tokens']
        labels = item['labels']
        
        # Skip if no tokens
        if not tokens or not labels:
            continue
        
        # Calculate features
        sentence_length = len(tokens)
        num_switches = count_switches(labels)
        en_proportion = item['en_proportion']
        ta_proportion = item['ta_proportion']
        na_proportion = item['na_proportion']
        
        rows.append({
            'sentence_id': sentence_id,
            'sentiment': sentiment,
            'en_proportion': en_proportion,
            'ta_proportion': ta_proportion,
            'na_proportion': na_proportion,
            'num_switches': num_switches,
            'sentence_length': sentence_length,
            'sentence': item['sentence']  # Keep for reference
        })
    
    # Write to CSV
    print(f"Writing {len(rows)} rows to {output_csv}...")
    fieldnames = ['sentence_id', 'sentiment', 'en_proportion', 'ta_proportion', 
                  'na_proportion', 'num_switches', 'sentence_length', 'sentence']
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    # Count by sentiment
    from collections import Counter
    sentiment_counts = Counter(row['sentiment'] for row in rows)
    
    print("\nSentiment Distribution:")
    for sentiment in sorted(sentiment_counts.keys()):
        count = sentiment_counts[sentiment]
        pct = count / len(rows) * 100
        print(f"  {sentiment}: {count:,} ({pct:.1f}%)")
    
    # Overall statistics
    import numpy as np
    
    en_props = [row['en_proportion'] for row in rows]
    ta_props = [row['ta_proportion'] for row in rows]
    na_props = [row['na_proportion'] for row in rows]
    switches = [row['num_switches'] for row in rows]
    lengths = [row['sentence_length'] for row in rows]
    
    print("\nOverall Statistics:")
    print(f"  English proportion - Mean: {np.mean(en_props):.3f}, SD: {np.std(en_props):.3f}")
    print(f"  Tamil proportion   - Mean: {np.mean(ta_props):.3f}, SD: {np.std(ta_props):.3f}")
    print(f"  NA proportion      - Mean: {np.mean(na_props):.3f}, SD: {np.std(na_props):.3f}")
    print(f"  Switches per sent. - Mean: {np.mean(switches):.2f}, SD: {np.std(switches):.2f}")
    print(f"  Sentence length    - Mean: {np.mean(lengths):.2f}, SD: {np.std(lengths):.2f}")
    
    # Statistics by sentiment
    print("\nEnglish Proportion by Sentiment:")
    for sentiment in sorted(sentiment_counts.keys()):
        sent_en = [row['en_proportion'] for row in rows if row['sentiment'] == sentiment]
        print(f"  {sentiment}: Mean={np.mean(sent_en):.3f}, SD={np.std(sent_en):.3f}")
    
    print("\nSwitches per Sentence by Sentiment:")
    for sentiment in sorted(sentiment_counts.keys()):
        sent_sw = [row['num_switches'] for row in rows if row['sentiment'] == sentiment]
        print(f"  {sentiment}: Mean={np.mean(sent_sw):.2f}, SD={np.std(sent_sw):.2f}")
    
    print("\n" + "="*70)
    print(f"Dataset saved to: {output_csv}")
    print("Ready for R analysis!")
    print("="*70)
    
    # Show sample rows
    print("\nSample rows (first 5):")
    for i, row in enumerate(rows[:5], 1):
        print(f"\n{i}. [{row['sentiment']}] {row['sentence'][:60]}...")
        print(f"   Length: {row['sentence_length']}, Switches: {row['num_switches']}, "
              f"EN: {row['en_proportion']:.2f}, TA: {row['ta_proportion']:.2f}")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_r_dataset.py <predictions.json> [output.csv]")
        print("\nExample:")
        print("  python prepare_r_dataset.py langid_predictions.json codeswitching_data.csv")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "codeswitching_data.csv"
    
    prepare_dataset(predictions_file, output_csv)

if __name__ == "__main__":
    main()