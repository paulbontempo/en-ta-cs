import json
from collections import Counter

def load_predictions(predictions_file):
    """Load predictions from JSON."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_predictions(data):
    """Generate quick evaluation metrics."""
    
    print("="*70)
    print("INFERENCE EVALUATION METRICS")
    print("="*70)
    
    # Overall statistics
    total_sentences = len(data)
    total_tokens = sum(len(item['tokens']) for item in data)
    
    # Label distribution
    all_labels = []
    for item in data:
        all_labels.extend(item['labels'])
    
    label_counts = Counter(all_labels)
    
    print(f"\nDataset Overview:")
    print(f"  Total sentences: {total_sentences:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens per sentence: {total_tokens / total_sentences:.1f}")
    
    print(f"\nLabel Distribution:")
    for label in ['en', 'ta', 'na']:
        count = label_counts[label]
        pct = count / total_tokens * 100
        print(f"  {label.upper()}: {count:,} ({pct:.1f}%)")
    
    # Proportion statistics
    en_props = [item['en_proportion'] for item in data]
    ta_props = [item['ta_proportion'] for item in data]
    na_props = [item['na_proportion'] for item in data]
    
    import numpy as np
    
    print(f"\nEnglish Proportion Statistics:")
    print(f"  Mean: {np.mean(en_props):.2%}")
    print(f"  Median: {np.median(en_props):.2%}")
    print(f"  Std Dev: {np.std(en_props):.2%}")
    print(f"  Min: {np.min(en_props):.2%}")
    print(f"  Max: {np.max(en_props):.2%}")
    
    print(f"\nTamil Proportion Statistics:")
    print(f"  Mean: {np.mean(ta_props):.2%}")
    print(f"  Median: {np.median(ta_props):.2%}")
    print(f"  Std Dev: {np.std(ta_props):.2%}")
    print(f"  Min: {np.min(ta_props):.2%}")
    print(f"  Max: {np.max(ta_props):.2%}")
    
    # Code-switching analysis
    purely_english = sum(1 for item in data if item['en_proportion'] == 1.0)
    purely_tamil = sum(1 for item in data if item['ta_proportion'] == 1.0)
    code_switched = total_sentences - purely_english - purely_tamil
    
    print(f"\nCode-Switching Analysis:")
    print(f"  Purely English: {purely_english:,} ({purely_english/total_sentences*100:.1f}%)")
    print(f"  Purely Tamil: {purely_tamil:,} ({purely_tamil/total_sentences*100:.1f}%)")
    print(f"  Code-switched: {code_switched:,} ({code_switched/total_sentences*100:.1f}%)")
    
    # Find interesting examples
    print(f"\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    # Sort by English proportion
    sorted_by_en = sorted(data, key=lambda x: x['en_proportion'])
    
    print(f"\nMost Tamil sentences:")
    for item in sorted_by_en[:5]:
        print(f"  EN:{item['en_proportion']:.0%} | {item['sentence'][:60]}...")
        print(f"    Tokens: {' '.join(item['tokens'][:10])}...")
        print(f"    Labels: {' '.join(item['labels'][:10])}...")
        print()
    
    print(f"\nMost English sentences:")
    for item in sorted_by_en[-5:]:
        print(f"  EN:{item['en_proportion']:.0%} | {item['sentence'][:60]}...")
        print(f"    Tokens: {' '.join(item['tokens'][:10])}...")
        print(f"    Labels: {' '.join(item['labels'][:10])}...")
        print()
    
    # Balanced code-switched examples
    mid_idx = len(sorted_by_en) // 2
    print(f"\nBalanced code-switched sentences (~50% each language):")
    balanced = [item for item in data 
                if 0.4 <= item['en_proportion'] <= 0.6][:5]
    for item in balanced:
        print(f"  EN:{item['en_proportion']:.0%} TA:{item['ta_proportion']:.0%} | {item['sentence'][:50]}...")
        print(f"    Tokens: {' '.join(item['tokens'][:10])}...")
        print(f"    Labels: {' '.join(item['labels'][:10])}...")
        print()
    
    # Sentiment distribution
    if 'sentiment' in data[0]:
        print(f"\n" + "="*70)
        print("SENTIMENT DISTRIBUTION")
        print("="*70)
        
        sentiment_counts = Counter(item['sentiment'] for item in data)
        for sentiment, count in sorted(sentiment_counts.items()):
            pct = count / total_sentences * 100
            print(f"  {sentiment}: {count:,} ({pct:.1f}%)")
    
    print(f"\n" + "="*70)

def check_quality(data, n_samples=20):
    """Manually inspect random samples for quality check."""
    import random
    
    print("\n" + "="*70)
    print(f"QUALITY CHECK - Random {n_samples} Samples")
    print("="*70)
    print("\nManually review these predictions for accuracy:")
    print("(Look for obvious errors like English words labeled as Tamil)\n")
    
    samples = random.sample(data, min(n_samples, len(data)))
    
    for i, item in enumerate(samples, 1):
        print(f"{i}. {item['sentence']}")
        tokens_labels = [f"{tok}[{lbl}]" for tok, lbl in 
                        zip(item['tokens'], item['labels'])]
        print(f"   {' '.join(tokens_labels)}")
        print()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_xlmr.py <predictions.json>")
        print("\nExample:")
        print("  python evaluate_xlmr.py langid_predictions.json")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    
    print("Loading predictions...")
    data = load_predictions(predictions_file)
    
    # Run evaluation
    evaluate_predictions(data)
    
    # Quality check
    check_quality(data, n_samples=20)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the random samples above for obvious errors")
    print("2. Run sentiment analysis: python analyze_sentiment.py langid_predictions.json analysis")
    print("3. If results look good, proceed with your research analysis!")

if __name__ == "__main__":
    main()
