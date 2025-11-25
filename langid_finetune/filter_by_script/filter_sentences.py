import json
import re
import csv

def contains_tamil_script(text):
    """Check if text contains Tamil Unicode characters (U+0B80 to U+0BFF)."""
    tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
    return bool(tamil_pattern.search(text))

def filter_sentences(input_csv, output_romanized, output_tamil_script, stats_file):
    """
    Split sentences into romanized-only and Tamil-script-containing groups.
    Preserves sentiment labels and generates unique IDs.
    
    Args:
        input_csv: Input CSV/TSV with sentiment label and sentence
        output_romanized: Output JSON for sentences without Tamil script
        output_tamil_script: Output JSON for sentences with Tamil script
        stats_file: Output file with statistics
    """
    
    print("Loading sentences from CSV/TSV...")
    data = []
    
    # Try tab-separated first (your format), then comma-separated
    for delimiter in ['\t', ',']:
        try:
            with open(input_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=delimiter)
                data = []
                for row in reader:
                    if len(row) >= 2:
                        sentiment = row[0].strip()
                        sentence = row[1].strip()
                        data.append({'sentiment': sentiment, 'sentence': sentence})
            if data:
                print(f"Successfully parsed with delimiter: '{delimiter}'")
                break
        except Exception as e:
            continue
    
    if not data:
        print("ERROR: Could not parse CSV/TSV file")
        return
    
    print(f"Total sentences: {len(data)}")
    
    romanized = []
    tamil_script = []
    
    print("Filtering sentences and generating IDs...")
    for idx, item in enumerate(data, start=1):
        sentence_id = f"sent_{idx:05d}"
        entry = {
            "sentence_id": sentence_id,
            "sentiment": item['sentiment'],
            "sentence": item['sentence']
        }
        
        if contains_tamil_script(item['sentence']):
            tamil_script.append(entry)
        else:
            romanized.append(entry)
    
    # Save romanized sentences
    print(f"\nSaving {len(romanized)} romanized sentences to {output_romanized}...")
    with open(output_romanized, 'w', encoding='utf-8') as f:
        json.dump(romanized, f, ensure_ascii=False, indent=2)
    
    # Save Tamil script sentences
    print(f"Saving {len(tamil_script)} Tamil script sentences to {output_tamil_script}...")
    with open(output_tamil_script, 'w', encoding='utf-8') as f:
        json.dump(tamil_script, f, ensure_ascii=False, indent=2)
    
    # Calculate sentiment distribution
    romanized_sentiments = {}
    tamil_sentiments = {}
    
    for item in romanized:
        sentiment = item['sentiment']
        romanized_sentiments[sentiment] = romanized_sentiments.get(sentiment, 0) + 1
    
    for item in tamil_script:
        sentiment = item['sentiment']
        tamil_sentiments[sentiment] = tamil_sentiments.get(sentiment, 0) + 1
    
    # Save statistics
    stats = {
        "total_sentences": len(data),
        "romanized_only": len(romanized),
        "contains_tamil_script": len(tamil_script),
        "romanized_percentage": round(len(romanized) / len(data) * 100, 2),
        "tamil_script_percentage": round(len(tamil_script) / len(data) * 100, 2),
        "romanized_sentiment_distribution": romanized_sentiments,
        "tamil_script_sentiment_distribution": tamil_sentiments
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {stats_file}")
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  Total sentences: {stats['total_sentences']:,}")
    print(f"  Romanized only: {stats['romanized_only']:,} ({stats['romanized_percentage']}%)")
    print(f"  Contains Tamil script: {stats['contains_tamil_script']:,} ({stats['tamil_script_percentage']}%)")
    print(f"\n  Romanized sentiment distribution:")
    for sentiment, count in romanized_sentiments.items():
        print(f"    {sentiment}: {count:,}")
    print(f"{'='*50}")
    
    # Show some examples
    print("\nSample romanized sentences (first 3):")
    for i, item in enumerate(romanized[:3], 1):
        print(f"  {i}. [{item['sentiment']}] {item['sentence']}")
    
    print("\nSample Tamil script sentences (first 3):")
    for i, item in enumerate(tamil_script[:3], 1):
        print(f"  {i}. [{item['sentiment']}] {item['sentence']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 5:
        print("Usage: python filter_sentences.py <input.csv> <romanized_output.json> <tamil_script_output.json> <stats.json>")
        print("\nInput CSV/TSV format (tab or comma separated):")
        print("  sentiment<TAB>sentence")
        print("  Negative<TAB>Enna da ellam avan seyal Mari iruku")
        print("\nExample:")
        print("  python filter_sentences.py data.csv romanized_sentences.json tamil_script_sentences.json filter_stats.json")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_romanized = sys.argv[2]
    output_tamil_script = sys.argv[3]
    stats_file = sys.argv[4]
    
    filter_sentences(input_csv, output_romanized, output_tamil_script, stats_file)
    
    print("\nDone! You can now run inference on the romanized sentences.")
    print("The sentiment labels are preserved in the JSON output.")