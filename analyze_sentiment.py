import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def load_predictions(predictions_file):
    """Load predictions with sentiment labels."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_by_sentiment(data):
    """Analyze code-switching proportions by sentiment."""
    
    sentiment_stats = defaultdict(lambda: {
        'en_proportions': [],
        'ta_proportions': [],
        'na_proportions': [],
        'count': 0
    })
    
    for item in data:
        sentiment = item['sentiment']
        sentiment_stats[sentiment]['en_proportions'].append(item['en_proportion'])
        sentiment_stats[sentiment]['ta_proportions'].append(item['ta_proportion'])
        sentiment_stats[sentiment]['na_proportions'].append(item['na_proportion'])
        sentiment_stats[sentiment]['count'] += 1
    
    print("="*70)
    print("CODE-SWITCHING ANALYSIS BY SENTIMENT")
    print("="*70)
    
    results = {}
    for sentiment in sorted(sentiment_stats.keys()):
        stats = sentiment_stats[sentiment]
        
        en_mean = np.mean(stats['en_proportions'])
        en_std = np.std(stats['en_proportions'])
        ta_mean = np.mean(stats['ta_proportions'])
        ta_std = np.std(stats['ta_proportions'])
        na_mean = np.mean(stats['na_proportions'])
        
        results[sentiment] = {
            'count': stats['count'],
            'en_mean': en_mean,
            'en_std': en_std,
            'ta_mean': ta_mean,
            'ta_std': ta_std,
            'na_mean': na_mean
        }
        
        print(f"\n{sentiment.upper()} (n={stats['count']:,})")
        print(f"  English proportion:  {en_mean:.2%} (±{en_std:.2%})")
        print(f"  Tamil proportion:    {ta_mean:.2%} (±{ta_std:.2%})")
        print(f"  Ambiguous proportion: {na_mean:.2%}")
    
    return results, sentiment_stats

def find_extreme_examples(data, n=5):
    """Find examples with highest/lowest English proportions per sentiment."""
    
    by_sentiment = defaultdict(list)
    for item in data:
        by_sentiment[item['sentiment']].append(item)
    
    print("\n" + "="*70)
    print("EXTREME EXAMPLES BY SENTIMENT")
    print("="*70)
    
    for sentiment in sorted(by_sentiment.keys()):
        items = by_sentiment[sentiment]
        
        # Sort by English proportion
        sorted_items = sorted(items, key=lambda x: x['en_proportion'], reverse=True)
        
        print(f"\n{sentiment.upper()} - Most English:")
        for item in sorted_items[:n]:
            print(f"  {item['en_proportion']:.0%} EN | {item['sentence'][:60]}...")
        
        print(f"\n{sentiment.upper()} - Least English (Most Tamil):")
        for item in sorted_items[-n:]:
            print(f"  {item['en_proportion']:.0%} EN | {item['sentence'][:60]}...")

def statistical_comparison(sentiment_stats):
    """Compare English proportions between sentiments."""
    
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    sentiments = list(sentiment_stats.keys())
    if len(sentiments) < 2:
        print("Need at least 2 sentiment categories for comparison")
        return
    
    from scipy import stats
    
    for i in range(len(sentiments)):
        for j in range(i+1, len(sentiments)):
            sent1, sent2 = sentiments[i], sentiments[j]
            
            en1 = sentiment_stats[sent1]['en_proportions']
            en2 = sentiment_stats[sent2]['en_proportions']
            
            # T-test
            t_stat, p_value = stats.ttest_ind(en1, en2)
            
            mean_diff = np.mean(en1) - np.mean(en2)
            
            print(f"\n{sent1.upper()} vs {sent2.upper()}:")
            print(f"  Mean English difference: {mean_diff:+.2%}")
            print(f"  T-statistic: {t_stat:.3f}")
            print(f"  P-value: {p_value:.4f}")
            if p_value < 0.001:
                print(f"  *** Highly significant (p < 0.001)")
            elif p_value < 0.01:
                print(f"  ** Very significant (p < 0.01)")
            elif p_value < 0.05:
                print(f"  * Significant (p < 0.05)")
            else:
                print(f"  Not significant (p >= 0.05)")

def create_visualizations(sentiment_stats, output_prefix):
    """Create visualization plots."""
    
    sentiments = sorted(sentiment_stats.keys())
    
    # Prepare data
    en_means = [np.mean(sentiment_stats[s]['en_proportions']) for s in sentiments]
    en_stds = [np.std(sentiment_stats[s]['en_proportions']) for s in sentiments]
    ta_means = [np.mean(sentiment_stats[s]['ta_proportions']) for s in sentiments]
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sentiments))
    width = 0.35
    
    ax.bar(x - width/2, en_means, width, yerr=en_stds, label='English', alpha=0.8)
    ax.bar(x + width/2, ta_means, width, label='Tamil', alpha=0.8)
    
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Proportion of tokens')
    ax.set_title('Language Proportion by Sentiment')
    ax.set_xticks(x)
    ax.set_xticklabels(sentiments)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_barplot.png', dpi=300)
    print(f"\nSaved bar plot to: {output_prefix}_barplot.png")
    
    # Box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_to_plot = [sentiment_stats[s]['en_proportions'] for s in sentiments]
    ax.boxplot(data_to_plot, labels=sentiments)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('English proportion')
    ax.set_title('Distribution of English Proportion by Sentiment')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_boxplot.png', dpi=300)
    print(f"Saved box plot to: {output_prefix}_boxplot.png")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_sentiment.py <predictions.json> [output_prefix]")
        print("\nExample:")
        print("  python analyze_sentiment.py langid_predictions.json analysis")
        sys.exit(1)
    
    predictions_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "sentiment_analysis"
    
    print("Loading predictions...")
    data = load_predictions(predictions_file)
    print(f"Loaded {len(data):,} sentences")
    
    # Main analysis
    results, sentiment_stats = analyze_by_sentiment(data)
    
    # Find extreme examples
    find_extreme_examples(data, n=5)
    
    # Statistical comparison
    try:
        statistical_comparison(sentiment_stats)
    except ImportError:
        print("\nNote: scipy not installed - skipping statistical tests")
        print("Install with: pip install scipy")
    
    # Create visualizations
    try:
        create_visualizations(sentiment_stats, output_prefix)
    except ImportError:
        print("\nNote: matplotlib not installed - skipping visualizations")
        print("Install with: pip install matplotlib")
    
    # Save summary
    summary = {
        'total_sentences': len(data),
        'by_sentiment': results
    }
    
    summary_file = f"{output_prefix}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to: {summary_file}")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()