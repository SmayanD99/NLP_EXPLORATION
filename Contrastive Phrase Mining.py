import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import fisher_exact, mannwhitneyu
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class ImbalancedContrastiveMiner:
    """
    Contrastive phrase mining adapted for 14,000 vs 680 imbalance
    
    Key changes:
    1. Absolute count thresholds (not percentages): min 3 in SAR, min 10 in non-issue
    2. Fisher's exact test for statistical significance
    3. Balanced sampling for phrase comparison
    4. Minority class (SAR) gets priority in analysis
    """
    
    def __init__(self, ngram_range=(2, 5), min_freq_sar=3, min_freq_ni=10):
        self.ngram_range = ngram_range
        self.min_freq_sar = min_freq_sar  # 0.44% of 680
        self.min_freq_ni = min_freq_ni    # 0.07% of 14,000
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s,;.!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_contextual_ngrams(self, narratives, min_freq):
        """Extract n-grams with class-specific minimum frequency"""
        phrase_counts = Counter()
        
        for narrative in narratives:
            text = self.preprocess_text(narrative)
            words = text.split()
            
            # Extract n-grams
            seen_in_doc = set()
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    if len(phrase) <= 60 and phrase not in seen_in_doc:
                        phrase_counts[phrase] += 1
                        seen_in_doc.add(phrase)
        
        # Filter by minimum frequency
        filtered = {p: c for p, c in phrase_counts.items() if c >= min_freq}
        return filtered
    
    def calculate_phrase_distinctiveness(self, ni_phrases, sar_phrases, 
                                        n_ni, n_sar):
        """
        Calculate distinctiveness using Fisher's exact test and odds ratios
        Handles imbalance properly
        """
        all_phrases = set(ni_phrases.keys()) | set(sar_phrases.keys())
        
        results = []
        
        for phrase in all_phrases:
            count_ni = ni_phrases.get(phrase, 0)
            count_sar = sar_phrases.get(phrase, 0)
            
            # Skip if too rare in both
            if count_ni + count_sar < 5:
                continue
            
            # Contingency table
            contingency = [
                [count_sar, n_sar - count_sar],
                [count_ni, n_ni - count_ni]
            ]
            
            try:
                odds_ratio, p_value = fisher_exact(contingency)
            except:
                continue
            
            if p_value >= 0.01:  # Only keep statistically significant
                continue
            
            # Calculate frequencies
            sar_freq = count_sar / n_sar
            ni_freq = count_ni / n_ni
            
            results.append({
                'phrase': phrase,
                'count_sar': count_sar,
                'count_ni': count_ni,
                'sar_freq_pct': sar_freq * 100,
                'ni_freq_pct': ni_freq * 100,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'effect_direction': 'SAR' if odds_ratio > 1 else 'Non-Issue'
            })
        
        df = pd.DataFrame(results)
        
        # Separate by direction
        if len(df) > 0:
            red_flags = df[df['effect_direction'] == 'SAR'].copy()
            red_flags['score'] = red_flags['odds_ratio'] * np.log1p(red_flags['count_sar'])
            red_flags = red_flags.sort_values('score', ascending=False)
            
            green_flags = df[df['effect_direction'] == 'Non-Issue'].copy()
            green_flags['score'] = (1/green_flags['odds_ratio']) * np.log1p(green_flags['count_ni'])
            green_flags = green_flags.sort_values('score', ascending=False)
            
            return red_flags, green_flags
        
        return pd.DataFrame(), pd.DataFrame()
    
    def extract_hedge_phrases(self, narratives, min_freq=3):
        """Extract hedging language with appropriate thresholds"""
        hedge_patterns = [
            r'\b(appears? to be|seems? to be|likely|possibly|probably|may be|might be|could be)\b',
            r'\b(unclear|uncertain|unknown|unverified|unable to (confirm|verify|determine))\b',
            r'\b(however|although|despite|nevertheless|nonetheless)\b',
            r'\b(consistent with|inconsistent with|in line with|typical of|unusual for)\b',
            r'\b(legitimate|suspicious|questionable|concerning|normal|expected|unexpected)\b',
            r'\b(no evidence of|no indication of|evidence of|indication of)\b',
            r'\b(upon review|further review|investigation revealed|analysis shows)\b'
        ]
        
        hedge_counts = Counter()
        
        for narrative in narratives:
            text = self.preprocess_text(narrative)
            
            for pattern in hedge_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    hedge_counts[match] += 1
        
        # Filter by minimum frequency
        return {h: c for h, c in hedge_counts.items() if c >= min_freq}


def run_imbalanced_contrastive_analysis(non_issue_df, sar_df):
    """
    Run contrastive analysis optimized for class imbalance
    """
    print("="*80)
    print("CONTRASTIVE PHRASE MINING - IMBALANCE OPTIMIZED")
    print("="*80)
    print(f"Dataset: {len(non_issue_df):,} non-issue vs {len(sar_df):,} SAR")
    print(f"Imbalance ratio: {len(non_issue_df)/len(sar_df):.1f}:1\n")
    
    miner = ImbalancedContrastiveMiner(
        ngram_range=(2, 5),
        min_freq_sar=3,   # 0.44% of 680 SARs
        min_freq_ni=10    # 0.07% of 14,000 non-issue
    )
    
    # Extract phrases with class-specific thresholds
    print("1. Extracting phrases from non-issue (min_freq=10)...")
    ni_phrases = miner.extract_contextual_ngrams(
        non_issue_df['NARRATIVE'].values, 
        min_freq=miner.min_freq_ni
    )
    print(f"   Found {len(ni_phrases):,} phrases")
    
    print("2. Extracting phrases from SAR (min_freq=3)...")
    sar_phrases = miner.extract_contextual_ngrams(
        sar_df['NARRATIVE'].values,
        min_freq=miner.min_freq_sar
    )
    print(f"   Found {len(sar_phrases):,} phrases")
    
    # Calculate distinctiveness
    print("3. Calculating phrase distinctiveness (Fisher's exact test)...")
    red_flags, green_flags = miner.calculate_phrase_distinctiveness(
        ni_phrases, sar_phrases,
        len(non_issue_df), len(sar_df)
    )
    
    print("\n" + "="*80)
    print("RED FLAG PHRASES (SAR Indicators)")
    print("="*80)
    print(f"\nFound {len(red_flags)} statistically significant SAR phrases:\n")
    print(f"{'Phrase':<50} {'SAR%':<8} {'NI%':<8} {'OR':<8} {'Count':<8}")
    print("-" * 90)
    
    for _, row in red_flags.head(30).iterrows():
        print(f"{row['phrase']:<50} "
              f"{row['sar_freq_pct']:>6.1f}% "
              f"{row['ni_freq_pct']:>6.1f}% "
              f"{row['odds_ratio']:>7.1f} "
              f"{row['count_sar']:>7}")
    
    print("\n" + "="*80)
    print("GREEN FLAG PHRASES (Non-Issue Indicators)")
    print("="*80)
    print(f"\nFound {len(green_flags)} statistically significant non-issue phrases:\n")
    print(f"{'Phrase':<50} {'NI%':<8} {'SAR%':<8} {'OR':<8} {'Count':<8}")
    print("-" * 90)
    
    for _, row in green_flags.head(30).iterrows():
        print(f"{row['phrase']:<50} "
              f"{row['ni_freq_pct']:>6.1f}% "
              f"{row['sar_freq_pct']:>6.1f}% "
              f"{1/row['odds_ratio']:>7.1f} "
              f"{row['count_ni']:>7}")
    
    # Hedge analysis with different thresholds
    print("\n4. Analyzing hedging language...")
    ni_hedges = miner.extract_hedge_phrases(non_issue_df['NARRATIVE'].values, min_freq=10)
    sar_hedges = miner.extract_hedge_phrases(sar_df['NARRATIVE'].values, min_freq=3)
    
    print("\n" + "="*80)
    print("HEDGING LANGUAGE COMPARISON")
    print("="*80)
    
    all_hedges = set(ni_hedges.keys()) | set(sar_hedges.keys())
    hedge_comparison = []
    
    for hedge in all_hedges:
        ni_count = ni_hedges.get(hedge, 0)
        sar_count = sar_hedges.get(hedge, 0)
        
        ni_freq = ni_count / len(non_issue_df)
        sar_freq = sar_count / len(sar_df)
        
        # Calculate odds ratio
        contingency = [
            [sar_count, len(sar_df) - sar_count],
            [ni_count, len(non_issue_df) - ni_count]
        ]
        
        try:
            odds_ratio, p_value = fisher_exact(contingency)
            if p_value < 0.01:
                hedge_comparison.append({
                    'hedge': hedge,
                    'sar_count': sar_count,
                    'ni_count': ni_count,
                    'sar_freq_pct': sar_freq * 100,
                    'ni_freq_pct': ni_freq * 100,
                    'odds_ratio': odds_ratio,
                    'p_value': p_value
                })
        except:
            pass
    
    if hedge_comparison:
        hedge_df = pd.DataFrame(hedge_comparison).sort_values('odds_ratio', ascending=False)
        
        print("\nHedges more common in SAR:")
        sar_hedges = hedge_df[hedge_df['odds_ratio'] > 1].head(10)
        for _, row in sar_hedges.iterrows():
            print(f"  {row['hedge']:<40} OR={row['odds_ratio']:>5.1f}  "
                  f"SAR:{row['sar_freq_pct']:>5.1f}%  NI:{row['ni_freq_pct']:>5.1f}%")
        
        print("\nHedges more common in Non-Issue:")
        ni_hedges = hedge_df[hedge_df['odds_ratio'] < 1].tail(10)
        for _, row in ni_hedges.iterrows():
            # Safe division for display
            inv_or = 1 / row['odds_ratio'] if row['odds_ratio'] > 0 else float('inf')
            print(f"  {row['hedge']:<40} OR={inv_or:>5.1f}  "
                  f"NI:{row['ni_freq_pct']:>5.1f}%  SAR:{row['sar_freq_pct']:>5.1f}%")
        
        hedge_df.to_csv('hedge_analysis_imbalanced.csv', index=False)
    
    # Save results
    red_flags.to_csv('contrastive_red_flags.csv', index=False)
    green_flags.to_csv('contrastive_green_flags.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ Results saved")
    print(f"✓ {len(red_flags)} red flag phrases")
    print(f"✓ {len(green_flags)} green flag phrases")
    
    return {
        'red_flags': red_flags,
        'green_flags': green_flags,
        'hedge_comparison': hedge_df if hedge_comparison else pd.DataFrame()
    }


# Example usage:
"""
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

results = run_imbalanced_contrastive_analysis(non_issue_df, sar_df)

# Look at top red flags
print("\nTop 10 SAR indicator phrases:")
print(results['red_flags'].head(10)[['phrase', 'odds_ratio', 'count_sar']])
"""
