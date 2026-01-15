import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency, fisher_exact
import re
from itertools import combinations
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class ContrastivePhraseMininer:
    """
    Mine phrases that appear in specific contexts within one class but not the other.
    This finds nuanced patterns like "however, upon review" (dismissive phrases) vs
    "unable to verify" (suspicious phrases) that simple TF-IDF misses.
    """
    
    def __init__(self, ngram_range=(2, 5), min_freq=5, context_window=10):
        self.ngram_range = ngram_range
        self.min_freq = min_freq
        self.context_window = context_window
        
    def preprocess_text(self, text):
        """Minimal preprocessing to preserve phrase structure"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Keep some punctuation for phrase boundaries
        text = re.sub(r'[^a-z0-9\s,;.!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_contextual_ngrams(self, narratives):
        """
        Extract n-grams with their surrounding context patterns
        Returns phrases with their typical preceding and following words
        """
        contextual_phrases = defaultdict(lambda: {'before': Counter(), 'after': Counter(), 'count': 0})
        
        for narrative in narratives:
            text = self.preprocess_text(narrative)
            words = text.split()
            
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    
                    # Capture context before
                    if i > 0:
                        context_before = ' '.join(words[max(0, i-self.context_window):i])
                        contextual_phrases[phrase]['before'][context_before] += 1
                    
                    # Capture context after
                    if i + n < len(words):
                        context_after = ' '.join(words[i+n:min(len(words), i+n+self.context_window)])
                        contextual_phrases[phrase]['after'][context_after] += 1
                    
                    contextual_phrases[phrase]['count'] += 1
        
        return contextual_phrases
    
    def calculate_phrase_distinctiveness(self, phrases_class1, phrases_class2, 
                                        total_docs_1, total_docs_2):
        """
        Calculate statistical distinctiveness of phrases using log-likelihood ratio
        and contextual consistency
        """
        all_phrases = set(phrases_class1.keys()) | set(phrases_class2.keys())
        
        distinctive_phrases = []
        
        for phrase in all_phrases:
            count_1 = phrases_class1.get(phrase, {'count': 0})['count']
            count_2 = phrases_class2.get(phrase, {'count': 0})['count']
            
            # Skip rare phrases
            if count_1 + count_2 < self.min_freq:
                continue
            
            # Calculate log-likelihood ratio for distinctiveness
            # This is more robust than simple ratios for sparse data
            contingency = np.array([
                [count_1, total_docs_1 - count_1],
                [count_2, total_docs_2 - count_2]
            ])
            
            if contingency.min() >= 5:  # Chi-square valid
                chi2, p_value, _, _ = chi2_contingency(contingency)
                log_likelihood = chi2
            else:  # Use Fisher's exact test for small counts
                _, p_value = fisher_exact(contingency[:, :2])
                log_likelihood = -np.log(p_value + 1e-10)
            
            # Calculate effect size (phi coefficient)
            phi = (count_1 / total_docs_1) - (count_2 / total_docs_2)
            
            # Get contextual consistency score
            context_score = self._calculate_context_consistency(
                phrases_class1.get(phrase, {}),
                phrases_class2.get(phrase, {})
            )
            
            distinctive_phrases.append({
                'phrase': phrase,
                'count_class1': count_1,
                'count_class2': count_2,
                'log_likelihood': log_likelihood,
                'p_value': p_value,
                'effect_size': phi,
                'context_consistency': context_score,
                'distinctiveness_score': log_likelihood * abs(phi) * context_score
            })
        
        return pd.DataFrame(distinctive_phrases)
    
    def _calculate_context_consistency(self, phrase_data_1, phrase_data_2):
        """
        Measure how consistently a phrase appears in specific contexts
        Higher score = phrase has distinct contextual usage patterns
        """
        if not phrase_data_1 or not phrase_data_2:
            return 1.0
        
        # Compare context distributions
        contexts_1 = set(phrase_data_1.get('before', {}).keys()) | set(phrase_data_1.get('after', {}).keys())
        contexts_2 = set(phrase_data_2.get('before', {}).keys()) | set(phrase_data_2.get('after', {}).keys())
        
        if not contexts_1 or not contexts_2:
            return 1.0
        
        # Jaccard distance of contexts
        overlap = len(contexts_1 & contexts_2)
        union = len(contexts_1 | contexts_2)
        
        consistency = 1 - (overlap / union if union > 0 else 0)
        return consistency
    
    def extract_hedge_phrases(self, narratives):
        """
        Specifically extract hedging language that indicates uncertainty or confidence
        Key for understanding investigator reasoning
        """
        hedge_patterns = [
            r'\b(appears? to be|seems? to be|likely|possibly|probably|may be|might be|could be)\b',
            r'\b(unclear|uncertain|unknown|unverified|unable to (confirm|verify|determine))\b',
            r'\b(however|although|despite|nevertheless|nonetheless)\b',
            r'\b(consistent with|inconsistent with|in line with|typical of|unusual for)\b',
            r'\b(legitimate|suspicious|questionable|concerning|normal|expected|unexpected)\b',
            r'\b(no evidence of|no indication of|evidence of|indication of)\b',
            r'\b(upon review|further review|investigation revealed|analysis shows)\b'
        ]
        
        hedge_counts = defaultdict(list)
        
        for narrative in narratives:
            text = self.preprocess_text(narrative)
            
            for pattern in hedge_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Get surrounding context (20 words before and after)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    
                    hedge_counts[match.group()].append(context)
        
        return hedge_counts
    
    def mine_conditional_patterns(self, narratives):
        """
        Find if-then reasoning patterns that investigators use
        e.g., "if X then concluded Y" patterns
        """
        conditional_keywords = ['if', 'when', 'after', 'upon', 'following', 'given', 'since', 'because']
        conclusive_keywords = ['therefore', 'thus', 'concluded', 'determined', 'found', 'closed', 'filed']
        
        patterns = []
        
        for narrative in narratives:
            text = self.preprocess_text(narrative)
            sentences = re.split(r'[.!?]+', text)
            
            for sent in sentences:
                words = sent.split()
                
                # Look for conditional + conclusive structure
                has_conditional = any(kw in words for kw in conditional_keywords)
                has_conclusive = any(kw in words for kw in conclusive_keywords)
                
                if has_conditional and has_conclusive:
                    # Extract the pattern
                    patterns.append({
                        'sentence': sent.strip(),
                        'length': len(words)
                    })
        
        return patterns


def run_contrastive_analysis(non_issue_df, sar_df):
    """
    Run comprehensive contrastive phrase mining analysis
    """
    print("="*80)
    print("CONTRASTIVE PHRASE MINING FOR AML ALERTS")
    print("="*80)
    
    miner = ContrastivePhraseMininer(
        ngram_range=(2, 5),
        min_freq=5,
        context_window=10
    )
    
    # Extract contextual n-grams
    print("\n1. Extracting contextual phrases from Non-Issue narratives...")
    ni_phrases = miner.extract_contextual_ngrams(non_issue_df['NARRATIVE'].values)
    
    print("2. Extracting contextual phrases from SAR narratives...")
    sar_phrases = miner.extract_contextual_ngrams(sar_df['NARRATIVE'].values)
    
    # Calculate distinctiveness
    print("3. Calculating phrase distinctiveness...")
    distinctive_df = miner.calculate_phrase_distinctiveness(
        ni_phrases, sar_phrases,
        len(non_issue_df), len(sar_df)
    )
    
    # Filter for significance
    significant = distinctive_df[distinctive_df['p_value'] < 0.01].copy()
    
    # Separate green flags and red flags
    green_flags = significant[significant['effect_size'] > 0].sort_values(
        'distinctiveness_score', ascending=False
    ).head(50)
    
    red_flags = significant[significant['effect_size'] < 0].sort_values(
        'distinctiveness_score', ascending=False
    ).head(50)
    
    print("\n" + "="*80)
    print("TOP DISTINCTIVE GREEN FLAG PHRASES (False Positive Indicators)")
    print("="*80)
    print(green_flags[['phrase', 'count_class1', 'count_class2', 'effect_size', 'context_consistency']].head(20))
    
    print("\n" + "="*80)
    print("TOP DISTINCTIVE RED FLAG PHRASES (True Positive Indicators)")
    print("="*80)
    print(red_flags[['phrase', 'count_class1', 'count_class2', 'effect_size', 'context_consistency']].head(20))
    
    # Extract hedge phrases
    print("\n4. Analyzing hedging language...")
    ni_hedges = miner.extract_hedge_phrases(non_issue_df['NARRATIVE'].values)
    sar_hedges = miner.extract_hedge_phrases(sar_df['NARRATIVE'].values)
    
    print("\n" + "="*80)
    print("HEDGING LANGUAGE COMPARISON")
    print("="*80)
    
    hedge_comparison = []
    all_hedges = set(ni_hedges.keys()) | set(sar_hedges.keys())
    
    for hedge in sorted(all_hedges):
        ni_count = len(ni_hedges.get(hedge, []))
        sar_count = len(sar_hedges.get(hedge, []))
        
        hedge_comparison.append({
            'hedge_phrase': hedge,
            'non_issue_freq': ni_count,
            'sar_freq': sar_count,
            'ratio': (ni_count / len(non_issue_df)) / (sar_count / len(sar_df) + 1e-6)
        })
    
    hedge_df = pd.DataFrame(hedge_comparison).sort_values('ratio', ascending=False)
    print("\nHedges more common in Non-Issue (confidence/dismissive):")
    print(hedge_df.head(10))
    print("\nHedges more common in SAR (uncertainty/suspicion):")
    print(hedge_df.tail(10))
    
    # Extract conditional reasoning patterns
    print("\n5. Mining conditional reasoning patterns...")
    ni_conditionals = miner.mine_conditional_patterns(non_issue_df['NARRATIVE'].values)
    sar_conditionals = miner.mine_conditional_patterns(sar_df['NARRATIVE'].values)
    
    print(f"\nNon-Issue conditional patterns found: {len(ni_conditionals)}")
    print(f"SAR conditional patterns found: {len(sar_conditionals)}")
    
    if ni_conditionals:
        print("\nExample Non-Issue reasoning patterns:")
        for pattern in ni_conditionals[:3]:
            print(f"  - {pattern['sentence'][:150]}...")
    
    if sar_conditionals:
        print("\nExample SAR reasoning patterns:")
        for pattern in sar_conditionals[:3]:
            print(f"  - {pattern['sentence'][:150]}...")
    
    # Return all results
    return {
        'green_flags': green_flags,
        'red_flags': red_flags,
        'hedge_analysis': hedge_df,
        'ni_hedges': ni_hedges,
        'sar_hedges': sar_hedges,
        'ni_conditionals': ni_conditionals,
        'sar_conditionals': sar_conditionals
    }


# Example usage:
"""
# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run contrastive analysis
results = run_contrastive_analysis(non_issue_df, sar_df)

# Save detailed results
results['green_flags'].to_csv('contrastive_green_flags.csv', index=False)
results['red_flags'].to_csv('contrastive_red_flags.csv', index=False)
results['hedge_analysis'].to_csv('hedge_language_analysis.csv', index=False)

# Get examples of specific hedge usage
specific_hedge = 'unable to verify'
if specific_hedge in results['sar_hedges']:
    print(f"\nExamples of '{specific_hedge}' in SAR narratives:")
    for example in results['sar_hedges'][specific_hedge][:5]:
        print(f"  - {example}")
"""