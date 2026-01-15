import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import chi2_contingency, fisher_exact
from itertools import combinations
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class CausalPhraseMiner:
    """
    Discovers phrases and phrase combinations that have CAUSAL relationship with outcomes.
    
    Unlike correlation-based methods, this finds:
    1. Phrases whose PRESENCE changes the probability of SAR significantly
    2. Phrase combinations that interact (co-occurrence patterns)
    3. Phrases that appear in specific narrative positions (beginning, middle, end)
    
    This answers: "What phrases, when present, CAUSE investigators to file SARs?"
    
    No assumptions about content - purely data-driven discovery.
    """
    
    def __init__(self, ngram_range=(1, 5), min_freq=5, max_phrase_length=50):
        self.ngram_range = ngram_range
        self.min_freq = min_freq
        self.max_phrase_length = max_phrase_length
        self.phrase_causality_scores = None
        
    def preprocess_text(self, text):
        """Minimal preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_phrases(self, narratives, label):
        """
        Extract all n-grams with their document-level occurrence
        """
        clean_texts = [self.preprocess_text(t) for t in narratives]
        
        # Use CountVectorizer to extract n-grams
        vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_freq,
            max_df=0.85,
            binary=True  # We only care if phrase is present, not frequency
        )
        
        doc_phrase_matrix = vectorizer.fit_transform(clean_texts)
        phrases = vectorizer.get_feature_names_out()
        
        # Filter by length
        valid_phrases = [p for p in phrases if len(p) <= self.max_phrase_length]
        valid_indices = [i for i, p in enumerate(phrases) if len(p) <= self.max_phrase_length]
        
        doc_phrase_matrix = doc_phrase_matrix[:, valid_indices]
        
        return doc_phrase_matrix, valid_phrases, vectorizer
    
    def calculate_causal_effect(self, phrase_presence, outcomes):
        """
        Calculate causal effect of phrase presence on outcome
        Using Average Treatment Effect (ATE) estimation
        
        Treatment: Phrase is present (1) or absent (0)
        Outcome: SAR (1) or non-issue (0)
        """
        # Split into treated (phrase present) and control (phrase absent)
        treated_mask = phrase_presence == 1
        control_mask = phrase_presence == 0
        
        if treated_mask.sum() < self.min_freq or control_mask.sum() < self.min_freq:
            return None
        
        # Calculate outcome rates
        treated_sar_rate = outcomes[treated_mask].mean()
        control_sar_rate = outcomes[control_mask].mean()
        
        # Average Treatment Effect
        ate = treated_sar_rate - control_sar_rate
        
        # Calculate statistical significance
        contingency = np.array([
            [outcomes[treated_mask].sum(), (~outcomes[treated_mask]).sum()],
            [outcomes[control_mask].sum(), (~outcomes[control_mask]).sum()]
        ])
        
        if contingency.min() >= 5:
            chi2, p_value, _, _ = chi2_contingency(contingency)
        else:
            _, p_value = fisher_exact(contingency)
        
        # Calculate relative risk (risk ratio)
        rr = treated_sar_rate / (control_sar_rate + 1e-10)
        
        return {
            'ate': ate,
            'treated_sar_rate': treated_sar_rate,
            'control_sar_rate': control_sar_rate,
            'relative_risk': rr,
            'p_value': p_value,
            'treated_count': treated_mask.sum(),
            'control_count': control_mask.sum()
        }
    
    def mine_causal_phrases(self, non_issue_df, sar_df, narrative_col='NARRATIVE'):
        """
        Mine phrases with strong causal effects on SAR disposition
        """
        # Combine datasets
        non_issue_clean = non_issue_df[narrative_col].values
        sar_clean = sar_df[narrative_col].values
        
        all_narratives = np.concatenate([non_issue_clean, sar_clean])
        outcomes = np.concatenate([
            np.zeros(len(non_issue_clean)),
            np.ones(len(sar_clean))
        ]).astype(bool)
        
        # Extract phrases
        print("Extracting phrases from all narratives...")
        doc_phrase_matrix, phrases, vectorizer = self.extract_phrases(
            all_narratives, 
            outcomes
        )
        
        print(f"Found {len(phrases)} candidate phrases")
        
        # Calculate causal effects
        print("Calculating causal effects for each phrase...")
        causal_effects = []
        
        for i, phrase in enumerate(phrases):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(phrases)} phrases...")
            
            phrase_presence = np.array(doc_phrase_matrix[:, i].toarray()).flatten()
            
            effect = self.calculate_causal_effect(phrase_presence, outcomes)
            
            if effect is not None:
                causal_effects.append({
                    'phrase': phrase,
                    **effect
                })
        
        self.phrase_causality_scores = pd.DataFrame(causal_effects)
        
        # Filter for statistical significance
        self.phrase_causality_scores = self.phrase_causality_scores[
            self.phrase_causality_scores['p_value'] < 0.01
        ]
        
        return self.phrase_causality_scores
    
    def get_causal_red_flags(self, top_n=100, min_ate=0.1):
        """
        Get phrases that causally increase SAR probability
        """
        red_flags = self.phrase_causality_scores[
            self.phrase_causality_scores['ate'] > min_ate
        ].copy()
        
        # Sort by causal effect size
        red_flags = red_flags.sort_values('ate', ascending=False).head(top_n)
        
        return red_flags
    
    def get_causal_green_flags(self, top_n=100, min_ate=-0.1):
        """
        Get phrases that causally decrease SAR probability
        """
        green_flags = self.phrase_causality_scores[
            self.phrase_causality_scores['ate'] < min_ate
        ].copy()
        
        # Sort by causal effect size (most negative)
        green_flags = green_flags.sort_values('ate', ascending=True).head(top_n)
        
        return green_flags
    
    def discover_phrase_interactions(self, non_issue_df, sar_df, 
                                    top_phrases, narrative_col='NARRATIVE',
                                    max_combinations=50):
        """
        Find phrases that interact: their combined effect is stronger than individual effects
        
        This finds patterns like: "phrase A alone is weak, phrase B alone is weak,
        but A+B together strongly indicates SAR"
        """
        # Get phrase presence for top phrases
        all_narratives = np.concatenate([
            non_issue_df[narrative_col].values,
            sar_df[narrative_col].values
        ])
        
        outcomes = np.concatenate([
            np.zeros(len(non_issue_df)),
            np.ones(len(sar_df))
        ]).astype(bool)
        
        # Create phrase presence matrix for top phrases
        phrase_presence = {}
        for phrase in top_phrases:
            presence = np.array([
                1 if phrase in self.preprocess_text(text) else 0 
                for text in all_narratives
            ])
            phrase_presence[phrase] = presence
        
        # Find interactions
        print("Discovering phrase interactions...")
        interactions = []
        
        # Test pairs of phrases
        phrase_list = list(top_phrases)
        for i, phrase1 in enumerate(phrase_list):
            for phrase2 in phrase_list[i+1:]:
                # Both phrases present
                both_present = (phrase_presence[phrase1] == 1) & (phrase_presence[phrase2] == 1)
                
                if both_present.sum() < self.min_freq:
                    continue
                
                # Calculate individual effects
                effect1 = self.calculate_causal_effect(phrase_presence[phrase1], outcomes)
                effect2 = self.calculate_causal_effect(phrase_presence[phrase2], outcomes)
                
                if effect1 is None or effect2 is None:
                    continue
                
                # Calculate combined effect
                combined_sar_rate = outcomes[both_present].mean()
                expected_combined = (effect1['treated_sar_rate'] + effect2['treated_sar_rate']) / 2
                
                # Interaction effect (synergy)
                interaction_effect = combined_sar_rate - expected_combined
                
                if abs(interaction_effect) > 0.1:  # Meaningful interaction
                    interactions.append({
                        'phrase1': phrase1,
                        'phrase2': phrase2,
                        'combined_sar_rate': combined_sar_rate,
                        'expected_rate': expected_combined,
                        'interaction_effect': interaction_effect,
                        'synergy_type': 'positive' if interaction_effect > 0 else 'negative',
                        'support': both_present.sum()
                    })
                
                if len(interactions) >= max_combinations:
                    break
            
            if len(interactions) >= max_combinations:
                break
        
        return pd.DataFrame(interactions).sort_values('interaction_effect', 
                                                      key=abs, 
                                                      ascending=False)
    
    def analyze_phrase_positions(self, narratives, phrases):
        """
        Analyze where phrases appear in narratives (beginning, middle, end)
        Some phrases are more meaningful when they appear at specific positions
        """
        position_analysis = []
        
        for phrase in phrases:
            positions = {'beginning': 0, 'middle': 0, 'end': 0}
            total_occurrences = 0
            
            for narrative in narratives:
                text = self.preprocess_text(narrative)
                if phrase not in text:
                    continue
                
                total_occurrences += 1
                
                # Find position of first occurrence
                idx = text.find(phrase)
                relative_position = idx / len(text) if len(text) > 0 else 0
                
                if relative_position < 0.33:
                    positions['beginning'] += 1
                elif relative_position < 0.67:
                    positions['middle'] += 1
                else:
                    positions['end'] += 1
            
            if total_occurrences >= self.min_freq:
                position_analysis.append({
                    'phrase': phrase,
                    'beginning_pct': positions['beginning'] / total_occurrences,
                    'middle_pct': positions['middle'] / total_occurrences,
                    'end_pct': positions['end'] / total_occurrences,
                    'total_occurrences': total_occurrences,
                    'dominant_position': max(positions, key=positions.get)
                })
        
        return pd.DataFrame(position_analysis)


def run_causal_phrase_discovery(non_issue_df, sar_df):
    """
    Run comprehensive causal phrase mining
    """
    print("="*80)
    print("CAUSAL PHRASE MINING - FULLY AUTOMATIC DISCOVERY")
    print("="*80)
    print("\nDiscovering phrases that CAUSE SAR dispositions")
    print("Using causal inference methods (Average Treatment Effect)\n")
    
    miner = CausalPhraseMiner(
        ngram_range=(2, 5),  # 2-5 word phrases
        min_freq=5,
        max_phrase_length=60
    )
    
    # Mine causal phrases
    print("1. Mining phrases with causal effects...")
    causal_scores = miner.mine_causal_phrases(non_issue_df, sar_df)
    
    print(f"   Found {len(causal_scores)} phrases with statistically significant causal effects")
    
    # Get causal red flags
    print("\n2. Extracting causal red flags...")
    red_flags = miner.get_causal_red_flags(top_n=50, min_ate=0.1)
    
    print("\n" + "="*80)
    print("TOP CAUSAL RED FLAGS")
    print("="*80)
    print("\nPhrases that CAUSE increased SAR probability:\n")
    print(f"{'Phrase':<50} {'ATE':<8} {'RR':<6} {'Treated%':<10} {'Control%':<10}")
    print("-" * 90)
    
    for _, row in red_flags.head(30).iterrows():
        print(f"{row['phrase']:<50} "
              f"{row['ate']:>7.1%} "
              f"{row['relative_risk']:>5.1f}x "
              f"{row['treated_sar_rate']:>9.1%} "
              f"{row['control_sar_rate']:>9.1%}")
    
    # Get causal green flags
    print("\n3. Extracting causal green flags...")
    green_flags = miner.get_causal_green_flags(top_n=50, min_ate=-0.1)
    
    print("\n" + "="*80)
    print("TOP CAUSAL GREEN FLAGS")
    print("="*80)
    print("\nPhrases that CAUSE decreased SAR probability:\n")
    print(f"{'Phrase':<50} {'ATE':<8} {'RR':<6} {'Treated%':<10} {'Control%':<10}")
    print("-" * 90)
    
    for _, row in green_flags.head(30).iterrows():
        print(f"{row['phrase']:<50} "
              f"{row['ate']:>7.1%} "
              f"{row['relative_risk']:>5.1f}x "
              f"{row['treated_sar_rate']:>9.1%} "
              f"{row['control_sar_rate']:>9.1%}")
    
    # Discover interactions
    print("\n4. Discovering phrase interactions...")
    top_red_phrases = red_flags.head(20)['phrase'].values
    interactions = miner.discover_phrase_interactions(
        non_issue_df, 
        sar_df,
        top_red_phrases,
        max_combinations=30
    )
    
    if len(interactions) > 0:
        print("\n" + "="*80)
        print("PHRASE INTERACTIONS (Synergistic Effects)")
        print("="*80)
        print("\nPhrase pairs with stronger combined effect:\n")
        
        for _, row in interactions.head(15).iterrows():
            print(f"\n{row['phrase1']}")
            print(f"  + {row['phrase2']}")
            print(f"  → Combined SAR rate: {row['combined_sar_rate']:.1%}")
            print(f"  → Expected rate: {row['expected_rate']:.1%}")
            print(f"  → Synergy: {row['interaction_effect']:+.1%} ({row['synergy_type']})")
            print(f"  → Support: {row['support']} cases")
    
    # Analyze phrase positions
    print("\n5. Analyzing phrase positions in narratives...")
    
    print("\n  Red flag positions (where they appear):")
    red_positions = miner.analyze_phrase_positions(
        sar_df['NARRATIVE'].values,
        red_flags.head(20)['phrase'].values
    )
    
    # Show phrases with strong position preferences
    if len(red_positions) > 0:
        beginning_phrases = red_positions[red_positions['beginning_pct'] > 0.6].head(5)
        if len(beginning_phrases) > 0:
            print("\n    Phrases appearing at BEGINNING (red flags mentioned early):")
            for _, row in beginning_phrases.iterrows():
                print(f"      - {row['phrase']} ({row['beginning_pct']:.0%} at beginning)")
        
        end_phrases = red_positions[red_positions['end_pct'] > 0.6].head(5)
        if len(end_phrases) > 0:
            print("\n    Phrases appearing at END (conclusions):")
            for _, row in end_phrases.iterrows():
                print(f"      - {row['phrase']} ({row['end_pct']:.0%} at end)")
    
    print("\n  Green flag positions:")
    green_positions = miner.analyze_phrase_positions(
        non_issue_df['NARRATIVE'].values,
        green_flags.head(20)['phrase'].values
    )
    
    if len(green_positions) > 0:
        beginning_phrases = green_positions[green_positions['beginning_pct'] > 0.6].head(5)
        if len(beginning_phrases) > 0:
            print("\n    Phrases appearing at BEGINNING:")
            for _, row in beginning_phrases.iterrows():
                print(f"      - {row['phrase']} ({row['beginning_pct']:.0%} at beginning)")
    
    # Save results
    red_flags.to_csv('causal_red_flags.csv', index=False)
    green_flags.to_csv('causal_green_flags.csv', index=False)
    
    if len(interactions) > 0:
        interactions.to_csv('phrase_interactions.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ Results saved:")
    print("  - causal_red_flags.csv")
    print("  - causal_green_flags.csv")
    if len(interactions) > 0:
        print("  - phrase_interactions.csv")
    print(f"\n✓ Discovered {len(red_flags)} causal red flags")
    print(f"✓ Discovered {len(green_flags)} causal green flags")
    if len(interactions) > 0:
        print(f"✓ Found {len(interactions)} significant phrase interactions")
    
    return {
        'red_flags': red_flags,
        'green_flags': green_flags,
        'interactions': interactions,
        'all_causal_scores': causal_scores,
        'miner': miner
    }


# Example usage:
"""
# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run causal phrase discovery
results = run_causal_phrase_discovery(non_issue_df, sar_df)

# Analyze specific findings
print("\n\nPhrases with highest causal effect:")
print(results['red_flags'].head(10)[['phrase', 'ate', 'relative_risk']])

print("\n\nStrongest phrase interactions:")
if len(results['interactions']) > 0:
    print(results['interactions'].head(5)[['phrase1', 'phrase2', 'interaction_effect']])

# Find phrases with specific patterns
high_rr = results['red_flags'][results['red_flags']['relative_risk'] > 10]
print(f"\n\nPhrases with 10x+ relative risk:")
print(high_rr[['phrase', 'relative_risk', 'treated_sar_rate']])
"""