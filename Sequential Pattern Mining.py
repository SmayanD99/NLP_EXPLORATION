import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import re
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

class SequentialPatternMiner:
    """
    Mine sequential patterns in investigator narratives to understand:
    1. The ORDER in which investigators mention facts/conclusions
    2. Investigation workflows (what they check first, second, etc.)
    3. Reasoning chains that lead to disposition decisions
    
    This reveals the PROCESS of investigation, not just keywords.
    """
    
    def __init__(self, min_support=0.05, max_gap=3):
        self.min_support = min_support
        self.max_gap = max_gap
        self.entity_patterns = self._compile_entity_patterns()
        
    def _compile_entity_patterns(self):
        """
        Define semantic categories for pattern mining
        """
        return {
            'verification': r'\b(verified|confirmed|documented|evidence|proof|records?|statement)\b',
            'negation': r'\b(no|not|unable|lack|without|absent|missing)\b',
            'transaction_type': r'\b(deposit|withdrawal|transfer|payment|wire|check|cash|ach)\b',
            'amount_descriptor': r'\b(large|small|round|structured|unusual|consistent|frequent)\b',
            'source': r'\b(employer|salary|payroll|business|customer|vendor|loan|inheritance|sale)\b',
            'relationship': r'\b(spouse|family|related|employee|customer|third[- ]party|known)\b',
            'temporal': r'\b(daily|weekly|monthly|regular|one[- ]time|recurring|pattern)\b',
            'risk_indicator': r'\b(suspicious|concerning|unusual|inconsistent|questionable|legitimate|normal)\b',
            'action': r'\b(reviewed|analyzed|examined|investigated|contacted|requested|obtained)\b',
            'conclusion': r'\b(determined|concluded|found|assessed|closed|filed|reported)\b',
            'reason': r'\b(because|due to|given|since|as|therefore|thus|hence)\b'
        }
    
    def extract_semantic_sequence(self, narrative):
        """
        Convert narrative into sequence of semantic tokens
        """
        text = str(narrative).lower()
        sentences = re.split(r'[.!?]+', text)
        
        sequence = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:  # Skip very short sentences
                continue
                
            # Extract semantic tokens in order of appearance
            sent_tokens = []
            for category, pattern in self.entity_patterns.items():
                matches = list(re.finditer(pattern, sent))
                for match in matches:
                    sent_tokens.append({
                        'category': category,
                        'word': match.group(),
                        'position': match.start()
                    })
            
            # Sort by position to maintain order
            sent_tokens.sort(key=lambda x: x['position'])
            
            # Add to sequence
            if sent_tokens:
                sequence.append([t['category'] for t in sent_tokens])
        
        return sequence
    
    def mine_frequent_sequences(self, narratives, label=''):
        """
        Find frequent sequential patterns using a modified PrefixSpan approach
        """
        # Extract all sequences
        all_sequences = []
        for narrative in narratives:
            seq = self.extract_semantic_sequence(narrative)
            if seq:
                all_sequences.append(seq)
        
        # Mine patterns of length 2-5
        pattern_counts = defaultdict(int)
        
        for sequence in all_sequences:
            # Flatten sequence for pattern matching
            flat_seq = [item for sublist in sequence for item in sublist]
            
            # Extract sequential patterns with gaps allowed
            for length in range(2, min(6, len(flat_seq) + 1)):
                for i in range(len(flat_seq) - length + 1):
                    # Allow gaps of up to max_gap tokens
                    for gap in range(0, self.max_gap + 1):
                        if i + length + gap <= len(flat_seq):
                            pattern = tuple(flat_seq[i:i+length+gap:max(1, gap+1)])
                            if len(pattern) == length:  # Ensure we got the right length
                                pattern_counts[pattern] += 1
        
        # Filter by minimum support
        min_count = int(len(all_sequences) * self.min_support)
        frequent_patterns = {
            pattern: count 
            for pattern, count in pattern_counts.items() 
            if count >= min_count
        }
        
        # Calculate metrics
        results = []
        for pattern, count in frequent_patterns.items():
            results.append({
                'pattern': ' -> '.join(pattern),
                'sequence': pattern,
                'count': count,
                'support': count / len(all_sequences),
                'length': len(pattern)
            })
        
        return pd.DataFrame(results).sort_values('support', ascending=False)
    
    def mine_discriminative_sequences(self, ni_narratives, sar_narratives):
        """
        Find sequences that strongly differentiate between classes
        """
        # Mine patterns from both classes
        ni_patterns = self.mine_frequent_sequences(ni_narratives, 'NonIssue')
        sar_patterns = self.mine_frequent_sequences(sar_narratives, 'SAR')
        
        # Create lookup dictionaries
        ni_dict = {tuple(row['sequence']): row['support'] for _, row in ni_patterns.iterrows()}
        sar_dict = {tuple(row['sequence']): row['support'] for _, row in sar_patterns.iterrows()}
        
        # Find discriminative patterns
        all_patterns = set(ni_dict.keys()) | set(sar_dict.keys())
        
        discriminative = []
        for pattern in all_patterns:
            ni_sup = ni_dict.get(pattern, 0)
            sar_sup = sar_dict.get(pattern, 0)
            
            if ni_sup > 0 or sar_sup > 0:
                # Calculate discriminative score
                ratio = (ni_sup + 1e-6) / (sar_sup + 1e-6)
                
                discriminative.append({
                    'pattern': ' -> '.join(pattern),
                    'ni_support': ni_sup,
                    'sar_support': sar_sup,
                    'ratio': ratio,
                    'dominant_class': 'NonIssue' if ratio > 1 else 'SAR',
                    'discrimination_score': abs(ni_sup - sar_sup)
                })
        
        return pd.DataFrame(discriminative).sort_values('discrimination_score', ascending=False)
    
    def extract_reasoning_chains(self, narratives):
        """
        Extract common reasoning patterns: action -> observation -> conclusion
        """
        chains = []
        
        action_words = ['reviewed', 'analyzed', 'examined', 'investigated', 'checked', 'contacted']
        observation_words = ['found', 'showed', 'indicated', 'revealed', 'noted', 'observed']
        conclusion_words = ['determined', 'concluded', 'assessed', 'closed', 'filed', 'decided']
        
        for narrative in narratives:
            text = str(narrative).lower()
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            # Look for chains across consecutive sentences
            for i in range(len(sentences) - 2):
                window = ' '.join(sentences[i:i+3])
                
                has_action = any(word in window for word in action_words)
                has_observation = any(word in window for word in observation_words)
                has_conclusion = any(word in window for word in conclusion_words)
                
                if has_action and has_observation and has_conclusion:
                    chains.append({
                        'chain': window[:300],  # First 300 chars
                        'pattern_type': 'action-observation-conclusion'
                    })
        
        return chains
    
    def analyze_narrative_structure(self, narratives):
        """
        Analyze the macro-structure of narratives:
        - Do they start with conclusions or facts?
        - Where do risk indicators appear?
        - What's mentioned early vs late?
        """
        structure_analysis = {
            'starts_with_conclusion': 0,
            'starts_with_action': 0,
            'starts_with_transaction': 0,
            'risk_early': 0,
            'risk_late': 0,
            'verification_early': 0,
            'verification_late': 0
        }
        
        for narrative in narratives:
            text = str(narrative).lower()
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            if len(sentences) < 2:
                continue
            
            first_sent = sentences[0]
            last_third = ' '.join(sentences[-len(sentences)//3:])
            first_third = ' '.join(sentences[:len(sentences)//3])
            
            # Analyze opening
            if any(word in first_sent for word in ['concluded', 'determined', 'closed', 'filed']):
                structure_analysis['starts_with_conclusion'] += 1
            elif any(word in first_sent for word in ['reviewed', 'analyzed', 'investigated']):
                structure_analysis['starts_with_action'] += 1
            elif any(word in first_sent for word in ['transaction', 'deposit', 'transfer', 'payment']):
                structure_analysis['starts_with_transaction'] += 1
            
            # Risk indicator placement
            if re.search(self.entity_patterns['risk_indicator'], first_third):
                structure_analysis['risk_early'] += 1
            if re.search(self.entity_patterns['risk_indicator'], last_third):
                structure_analysis['risk_late'] += 1
            
            # Verification placement
            if re.search(self.entity_patterns['verification'], first_third):
                structure_analysis['verification_early'] += 1
            if re.search(self.entity_patterns['verification'], last_third):
                structure_analysis['verification_late'] += 1
        
        # Normalize
        total = len(narratives)
        return {k: v/total for k, v in structure_analysis.items()}


def run_sequential_analysis(non_issue_df, sar_df):
    """
    Run comprehensive sequential pattern analysis
    """
    print("="*80)
    print("SEQUENTIAL PATTERN MINING FOR AML INVESTIGATIONS")
    print("="*80)
    
    miner = SequentialPatternMiner(min_support=0.05, max_gap=2)
    
    # Mine discriminative sequences
    print("\n1. Mining discriminative sequential patterns...")
    discriminative = miner.mine_discriminative_sequences(
        non_issue_df['NARRATIVE'].values,
        sar_df['NARRATIVE'].values
    )
    
    print("\n" + "="*80)
    print("TOP SEQUENTIAL PATTERNS IN NON-ISSUE (Investigation Workflow)")
    print("="*80)
    ni_patterns = discriminative[discriminative['dominant_class'] == 'NonIssue'].head(20)
    print(ni_patterns[['pattern', 'ni_support', 'sar_support']])
    
    print("\n" + "="*80)
    print("TOP SEQUENTIAL PATTERNS IN SAR (Investigation Workflow)")
    print("="*80)
    sar_patterns = discriminative[discriminative['dominant_class'] == 'SAR'].head(20)
    print(sar_patterns[['pattern', 'ni_support', 'sar_support']])
    
    # Extract reasoning chains
    print("\n2. Extracting reasoning chains...")
    ni_chains = miner.extract_reasoning_chains(non_issue_df['NARRATIVE'].values)
    sar_chains = miner.extract_reasoning_chains(sar_df['NARRATIVE'].values)
    
    print(f"\nReasoning chains found:")
    print(f"  Non-Issue: {len(ni_chains)}")
    print(f"  SAR: {len(sar_chains)}")
    
    if ni_chains:
        print("\nExample Non-Issue reasoning chain:")
        print(f"  {ni_chains[0]['chain'][:250]}...")
    
    if sar_chains:
        print("\nExample SAR reasoning chain:")
        print(f"  {sar_chains[0]['chain'][:250]}...")
    
    # Analyze narrative structure
    print("\n3. Analyzing narrative structure patterns...")
    ni_structure = miner.analyze_narrative_structure(non_issue_df['NARRATIVE'].values)
    sar_structure = miner.analyze_narrative_structure(sar_df['NARRATIVE'].values)
    
    print("\n" + "="*80)
    print("NARRATIVE STRUCTURE COMPARISON")
    print("="*80)
    
    structure_comparison = pd.DataFrame({
        'Structure Element': ni_structure.keys(),
        'Non-Issue %': [f"{v*100:.1f}%" for v in ni_structure.values()],
        'SAR %': [f"{v*100:.1f}%" for v in sar_structure.values()]
    })
    print(structure_comparison)
    
    # Key insights
    print("\n" + "="*80)
    print("KEY SEQUENTIAL INSIGHTS")
    print("="*80)
    
    print("\n📊 Workflow Differences:")
    print(f"  • Non-Issue narratives start with conclusions {ni_structure['starts_with_conclusion']*100:.0f}% of the time")
    print(f"  • SAR narratives start with conclusions {sar_structure['starts_with_conclusion']*100:.0f}% of the time")
    print(f"  • Non-Issue mentions verification early {ni_structure['verification_early']*100:.0f}% vs SAR {sar_structure['verification_early']*100:.0f}%")
    print(f"  • Risk indicators appear early in SAR {sar_structure['risk_early']*100:.0f}% vs Non-Issue {ni_structure['risk_early']*100:.0f}%")
    
    return {
        'discriminative_sequences': discriminative,
        'ni_chains': ni_chains,
        'sar_chains': sar_chains,
        'ni_structure': ni_structure,
        'sar_structure': sar_structure,
        'structure_comparison': structure_comparison
    }


# Example usage:
"""
# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run sequential analysis
results = run_sequential_analysis(non_issue_df, sar_df)

# Save results
results['discriminative_sequences'].to_csv('sequential_patterns.csv', index=False)

# Analyze specific patterns
# Look for patterns that indicate thorough investigation (verification -> found -> concluded)
thorough_patterns = results['discriminative_sequences'][
    results['discriminative_sequences']['pattern'].str.contains('verification.*found.*concluded')
]

print("\nPatterns indicating thorough investigation:")
print(thorough_patterns)

# Export reasoning chains for detailed review
pd.DataFrame(results['ni_chains']).to_csv('ni_reasoning_chains.csv', index=False)
pd.DataFrame(results['sar_chains']).to_csv('sar_reasoning_chains.csv', index=False)
"""