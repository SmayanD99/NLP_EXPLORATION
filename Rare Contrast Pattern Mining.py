import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from itertools import combinations
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ImbalancedRarePatternMiner:
    """
    Rare pattern mining optimized for 14,000 vs 680 imbalance
    
    Key adaptations:
    1. SAR-focused: Look for patterns in 30-150 SARs (5-22% of minority class)
    2. Absolute counts, not percentages: min 30 SAR, max 30 non-issue
    3. Fisher's exact test for significance
    4. High precision requirement (>80%) to avoid false positives
    """
    
    def __init__(self, min_sar_support=30, max_sar_support=150, 
                 max_ni_support=30, min_precision=0.80):
        self.min_sar_support = min_sar_support      # ~5% of 680 SARs
        self.max_sar_support = max_sar_support      # ~22% of 680 SARs
        self.max_ni_support = max_ni_support        # ~0.2% of 14,000 non-issue
        self.min_precision = min_precision
        
    def extract_narrative_features(self, narrative):
        """Extract binary features from narratives"""
        if pd.isna(narrative):
            return {}
        
        text = str(narrative).lower()
        
        features = {
            # Customer behavior
            'narr_customer_requested': bool(re.search(r'customer (requested|asked|instructed|demanded)', text)),
            'narr_customer_refused': bool(re.search(r'(refused|declined|unwilling|would not)', text)),
            'narr_customer_evasive': bool(re.search(r'(evasive|vague|unclear response|unable to explain)', text)),
            'narr_customer_unresponsive': bool(re.search(r'(no response|did not respond|failed to respond)', text)),
            
            # Documentation
            'narr_no_documentation': bool(re.search(r'(no documentation|lack of (records|invoices)|unable to provide)', text)),
            'narr_inconsistent_docs': bool(re.search(r'(inconsistent|conflicting|discrepancy)', text)),
            'narr_docs_provided': bool(re.search(r'(documentation|records|invoices) (provided|submitted|received)', text)),
            
            # Verification
            'narr_unable_verify': bool(re.search(r'unable to (verify|confirm|validate)', text)),
            'narr_verified': bool(re.search(r'verified (with|by|through)', text)),
            'narr_attempted_contact': bool(re.search(r'attempted to (contact|reach)', text)),
            
            # Business legitimacy
            'narr_new_business': bool(re.search(r'(new business|recently (opened|formed))', text)),
            'narr_cash_intensive': bool(re.search(r'cash[ -]intensive', text)),
            'narr_legitimate_business': bool(re.search(r'legitimate (business|activity|purpose)', text)),
            
            # Transaction patterns
            'narr_structuring': bool(re.search(r'(structur|smurfing|below reporting)', text)),
            'narr_round_amounts': bool(re.search(r'(round (dollar )?amount|even amount)', text)),
            'narr_rapid_movement': bool(re.search(r'(rapid|quick|immediate|same day)', text)),
            'narr_unusual_pattern': bool(re.search(r'unusual (pattern|activity|behavior)', text)),
            
            # Geographic
            'narr_high_risk_country': bool(re.search(r'(offshore|foreign|high[- ]risk (country|jurisdiction))', text)),
            'narr_multiple_jurisdictions': bool(re.search(r'(multiple (countries|jurisdictions)|cross[- ]border)', text)),
            
            # Conclusions
            'narr_suspicious': bool(re.search(r'\bsuspicious\b', text)),
            'narr_concerning': bool(re.search(r'\bconcerning\b', text)),
            'narr_normal': bool(re.search(r'(normal|typical|expected) (activity|behavior|pattern)', text)),
            'narr_consistent': bool(re.search(r'consistent with', text)),
        }
        
        return features
    
    def create_feature_matrix(self, df, narrative_col='NARRATIVE'):
        """Create binary feature matrix from narratives"""
        # Extract narrative features
        narrative_features = df[narrative_col].apply(self.extract_narrative_features)
        narrative_df = pd.DataFrame(list(narrative_features))
        
        # Get any transaction features if they exist
        exclude_cols = ['ALERT_ID', 'SAR_ID', 'NARRATIVE', 'label']
        trans_cols = [col for col in df.columns if col not in exclude_cols]
        
        if trans_cols:
            # Binarize transaction features if needed
            trans_df = df[trans_cols].copy()
            
            for col in trans_df.columns:
                if trans_df[col].dtype in ['int64', 'float64']:
                    unique_vals = trans_df[col].nunique()
                    if unique_vals > 10:
                        # Bin into high/low
                        median = trans_df[col].median()
                        trans_df[col] = (trans_df[col] > median).astype(int)
            
            combined_df = pd.concat([trans_df.reset_index(drop=True), 
                                    narrative_df.reset_index(drop=True)], axis=1)
        else:
            combined_df = narrative_df
        
        return combined_df
    
    def mine_rare_patterns(self, sar_df, non_issue_df):
        """
        Mine patterns that are rare but highly indicative
        Focus on minority class (SAR)
        """
        print("1. Creating feature matrices...")
        sar_features = self.create_feature_matrix(sar_df)
        ni_features = self.create_feature_matrix(non_issue_df)
        
        print(f"   SAR features: {sar_features.shape}")
        print(f"   Non-Issue features: {ni_features.shape}")
        
        # Find frequent itemsets in SAR class only
        print("\n2. Mining patterns in SAR class...")
        sar_patterns = self._find_itemsets(sar_features, 
                                           min_support=self.min_sar_support,
                                           max_support=self.max_sar_support)
        
        print(f"   Found {len(sar_patterns)} candidate patterns in SAR")
        
        # Check their frequency in non-issue
        print("\n3. Checking pattern frequency in non-issue class...")
        rare_discriminative = []
        
        for pattern, sar_count in sar_patterns.items():
            # Count in non-issue
            ni_count = self._count_pattern(ni_features, pattern)
            
            # Filter by non-issue frequency
            if ni_count > self.max_ni_support:
                continue
            
            # Calculate precision
            precision = sar_count / (sar_count + ni_count)
            
            if precision < self.min_precision:
                continue
            
            # Statistical significance (Fisher's exact)
            contingency = [
                [sar_count, len(sar_df) - sar_count],
                [ni_count, len(non_issue_df) - ni_count]
            ]
            
            try:
                odds_ratio, p_value = fisher_exact(contingency)
            except:
                continue
            
            if p_value >= 0.01:
                continue
            
            rare_discriminative.append({
                'pattern': ' + '.join(sorted(pattern)),
                'pattern_items': pattern,
                'sar_count': sar_count,
                'ni_count': ni_count,
                'sar_pct': sar_count / len(sar_df) * 100,
                'ni_pct': ni_count / len(non_issue_df) * 100,
                'precision': precision,
                'odds_ratio': odds_ratio,
                'p_value': p_value
            })
        
        if rare_discriminative:
            df = pd.DataFrame(rare_discriminative)
            df = df.sort_values('precision', ascending=False)
            return df
        
        return pd.DataFrame()
    
    def _find_itemsets(self, df, min_support, max_support, max_length=4):
        """Find frequent itemsets in minority class"""
        patterns = defaultdict(int)
        
        # Convert to transaction format
        for _, row in df.iterrows():
            transaction = set()
            for col in df.columns:
                if pd.notna(row[col]) and row[col] == 1:
                    transaction.add(col)
            
            # Generate itemsets of length 2-4
            for length in range(2, min(max_length + 1, len(transaction) + 1)):
                for itemset in combinations(sorted(transaction), length):
                    patterns[itemset] += 1
        
        # Filter by support
        filtered = {
            pattern: count
            for pattern, count in patterns.items()
            if min_support <= count <= max_support
        }
        
        return filtered
    
    def _count_pattern(self, df, pattern):
        """Count how many rows match a pattern"""
        mask = pd.Series([True] * len(df), index=df.index)
        
        for item in pattern:
            if item in df.columns:
                mask &= (df[item] == 1)
        
        return mask.sum()


def run_imbalanced_rare_pattern_mining(non_issue_df, sar_df):
    """
    Run rare pattern mining optimized for severe imbalance
    """
    print("="*80)
    print("RARE PATTERN MINING - IMBALANCE OPTIMIZED")
    print("="*80)
    print(f"Dataset: {len(non_issue_df):,} non-issue vs {len(sar_df):,} SAR")
    print(f"Imbalance ratio: {len(non_issue_df)/len(sar_df):.1f}:1")
    print("\nSearching for patterns that:")
    print("  • Appear in 30-150 SARs (5-22% of minority class)")
    print("  • Appear in <30 non-issue (<0.2% of majority class)")
    print("  • Have >80% precision")
    print("  • Are statistically significant (p<0.01)\n")
    
    miner = ImbalancedRarePatternMiner(
        min_sar_support=30,      # ~5% of 680
        max_sar_support=150,     # ~22% of 680
        max_ni_support=30,       # ~0.2% of 14,000
        min_precision=0.80
    )
    
    # Mine patterns
    patterns = miner.mine_rare_patterns(sar_df, non_issue_df)
    
    if len(patterns) > 0:
        print("\n" + "="*80)
        print("RARE BUT HIGHLY INDICATIVE PATTERNS")
        print("="*80)
        print(f"\nFound {len(patterns)} rare patterns meeting criteria:\n")
        
        print(f"{'Pattern':<80} {'SAR':<6} {'NI':<6} {'Prec':<6} {'OR':<8}")
        print("-" * 110)
        
        for _, row in patterns.head(30).iterrows():
            pattern_str = row['pattern'][:75] + "..." if len(row['pattern']) > 75 else row['pattern']
            print(f"{pattern_str:<80} "
                  f"{row['sar_count']:>5} "
                  f"{row['ni_count']:>5} "
                  f"{row['precision']*100:>5.1f}% "
                  f"{row['odds_ratio']:>7.1f}")
        
        # Show detailed examples for top patterns
        print("\n" + "="*80)
        print("DETAILED PATTERN EXAMPLES")
        print("="*80)
        
        for idx, row in patterns.head(5).iterrows():
            print(f"\n{'='*70}")
            print(f"Pattern: {row['pattern']}")
            print(f"  • Found in {row['sar_count']} SARs ({row['sar_pct']:.1f}%)")
            print(f"  • Found in {row['ni_count']} non-issue ({row['ni_pct']:.2f}%)")
            print(f"  • Precision: {row['precision']*100:.1f}%")
            print(f"  • Odds Ratio: {row['odds_ratio']:.1f}x")
            print(f"  • Statistical significance: p={row['p_value']:.2e}")
        
        # Save
        patterns.to_csv('rare_patterns_imbalanced.csv', index=False)
        
        print("\n" + "="*80)
        print(f"✓ Results saved to 'rare_patterns_imbalanced.csv'")
        print(f"✓ Found {len(patterns)} rare but high-precision SAR patterns")
        
        return patterns
    else:
        print("\nNo patterns found meeting the criteria.")
        print("Consider adjusting parameters:")
        print("  • Lower min_sar_support (currently 30)")
        print("  • Increase max_ni_support (currently 30)")
        print("  • Lower min_precision (currently 0.80)")
        return pd.DataFrame()


# Example usage:
"""
non_issue_df = pd.read_csv('non_issue_alerts.csv')  # Should have NARRATIVE column
sar_df = pd.read_csv('sar_alerts.csv')

patterns = run_imbalanced_rare_pattern_mining(non_issue_df, sar_df)

# Analyze specific patterns
if len(patterns) > 0:
    # Find patterns involving specific features
    verification_patterns = patterns[
        patterns['pattern'].str.contains('narr_unable_verify')
    ]
    print(f"\nPatterns involving verification failure: {len(verification_patterns)}")
    print(verification_patterns[['pattern', 'precision', 'sar_count']])
"""
