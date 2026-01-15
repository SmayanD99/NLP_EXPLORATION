import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from itertools import combinations, product
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class RareContrastSetMiner:
    """
    Mine RARE but statistically significant patterns that distinguish SAR from non-issue.
    
    This finds combinations like:
    - "High jewelry spend" + "new business" + "narrative mentions 'gift cards'" = 8% of SARs, 0.5% of non-issue
    - "Electronics merchant" + "weekend deposits" + "round amounts" + "narrative says 'customer requested'" = rare but highly suspicious
    
    Focus on patterns that appear in 5-15% of SARs but <2% of non-issue (precision over recall)
    """
    
    def __init__(self, min_support_sar=0.05, max_support_sar=0.20, 
                 max_support_non_issue=0.02, min_confidence=0.70):
        self.min_support_sar = min_support_sar
        self.max_support_sar = max_support_sar
        self.max_support_non_issue = max_support_non_issue
        self.min_confidence = min_confidence
        
    def extract_narrative_features(self, narrative):
        """
        Extract binary features from narratives that could interact with transaction features
        """
        if pd.isna(narrative):
            return {}
        
        text = str(narrative).lower()
        
        features = {
            # Customer behavior signals
            'narrative_customer_requested': bool(re.search(r'customer (requested|asked|instructed|demanded)', text)),
            'narrative_customer_refused': bool(re.search(r'(refused|declined|unwilling|would not)', text)),
            'narrative_customer_evasive': bool(re.search(r'(evasive|vague|unclear|unable to explain)', text)),
            'narrative_customer_pressured': bool(re.search(r'(urgent|immediately|rush|time sensitive)', text)),
            
            # Documentation signals
            'narrative_no_documentation': bool(re.search(r'(no documentation|lack of (records|invoices|receipts)|unable to provide)', text)),
            'narrative_inconsistent_docs': bool(re.search(r'(inconsistent|conflicting|discrepancy|mismatch)', text)),
            'narrative_altered_docs': bool(re.search(r'(altered|modified|changed|tampered)', text)),
            
            # Business legitimacy signals
            'narrative_new_business': bool(re.search(r'(new business|recently (opened|formed|incorporated)|startup)', text)),
            'narrative_shell_company': bool(re.search(r'(shell company|no physical|no operations|no employees)', text)),
            'narrative_cash_intensive': bool(re.search(r'cash[ -]intensive', text)),
            'narrative_high_risk_industry': bool(re.search(r'(money service|remittance|crypto|virtual currency|precious metal|jewelry)', text)),
            
            # Transaction pattern signals
            'narrative_structuring': bool(re.search(r'(structur|smurfing|layering|below reporting)', text)),
            'narrative_round_amounts': bool(re.search(r'(round (dollar )?amount|even amount)', text)),
            'narrative_rapid_movement': bool(re.search(r'(rapid|quick|immediate|same day|shortly after)', text)),
            'narrative_multiple_parties': bool(re.search(r'(multiple (parties|entities|accounts)|third[- ]party)', text)),
            
            # Geographic signals
            'narrative_high_risk_country': bool(re.search(r'(offshore|foreign|international|overseas)', text)),
            'narrative_multiple_jurisdictions': bool(re.search(r'(multiple (countries|jurisdictions)|cross[- ]border)', text)),
            
            # Investigator findings
            'narrative_unable_to_verify': bool(re.search(r'unable to (verify|confirm|validate)', text)),
            'narrative_attempted_contact': bool(re.search(r'(attempted to contact|tried to reach|no response)', text)),
            'narrative_conflicting_info': bool(re.search(r'(conflicting|contradictory|discrepan)', text)),
            
            # Positive signals (green flags)
            'narrative_verified_source': bool(re.search(r'(verified|confirmed|documented) (with|by|through)', text)),
            'narrative_legitimate_business': bool(re.search(r'(legitimate business|normal (business|commercial)|consistent with)', text)),
            'narrative_known_customer': bool(re.search(r'(long[- ]standing|established|known) customer', text)),
        }
        
        return features
    
    def create_combined_feature_space(self, df, narrative_col='NARRATIVE'):
        """
        Combine transaction features with narrative-derived features
        """
        # Extract narrative features
        narrative_features = df[narrative_col].apply(self.extract_narrative_features)
        narrative_df = pd.DataFrame(list(narrative_features))
        
        # Get transaction features (exclude ID and narrative columns)
        exclude_cols = ['ALERT_ID', 'SAR_ID', 'NARRATIVE', 'label']
        trans_cols = [col for col in df.columns if col not in exclude_cols]
        
        if trans_cols:
            # Combine
            combined_df = pd.concat([df[trans_cols].reset_index(drop=True), 
                                    narrative_df.reset_index(drop=True)], axis=1)
        else:
            combined_df = narrative_df
        
        return combined_df
    
    def discretize_continuous_features(self, df, n_bins=5):
        """
        Convert continuous features to categorical bins for pattern mining
        """
        df_discrete = df.copy()
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's already binary
                unique_vals = df[col].nunique()
                if unique_vals <= 2:
                    df_discrete[col] = df[col].astype(bool)
                elif unique_vals <= 10:
                    # Keep as is if few categories
                    df_discrete[col] = df[col].astype(str)
                else:
                    # Bin continuous features
                    df_discrete[col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                    df_discrete[col] = f"{col}_q" + df_discrete[col].astype(str)
        
        return df_discrete
    
    def mine_emerging_patterns(self, sar_df, non_issue_df):
        """
        Find patterns that emerge strongly in SAR class (Emerging Pattern Mining)
        These are combinations of features that are rare but highly discriminative
        """
        # Convert to binary/categorical
        sar_discrete = self.discretize_continuous_features(sar_df)
        non_issue_discrete = self.discretize_continuous_features(non_issue_df)
        
        # Mine frequent itemsets in each class
        sar_patterns = self._find_frequent_itemsets(sar_discrete, 
                                                     min_support=self.min_support_sar,
                                                     max_support=self.max_support_sar)
        
        non_issue_patterns = self._find_frequent_itemsets(non_issue_discrete,
                                                          min_support=0,
                                                          max_support=self.max_support_non_issue)
        
        # Calculate growth rates (emerging patterns)
        emerging = []
        
        for pattern, sar_support in sar_patterns.items():
            non_issue_support = non_issue_patterns.get(pattern, 0)
            
            if non_issue_support < self.max_support_non_issue:
                # Calculate growth rate
                growth_rate = sar_support / (non_issue_support + 1e-6)
                
                # Calculate statistical significance
                contingency = np.array([
                    [sar_support * len(sar_df), (1 - sar_support) * len(sar_df)],
                    [non_issue_support * len(non_issue_df), (1 - non_issue_support) * len(non_issue_df)]
                ])
                
                if contingency.min() >= 5:
                    _, p_value, _, _ = chi2_contingency(contingency)
                else:
                    _, p_value = fisher_exact(contingency)
                
                # Calculate confidence (precision)
                confidence = (sar_support * len(sar_df)) / (
                    sar_support * len(sar_df) + non_issue_support * len(non_issue_df) + 1e-6
                )
                
                if confidence >= self.min_confidence and p_value < 0.01:
                    emerging.append({
                        'pattern': ' + '.join(sorted(pattern)),
                        'pattern_items': pattern,
                        'sar_support': sar_support,
                        'non_issue_support': non_issue_support,
                        'growth_rate': growth_rate,
                        'confidence': confidence,
                        'p_value': p_value,
                        'sar_count': int(sar_support * len(sar_df)),
                        'non_issue_count': int(non_issue_support * len(non_issue_df))
                    })
        
        return pd.DataFrame(emerging).sort_values('growth_rate', ascending=False)
    
    def _find_frequent_itemsets(self, df, min_support=0.05, max_support=1.0, max_length=4):
        """
        Simplified frequent itemset mining using Apriori-like approach
        """
        n_transactions = len(df)
        patterns = {}
        
        # Convert DataFrame to transaction format
        transactions = []
        for _, row in df.iterrows():
            transaction = set()
            for col in df.columns:
                if pd.notna(row[col]):
                    if isinstance(row[col], bool):
                        if row[col]:
                            transaction.add(col)
                    elif isinstance(row[col], (int, float)):
                        if row[col] == 1 or row[col] == True:
                            transaction.add(col)
                    else:
                        transaction.add(f"{col}={row[col]}")
            transactions.append(transaction)
        
        # Mine patterns of increasing length
        for length in range(2, max_length + 1):
            for transaction in transactions:
                if len(transaction) >= length:
                    for itemset in combinations(sorted(transaction), length):
                        patterns[itemset] = patterns.get(itemset, 0) + 1
        
        # Filter by support
        filtered_patterns = {
            pattern: count / n_transactions
            for pattern, count in patterns.items()
            if min_support <= count / n_transactions <= max_support
        }
        
        return filtered_patterns
    
    def explain_pattern(self, pattern_items, sar_df, non_issue_df, narrative_col='NARRATIVE'):
        """
        Get example narratives and statistics for a specific pattern
        """
        # Find examples in SAR
        sar_matches = self._match_pattern(sar_df, pattern_items)
        non_issue_matches = self._match_pattern(non_issue_df, pattern_items)
        
        explanation = {
            'pattern': ' + '.join(sorted(pattern_items)),
            'sar_examples': sar_matches.head(5)[narrative_col].tolist() if len(sar_matches) > 0 else [],
            'non_issue_examples': non_issue_matches.head(3)[narrative_col].tolist() if len(non_issue_matches) > 0 else [],
            'sar_count': len(sar_matches),
            'non_issue_count': len(non_issue_matches)
        }
        
        return explanation
    
    def _match_pattern(self, df, pattern_items):
        """
        Find rows that match a pattern
        """
        mask = pd.Series([True] * len(df), index=df.index)
        
        for item in pattern_items:
            if '=' in item:
                col, val = item.split('=', 1)
                if col in df.columns:
                    mask &= (df[col].astype(str) == val)
            else:
                if item in df.columns:
                    mask &= (df[item] == True) | (df[item] == 1)
        
        return df[mask]


def run_rare_pattern_discovery(non_issue_df, sar_df):
    """
    Discover rare but highly indicative patterns
    """
    print("="*80)
    print("RARE PATTERN DISCOVERY FOR AML ALERTS")
    print("="*80)
    
    miner = RareContrastSetMiner(
        min_support_sar=0.05,      # Pattern appears in 5%+ of SARs
        max_support_sar=0.20,       # But not more than 20% (keep it rare)
        max_support_non_issue=0.02, # And less than 2% of non-issue
        min_confidence=0.70         # 70%+ precision
    )
    
    print("\n1. Creating combined feature space...")
    sar_features = miner.create_combined_feature_space(sar_df)
    non_issue_features = miner.create_combined_feature_space(non_issue_df)
    
    print(f"   SAR features: {sar_features.shape}")
    print(f"   Non-Issue features: {non_issue_features.shape}")
    
    print("\n2. Mining emerging patterns (this may take a few minutes)...")
    emerging_patterns = miner.mine_emerging_patterns(sar_features, non_issue_features)
    
    print("\n" + "="*80)
    print("TOP RARE BUT HIGHLY SUSPICIOUS PATTERNS")
    print("="*80)
    print("\nThese patterns are:")
    print("  • RARE: Appear in only 5-20% of SARs")
    print("  • DISCRIMINATIVE: Nearly absent in non-issue alerts (<2%)")
    print("  • STATISTICALLY SIGNIFICANT: p < 0.01")
    print("  • HIGH PRECISION: 70%+ confidence")
    print()
    
    if len(emerging_patterns) > 0:
        top_patterns = emerging_patterns.head(20)
        
        for idx, row in top_patterns.iterrows():
            print(f"\nPattern #{idx + 1}:")
            print(f"  {row['pattern']}")
            print(f"  • Found in: {row['sar_count']} SARs ({row['sar_support']*100:.1f}%) vs {row['non_issue_count']} non-issue ({row['non_issue_support']*100:.2f}%)")
            print(f"  • Growth Rate: {row['growth_rate']:.1f}x more likely in SARs")
            print(f"  • Precision: {row['confidence']*100:.1f}%")
            print(f"  • Statistical significance: p={row['p_value']:.4f}")
        
        # Get detailed examples for top pattern
        if len(top_patterns) > 0:
            print("\n" + "="*80)
            print("DETAILED EXAMPLE: TOP PATTERN")
            print("="*80)
            
            top_pattern = top_patterns.iloc[0]
            explanation = miner.explain_pattern(
                top_pattern['pattern_items'],
                sar_df,
                non_issue_df
            )
            
            print(f"\nPattern: {explanation['pattern']}")
            print(f"\nExample SAR narratives with this pattern:")
            for i, narrative in enumerate(explanation['sar_examples'][:3], 1):
                print(f"\n{i}. {str(narrative)[:300]}...")
            
            if explanation['non_issue_examples']:
                print(f"\nRare non-issue examples (for comparison):")
                for i, narrative in enumerate(explanation['non_issue_examples'][:2], 1):
                    print(f"\n{i}. {str(narrative)[:300]}...")
        
        # Save results
        emerging_patterns.to_csv('rare_emerging_patterns.csv', index=False)
        print("\n" + "="*80)
        print(f"✓ Results saved to 'rare_emerging_patterns.csv'")
        print(f"✓ Found {len(emerging_patterns)} rare but highly suspicious patterns")
        
        return emerging_patterns
    else:
        print("\nNo patterns found matching the criteria.")
        print("Try adjusting parameters (min_support_sar, max_support_non_issue, min_confidence)")
        return pd.DataFrame()


# Example usage:
"""
# Load your data with transaction features already included
non_issue_df = pd.read_csv('non_issue_alerts_with_features.csv')
sar_df = pd.read_csv('sar_alerts_with_features.csv')

# Make sure you have a NARRATIVE column
# Other columns can be any transaction features from your 130+ feature model

# Run rare pattern discovery
patterns = run_rare_pattern_discovery(non_issue_df, sar_df)

# Analyze specific patterns
if len(patterns) > 0:
    # Get patterns involving specific features
    jewelry_patterns = patterns[patterns['pattern'].str.contains('jewelry|jewellery')]
    print(f"\nFound {len(jewelry_patterns)} patterns involving jewelry purchases")
    
    # Get patterns with narrative signals
    narrative_patterns = patterns[patterns['pattern'].str.contains('narrative_')]
    print(f"\nFound {len(narrative_patterns)} patterns combining transactions + narrative signals")
"""