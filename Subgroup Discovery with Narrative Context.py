import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from scipy.stats import fisher_exact
import re
import warnings
warnings.filterwarnings('ignore')

class ImbalancedSubgroupDiscovery:
    """
    Subgroup discovery optimized for 14,000 vs 680 imbalance
    
    Key changes:
    1. Relaxed precision threshold (60% instead of 70%)
    2. Lower minimum support (20 SARs instead of 50)
    3. Balanced class weights in decision tree
    4. Multiple tree depths to find patterns at different granularities
    """
    
    def __init__(self, min_support=20, min_precision=0.60, max_depth=4):
        self.min_support = min_support
        self.min_precision = min_precision
        self.max_depth = max_depth
        
    def extract_narrative_features(self, narrative):
        """Extract contextual features from narrative"""
        if pd.isna(narrative):
            return {}
        
        text = str(narrative).lower()
        
        features = {
            # Verification status
            'ctx_verification_failed': bool(re.search(r'unable to (verify|confirm)|could not (verify|confirm)|failed to verify', text)),
            'ctx_verification_success': bool(re.search(r'verified (with|by|through)|confirmed (with|by)', text)),
            'ctx_documentation_missing': bool(re.search(r'no (documentation|records|invoices)|lack of documentation', text)),
            'ctx_documentation_provided': bool(re.search(r'provided (documentation|records|invoices)', text)),
            
            # Customer behavior
            'ctx_customer_initiated': bool(re.search(r'customer (requested|asked|instructed|initiated)', text)),
            'ctx_customer_explanation': bool(re.search(r'customer (explained|stated|indicated|claimed)', text)),
            'ctx_customer_no_explanation': bool(re.search(r'unable to explain|no explanation|could not explain', text)),
            'ctx_customer_unresponsive': bool(re.search(r'(no response|did not respond|unresponsive)', text)),
            
            # Timing context
            'ctx_timing_urgent': bool(re.search(r'urgent|immediately|rush|same day', text)),
            
            # Business maturity
            'ctx_new_customer': bool(re.search(r'new (customer|account|relationship)|recently opened', text)),
            'ctx_established_customer': bool(re.search(r'(long[- ]standing|established) (customer|relationship)', text)),
            
            # Pattern assessment
            'ctx_pattern_change': bool(re.search(r'(change|shift|deviation|departure) (in|from) (pattern|behavior)', text)),
            'ctx_pattern_consistent': bool(re.search(r'consistent with (prior|previous|historical)', text)),
            
            # Explanation quality
            'ctx_explanation_weak': bool(re.search(r'(vague|unclear|inconsistent|conflicting) explanation', text)),
            
            # Investigation depth
            'ctx_deep_investigation': bool(re.search(r'(contacted|interviewed|obtained|reviewed) (customer|records)', text)),
            
            # Suspicion level
            'ctx_strong_suspicion': bool(re.search(r'(highly|very|extremely) (suspicious|unusual|concerning)', text)),
            'ctx_mild_suspicion': bool(re.search(r'(somewhat|slightly|potentially|possibly) (suspicious|unusual)', text)),
            'ctx_no_suspicion': bool(re.search(r'(not suspicious|not concerning|normal|legitimate)', text)),
            
            # Specific patterns
            'ctx_structuring': bool(re.search(r'structur|smurfing|below reporting', text)),
            'ctx_round_amounts': bool(re.search(r'round (dollar )?amount|even amount', text)),
            'ctx_rapid_movement': bool(re.search(r'rapid|quick|immediate', text)),
            'ctx_multiple_parties': bool(re.search(r'multiple (parties|entities|accounts)|third[- ]party', text)),
            'ctx_high_risk_geo': bool(re.search(r'offshore|high[- ]risk (country|jurisdiction)', text)),
        }
        
        return features
    
    def build_contextual_rules(self, df, label_col='label', narrative_col='NARRATIVE'):
        """
        Build rules with class-imbalance aware parameters
        """
        # Extract narrative features
        context_features = df[narrative_col].apply(self.extract_narrative_features)
        context_df = pd.DataFrame(list(context_features))
        
        # Get transaction features if they exist
        exclude = ['ALERT_ID', 'SAR_ID', 'NARRATIVE', 'label']
        trans_features = [col for col in df.columns if col not in exclude]
        
        if trans_features:
            X = pd.concat([
                df[trans_features].reset_index(drop=True),
                context_df.reset_index(drop=True)
            ], axis=1)
        else:
            X = context_df
        
        y = df[label_col].values
        
        # Try multiple configurations to find rules
        all_rules = []
        
        for depth in [2, 3, 4]:
            for min_samples in [self.min_support, self.min_support * 2]:
                try:
                    clf = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=min_samples,
                        min_samples_leaf=int(min_samples / 2),
                        class_weight='balanced',  # Critical for imbalance
                        random_state=42
                    )
                    
                    clf.fit(X, y)
                    
                    # Extract rules
                    rules = self._extract_rules_from_tree(clf, X.columns, y)
                    all_rules.extend(rules)
                    
                except Exception as e:
                    continue
        
        return all_rules, X
    
    def _extract_rules_from_tree(self, tree, feature_names, y):
        """Extract rules from decision tree"""
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
            for i in tree_.feature
        ]
        
        rules = []
        
        def recurse(node, conditions, depth):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                
                # Left child
                left_conditions = conditions + [(name, '<=', threshold)]
                recurse(tree_.children_left[node], left_conditions, depth + 1)
                
                # Right child
                right_conditions = conditions + [(name, '>', threshold)]
                recurse(tree_.children_right[node], right_conditions, depth + 1)
            else:
                # Leaf node
                samples = tree_.n_node_samples[node]
                values = tree_.value[node][0]
                sar_count = int(values[1]) if len(values) > 1 else 0
                non_issue_count = int(values[0])
                
                if sar_count >= self.min_support and samples >= self.min_support:
                    precision = sar_count / samples
                    
                    if precision >= self.min_precision:
                        rules.append({
                            'conditions': conditions,
                            'sar_count': sar_count,
                            'non_issue_count': non_issue_count,
                            'total_support': samples,
                            'precision': precision,
                            'depth': depth
                        })
        
        recurse(0, [], 0)
        return rules
    
    def find_contextual_rules(self, non_issue_df, sar_df, narrative_col='NARRATIVE'):
        """Find contextual rules"""
        print("Building contextual rules...")
        
        # Prepare data
        sar_labeled = sar_df.copy()
        sar_labeled['label'] = 1
        non_issue_labeled = non_issue_df.copy()
        non_issue_labeled['label'] = 0
        
        combined_df = pd.concat([sar_labeled, non_issue_labeled], ignore_index=True)
        
        # Build rules
        rules, X = self.build_contextual_rules(combined_df, narrative_col=narrative_col)
        
        if not rules:
            return pd.DataFrame(), X
        
        # Format rules
        formatted_rules = []
        baseline_sar_rate = len(sar_df) / len(combined_df)
        
        for rule in rules:
            rule_text = self._format_rule_conditions(rule['conditions'])
            lift = rule['precision'] / baseline_sar_rate
            
            formatted_rules.append({
                'rule': rule_text,
                'conditions': rule['conditions'],
                'sar_count': rule['sar_count'],
                'non_issue_count': rule['non_issue_count'],
                'precision': rule['precision'],
                'lift': lift,
                'support': rule['total_support'],
                'depth': rule['depth']
            })
        
        df = pd.DataFrame(formatted_rules)
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['rule'])
        df = df.sort_values(['precision', 'sar_count'], ascending=[False, False])
        
        return df, X
    
    def _format_rule_conditions(self, conditions):
        """Format conditions into readable rule"""
        parts = []
        for feature, op, threshold in conditions:
            if isinstance(threshold, (int, float)):
                if threshold == 0.5:  # Boolean feature
                    if op == '>':
                        parts.append(f"[{feature}=True]")
                    else:
                        parts.append(f"[{feature}=False]")
                else:
                    parts.append(f"[{feature} {op} {threshold:.2f}]")
            else:
                parts.append(f"[{feature} {op} {threshold}]")
        
        return " AND ".join(parts)


def run_imbalanced_subgroup_discovery(non_issue_df, sar_df, narrative_col='NARRATIVE'):
    """
    Run subgroup discovery optimized for severe imbalance
    """
    print("="*80)
    print("CONTEXTUAL SUBGROUP DISCOVERY - IMBALANCE OPTIMIZED")
    print("="*80)
    print(f"\nDataset: {len(non_issue_df):,} non-issue vs {len(sar_df):,} SAR")
    print(f"Imbalance ratio: {len(non_issue_df)/len(sar_df):.1f}:1")
    print("\nSearching for rules with:")
    print("  • Minimum 20 SARs per rule")
    print("  • Minimum 60% precision")
    print("  • Multiple tree depths (2-4)")
    print("  • Balanced class weights\n")
    
    discoverer = ImbalancedSubgroupDiscovery(
        min_support=20,
        min_precision=0.60,
        max_depth=4
    )
    
    # Find rules
    rules_df, X = discoverer.find_contextual_rules(
        non_issue_df,
        sar_df,
        narrative_col=narrative_col
    )
    
    if len(rules_df) == 0:
        print("\n" + "="*80)
        print("NO RULES FOUND")
        print("="*80)
        print("\nTry adjusting parameters:")
        print("  • Lower min_support: 20 → 15 or 10")
        print("  • Lower min_precision: 0.60 → 0.55 or 0.50")
        print("  • Increase max_depth: 4 → 5")
        print("\nOR add more narrative features to extract_narrative_features()")
        return pd.DataFrame()
    
    print("\n" + "="*80)
    print(f"FOUND {len(rules_df)} CONTEXTUAL RULES")
    print("="*80)
    
    # Show top rules
    print(f"\nTop 20 rules by precision:\n")
    print(f"{'Rule':<100} {'SAR':<6} {'NI':<6} {'Prec':<7} {'Lift':<6}")
    print("-" * 130)
    
    for _, rule in rules_df.head(20).iterrows():
        rule_str = rule['rule'][:95] + "..." if len(rule['rule']) > 95 else rule['rule']
        print(f"{rule_str:<100} "
              f"{rule['sar_count']:>5} "
              f"{rule['non_issue_count']:>5} "
              f"{rule['precision']*100:>6.1f}% "
              f"{rule['lift']:>5.1f}x")
    
    # Show detailed examples
    print("\n" + "="*80)
    print("DETAILED RULE EXAMPLES")
    print("="*80)
    
    for idx, rule in rules_df.head(5).iterrows():
        print(f"\n{'='*70}")
        print(f"Rule {idx + 1}:")
        print(f"{rule['rule']}")
        print(f"  • Found in {rule['sar_count']} SARs, {rule['non_issue_count']} non-issue")
        print(f"  • Precision: {rule['precision']*100:.1f}%")
        print(f"  • Lift: {rule['lift']:.1f}x over baseline")
        print(f"  • Rule depth: {rule['depth']}")
    
    # Save
    rules_df.to_csv('contextual_rules_imbalanced.csv', index=False)
    
    print("\n" + "="*80)
    print(f"✓ Results saved to 'contextual_rules_imbalanced.csv'")
    print(f"✓ Found {len(rules_df)} contextual rules")
    
    return rules_df


# USAGE:
"""
import pandas as pd

non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run with relaxed parameters
rules = run_imbalanced_subgroup_discovery(non_issue_df, sar_df)

# If still no rules found, try even more relaxed:
discoverer = ImbalancedSubgroupDiscovery(
    min_support=15,      # Lower to 15 SARs
    min_precision=0.55,  # Lower to 55% precision
    max_depth=5          # Deeper trees
)

rules_df, X = discoverer.find_contextual_rules(non_issue_df, sar_df)

# Analyze specific rule types
if len(rules_df) > 0:
    # Rules involving verification failure
    verification_rules = rules_df[
        rules_df['rule'].str.contains('verification_failed', case=False)
    ]
    print(f"Rules with verification failure: {len(verification_rules)}")
"""
