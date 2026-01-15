import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from scipy.stats import chi2_contingency
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NarrativeContextualSubgroupDiscovery:
    """
    Advanced subgroup discovery that finds suspicious transaction patterns CONDITIONAL
    on narrative context. This answers questions like:
    
    - "When do high jewelry purchases become suspicious?" 
      Answer: When narrative mentions "customer requested" + "no prior history" + weekends
    
    - "When are round amounts concerning?"
      Answer: In electronics merchants + when investigator notes "unable to verify" + business <6 months old
    
    This finds interaction effects between quantitative features and qualitative narrative signals
    that your 130-feature model might miss.
    """
    
    def __init__(self, min_support=50, min_lift=3.0, max_depth=4):
        self.min_support = min_support
        self.min_lift = min_lift
        self.max_depth = max_depth
        
    def extract_narrative_context_features(self, narrative):
        """
        Extract contextual signals that MODIFY how we interpret transaction features
        """
        if pd.isna(narrative):
            return {}
        
        text = str(narrative).lower()
        
        # These are MODIFIERS - they change the meaning of transaction patterns
        context = {
            # Verification status modifiers
            'ctx_verification_failed': bool(re.search(r'unable to (verify|confirm)|could not (verify|confirm)|failed to verify', text)),
            'ctx_verification_success': bool(re.search(r'verified (with|by|through)|confirmed (with|by)', text)),
            'ctx_documentation_missing': bool(re.search(r'no (documentation|records|invoices)|lack of documentation|unable to provide', text)),
            'ctx_documentation_provided': bool(re.search(r'provided (documentation|records|invoices|receipts)', text)),
            
            # Customer behavior modifiers
            'ctx_customer_initiated': bool(re.search(r'customer (requested|asked|instructed|initiated)', text)),
            'ctx_customer_passive': bool(re.search(r'automatic|recurring|scheduled|standing order', text)),
            'ctx_customer_explanation': bool(re.search(r'customer (explained|stated|indicated|claimed)', text)),
            'ctx_customer_no_explanation': bool(re.search(r'(unable to explain|no explanation|could not explain|refused to explain)', text)),
            
            # Timing context modifiers
            'ctx_timing_urgent': bool(re.search(r'urgent|immediately|rush|same day|time sensitive', text)),
            'ctx_timing_planned': bool(re.search(r'planned|scheduled|anticipated|expected', text)),
            
            # Business maturity modifiers
            'ctx_new_customer': bool(re.search(r'new (customer|account|relationship)|recently opened|just opened', text)),
            'ctx_established_customer': bool(re.search(r'(long[- ]standing|established|longterm) (customer|relationship)', text)),
            
            # Pattern change modifiers
            'ctx_pattern_change': bool(re.search(r'(change|shift|deviation|departure) (in|from) (pattern|behavior|activity)', text)),
            'ctx_pattern_consistent': bool(re.search(r'consistent with (prior|previous|historical|past)', text)),
            
            # Explanation quality modifiers
            'ctx_explanation_reasonable': bool(re.search(r'(reasonable|plausible|consistent|logical) explanation', text)),
            'ctx_explanation_weak': bool(re.search(r'(vague|unclear|inconsistent|conflicting|suspicious) explanation', text)),
            
            # Investigation depth modifiers
            'ctx_deep_investigation': bool(re.search(r'(contacted|interviewed|obtained|reviewed) (customer|records|documentation)', text)),
            'ctx_surface_investigation': bool(re.search(r'based on (screen review|system review)|cursory review', text)),
            
            # Red flag language intensity
            'ctx_strong_suspicion': bool(re.search(r'(highly suspicious|very concerning|extremely unusual|clearly suspicious)', text)),
            'ctx_mild_suspicion': bool(re.search(r'(somewhat|slightly|potentially|possibly) (suspicious|unusual|concerning)', text)),
            'ctx_no_suspicion': bool(re.search(r'(not suspicious|not concerning|normal|legitimate|consistent with)', text)),
        }
        
        return context
    
    def build_contextual_rules(self, df, label_col='label', narrative_col='NARRATIVE', 
                               transaction_features=None):
        """
        Build decision rules that combine transaction features with narrative context
        using a specialized decision tree approach
        """
        # Extract narrative context
        context_features = df[narrative_col].apply(self.extract_narrative_context_features)
        context_df = pd.DataFrame(list(context_features))
        
        # Get transaction features
        if transaction_features is None:
            exclude = ['ALERT_ID', 'SAR_ID', 'NARRATIVE', 'label']
            transaction_features = [col for col in df.columns if col not in exclude]
        
        # Combine features
        X = pd.concat([
            df[transaction_features].reset_index(drop=True),
            context_df.reset_index(drop=True)
        ], axis=1)
        
        y = df[label_col].values
        
        # Train decision tree with constraints for interpretability
        clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_support,
            min_samples_leaf=int(self.min_support / 2),
            class_weight='balanced'
        )
        
        clf.fit(X, y)
        
        # Extract rules
        rules = self._extract_rules_from_tree(clf, X.columns, y)
        
        return rules, clf, X
    
    def _extract_rules_from_tree(self, tree, feature_names, y):
        """
        Extract interpretable rules from decision tree
        """
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
                
                # Left child (<=)
                left_conditions = conditions + [(name, '<=', threshold)]
                recurse(tree_.children_left[node], left_conditions, depth + 1)
                
                # Right child (>)
                right_conditions = conditions + [(name, '>', threshold)]
                recurse(tree_.children_right[node], right_conditions, depth + 1)
            else:
                # Leaf node - extract rule
                samples = tree_.n_node_samples[node]
                values = tree_.value[node][0]
                sar_count = values[1] if len(values) > 1 else 0
                non_issue_count = values[0]
                
                if sar_count > 0 and samples >= self.min_support:
                    precision = sar_count / samples
                    
                    rules.append({
                        'conditions': conditions,
                        'sar_count': int(sar_count),
                        'non_issue_count': int(non_issue_count),
                        'total_support': int(samples),
                        'precision': precision,
                        'depth': depth
                    })
        
        recurse(0, [], 0)
        return rules
    
    def find_contextual_anomalies(self, sar_df, non_issue_df, 
                                  narrative_col='NARRATIVE', top_n=30):
        """
        Find transaction patterns that are NORMAL in non-issue but SUSPICIOUS when
        combined with specific narrative contexts
        """
        print("Building contextual rules for SAR alerts...")
        
        # Prepare data
        sar_labeled = sar_df.copy()
        sar_labeled['label'] = 1
        non_issue_labeled = non_issue_df.copy()
        non_issue_labeled['label'] = 0
        
        combined_df = pd.concat([sar_labeled, non_issue_labeled], ignore_index=True)
        
        # Build rules
        rules, clf, X = self.build_contextual_rules(
            combined_df,
            narrative_col=narrative_col
        )
        
        # Filter for high-precision SAR rules
        sar_rules = [r for r in rules if r['precision'] > 0.7 and r['sar_count'] >= self.min_support]
        sar_rules = sorted(sar_rules, key=lambda x: x['precision'] * x['sar_count'], reverse=True)[:top_n]
        
        # Format rules
        formatted_rules = []
        for rule in sar_rules:
            rule_text = self._format_rule_conditions(rule['conditions'])
            
            # Calculate lift
            baseline_sar_rate = sar_labeled.shape[0] / combined_df.shape[0]
            lift = rule['precision'] / baseline_sar_rate
            
            formatted_rules.append({
                'rule': rule_text,
                'conditions': rule['conditions'],
                'sar_count': rule['sar_count'],
                'non_issue_count': rule['non_issue_count'],
                'precision': rule['precision'],
                'lift': lift,
                'support': rule['total_support']
            })
        
        return pd.DataFrame(formatted_rules), clf, X
    
    def _format_rule_conditions(self, conditions):
        """
        Format conditions into readable rule
        """
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
    
    def analyze_feature_interactions(self, rules_df):
        """
        Analyze which narrative contexts most frequently appear with which transaction features
        """
        interaction_matrix = defaultdict(lambda: defaultdict(int))
        
        for _, rule in rules_df.iterrows():
            conditions = rule['conditions']
            
            trans_features = []
            context_features = []
            
            for feature, op, threshold in conditions:
                if feature.startswith('ctx_'):
                    if (op == '>' and threshold == 0.5) or (op == '<=' and threshold < 0.5):
                        context_features.append(feature)
                else:
                    trans_features.append(feature)
            
            # Record interactions
            for tf in trans_features:
                for cf in context_features:
                    interaction_matrix[tf][cf] += rule['sar_count']
        
        # Convert to DataFrame
        if interaction_matrix:
            interaction_df = pd.DataFrame(interaction_matrix).fillna(0).T
            return interaction_df
        else:
            return pd.DataFrame()
    
    def get_rule_examples(self, rule_conditions, df, narrative_col='NARRATIVE', n=5):
        """
        Get example narratives that match a specific rule
        """
        mask = pd.Series([True] * len(df), index=df.index)
        
        for feature, op, threshold in rule_conditions:
            if feature in df.columns:
                if op == '>':
                    mask &= (df[feature] > threshold)
                else:
                    mask &= (df[feature] <= threshold)
        
        matches = df[mask]
        
        if len(matches) > 0:
            return matches.head(n)[[narrative_col]].values.flatten()
        else:
            return []


def run_contextual_subgroup_discovery(non_issue_df, sar_df):
    """
    Run comprehensive contextual subgroup discovery
    """
    print("="*80)
    print("CONTEXTUAL SUBGROUP DISCOVERY FOR AML ALERTS")
    print("="*80)
    print("\nThis finds: Transaction patterns + Narrative context = High SAR risk")
    print()
    
    discoverer = NarrativeContextualSubgroupDiscovery(
        min_support=50,
        min_lift=3.0,
        max_depth=4
    )
    
    print("Analyzing contextual patterns...")
    rules_df, clf, X = discoverer.find_contextual_anomalies(
        sar_df,
        non_issue_df,
        narrative_col='NARRATIVE',
        top_n=30
    )
    
    print("\n" + "="*80)
    print("TOP CONTEXTUAL RULES (Transaction + Narrative Context)")
    print("="*80)
    print("\nThese rules show WHEN transaction patterns become suspicious\n")
    
    if len(rules_df) > 0:
        for idx, rule in rules_df.head(15).iterrows():
            print(f"\n{'='*70}")
            print(f"Rule #{idx + 1}:")
            print(f"{rule['rule']}")
            print(f"  → Found in {rule['sar_count']} SARs, {rule['non_issue_count']} non-issue")
            print(f"  → Precision: {rule['precision']*100:.1f}%")
            print(f"  → Lift: {rule['lift']:.1f}x (vs baseline)")
            
            # Get examples
            examples = discoverer.get_rule_examples(
                rule['conditions'],
                sar_df,
                n=2
            )
            
            if len(examples) > 0:
                print(f"\n  Example SAR narrative:")
                print(f"  \"{examples[0][:250]}...\"")
        
        # Analyze feature interactions
        print("\n" + "="*80)
        print("FEATURE INTERACTION ANALYSIS")
        print("="*80)
        print("\nWhich narrative contexts appear with which transaction features?\n")
        
        interactions = discoverer.analyze_feature_interactions(rules_df)
        
        if not interactions.empty:
            # Get top interactions
            top_interactions = []
            for trans_feat in interactions.index:
                for ctx_feat in interactions.columns:
                    count = interactions.loc[trans_feat, ctx_feat]
                    if count > 0:
                        top_interactions.append({
                            'transaction_feature': trans_feat,
                            'narrative_context': ctx_feat,
                            'frequency': count
                        })
            
            interaction_df = pd.DataFrame(top_interactions).sort_values('frequency', ascending=False)
            print(interaction_df.head(20))
        
        # Save results
        rules_df.to_csv('contextual_subgroup_rules.csv', index=False)
        if not interactions.empty:
            interactions.to_csv('feature_narrative_interactions.csv')
        
        print("\n" + "="*80)
        print(f"✓ Found {len(rules_df)} contextual rules")
        print(f"✓ Results saved to 'contextual_subgroup_rules.csv'")
        print(f"✓ Interaction matrix saved to 'feature_narrative_interactions.csv'")
        
        return rules_df, interactions
    else:
        print("\nNo high-precision rules found. Try adjusting parameters.")
        return pd.DataFrame(), pd.DataFrame()


# Example usage:
"""
# Load data with transaction features
non_issue_df = pd.read_csv('non_issue_alerts_with_features.csv')
sar_df = pd.read_csv('sar_alerts_with_features.csv')

# Run contextual subgroup discovery
rules, interactions = run_contextual_subgroup_discovery(non_issue_df, sar_df)

# Analyze specific scenarios
# Example: When do jewelry purchases trigger SARs?
if not rules.empty:
    jewelry_rules = rules[rules['rule'].str.contains('jewelry', case=False, na=False)]
    print(f"\n\nRules involving jewelry purchases:")
    print(jewelry_rules[['rule', 'precision', 'sar_count']])

# Example: What narrative contexts matter for electronics merchants?
if not interactions.empty:
    electronics_contexts = interactions.loc['electronics_merchant_flag'] if 'electronics_merchant_flag' in interactions.index else pd.Series()
    if not electronics_contexts.empty:
        print("\n\nNarrative contexts important for electronics merchants:")
        print(electronics_contexts.sort_values(ascending=False).head(10))
"""