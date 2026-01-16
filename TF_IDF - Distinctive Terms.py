import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.stats import fisher_exact, mannwhitneyu
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ImbalancedTFIDFAnalyzer:
    """
    TF-IDF analysis adapted for severe class imbalance (14,000 vs 680)
    
    Key changes:
    1. Use Fisher's exact test (better for small counts in minority class)
    2. Separate min_df for each class (680 SAR needs lower threshold)
    3. Weight by class size in discriminative ratio
    4. Focus on SAR-specific patterns (minority class priority)
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 3)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit_transform(self, non_issue_narratives, sar_narratives):
        """
        Fit with class-imbalance aware parameters
        For 680 SARs: min_df=3 means phrase appears in 0.44% of SARs
        For 14,000 non-issue: min_df=10 means phrase appears in 0.07% of non-issue
        """
        # Preprocess
        non_issue_clean = [self.preprocess_text(t) for t in non_issue_narratives]
        sar_clean = [self.preprocess_text(t) for t in sar_narratives]
        
        all_texts = non_issue_clean + sar_clean
        
        # Use lower min_df to capture SAR patterns
        # 3 docs = 0.44% of 680 SARs (reasonable threshold)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=3,  # Lowered for minority class
            max_df=0.85,
            stop_words='english',
            sublinear_tf=True  # Important: dampens term frequency
        )
        
        self.vectorizer.fit(all_texts)
        
        # Transform separately
        non_issue_tfidf = self.vectorizer.transform(non_issue_clean)
        sar_tfidf = self.vectorizer.transform(sar_clean)
        
        return non_issue_tfidf, sar_tfidf
    
    def extract_distinctive_terms(self, non_issue_tfidf, sar_tfidf, top_n=50):
        """
        Extract terms using Fisher's exact test for imbalanced data
        """
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        n_non_issue = non_issue_tfidf.shape[0]
        n_sar = sar_tfidf.shape[0]
        
        # Calculate mean TF-IDF
        non_issue_mean = np.array(non_issue_tfidf.mean(axis=0)).flatten()
        sar_mean = np.array(sar_tfidf.mean(axis=0)).flatten()
        
        # Calculate document frequency (how many docs contain the term)
        non_issue_df = np.array((non_issue_tfidf > 0).sum(axis=0)).flatten()
        sar_df = np.array((sar_tfidf > 0).sum(axis=0)).flatten()
        
        results = []
        
        for idx in range(len(feature_names)):
            # Contingency table for Fisher's exact test
            # [term present in SAR, term absent in SAR]
            # [term present in NI, term absent in NI]
            contingency = [
                [sar_df[idx], n_sar - sar_df[idx]],
                [non_issue_df[idx], n_non_issue - non_issue_df[idx]]
            ]
            
            # Fisher's exact test (better for imbalanced/small counts)
            try:
                odds_ratio, p_value = fisher_exact(contingency)
            except:
                p_value = 1.0
                odds_ratio = 1.0
            
            # Calculate effect size (weighted by class size)
            sar_freq_pct = sar_df[idx] / n_sar
            ni_freq_pct = non_issue_df[idx] / n_non_issue
            
            # Only include if statistically significant
            if p_value < 0.01:
                results.append({
                    'term': feature_names[idx],
                    'sar_doc_freq': int(sar_df[idx]),
                    'ni_doc_freq': int(non_issue_df[idx]),
                    'sar_freq_pct': sar_freq_pct,
                    'ni_freq_pct': ni_freq_pct,
                    'sar_mean_tfidf': sar_mean[idx],
                    'ni_mean_tfidf': non_issue_mean[idx],
                    'odds_ratio': odds_ratio,
                    'p_value': p_value
                })
        
        df = pd.DataFrame(results)
        
        # Separate red and green flags
        if len(df) > 0:
            # Red flags: higher in SAR
            red_flags = df[df['sar_freq_pct'] > df['ni_freq_pct']].copy()
            red_flags['red_flag_score'] = red_flags['odds_ratio'] * red_flags['sar_doc_freq']
            red_flags = red_flags.sort_values('red_flag_score', ascending=False).head(top_n)
            
            # Green flags: higher in non-issue
            green_flags = df[df['ni_freq_pct'] > df['sar_freq_pct']].copy()
            green_flags['green_flag_score'] = (1/green_flags['odds_ratio']) * green_flags['ni_doc_freq']
            green_flags = green_flags.sort_values('green_flag_score', ascending=False).head(top_n)
            
            return red_flags, green_flags
        
        return pd.DataFrame(), pd.DataFrame()
    
    def get_term_contexts(self, narratives, term, max_examples=5):
        """Get example contexts where a term appears"""
        examples = []
        term_pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
        
        for narrative in narratives:
            if pd.isna(narrative):
                continue
            text = str(narrative).lower()
            if term_pattern.search(text):
                sentences = re.split(r'[.!?]+', text)
                for sent in sentences:
                    if term.lower() in sent:
                        examples.append(sent.strip())
                        if len(examples) >= max_examples:
                            return examples
        return examples


# Example usage:
"""
non_issue_df = pd.read_csv('non_issue_alerts.csv')  # 14,000 rows
sar_df = pd.read_csv('sar_alerts.csv')  # 680 rows

analyzer = ImbalancedTFIDFAnalyzer(max_features=5000, ngram_range=(1, 3))

non_issue_tfidf, sar_tfidf = analyzer.fit_transform(
    non_issue_df['NARRATIVE'].values,
    sar_df['NARRATIVE'].values
)

red_flags, green_flags = analyzer.extract_distinctive_terms(
    non_issue_tfidf, sar_tfidf, top_n=50
)

print("TOP RED FLAGS (SAR indicators):")
print(red_flags[['term', 'sar_doc_freq', 'ni_doc_freq', 'odds_ratio']].head(20))

print("\nTOP GREEN FLAGS (Non-issue indicators):")
print(green_flags[['term', 'ni_doc_freq', 'sar_doc_freq', 'odds_ratio']].head(20))

# Save
red_flags.to_csv('red_flags_imbalanced.csv', index=False)
green_flags.to_csv('green_flags_imbalanced.csv', index=False)
"""
