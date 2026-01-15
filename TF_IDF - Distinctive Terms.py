import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Sample data structure (replace with your actual data)
# non_issue_df = pd.DataFrame({'ALERT_ID': [...], 'NARRATIVE': [...]})
# sar_df = pd.DataFrame({'SAR_ID': [...], 'NARRATIVE': [...]})

class AMLDistinctiveTermsAnalyzer:
    """
    Extract distinctive terms that characterize false positives (green flags) 
    and true positives (red flags) using TF-IDF
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 3), min_df=3):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit_transform(self, non_issue_narratives, sar_narratives):
        """
        Fit TF-IDF vectorizer on combined corpus and transform both datasets
        """
        # Preprocess
        non_issue_clean = [self.preprocess_text(t) for t in non_issue_narratives]
        sar_clean = [self.preprocess_text(t) for t in sar_narratives]
        
        # Combine for vocabulary building
        all_texts = non_issue_clean + sar_clean
        
        # Fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=0.85,
            stop_words='english'
        )
        
        self.vectorizer.fit(all_texts)
        
        # Transform separately
        non_issue_tfidf = self.vectorizer.transform(non_issue_clean)
        sar_tfidf = self.vectorizer.transform(sar_clean)
        
        return non_issue_tfidf, sar_tfidf
    
    def extract_distinctive_terms(self, non_issue_tfidf, sar_tfidf, top_n=50):
        """
        Extract terms that are distinctive to each class using mean TF-IDF scores
        and discriminative ratio
        """
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Calculate mean TF-IDF for each class
        non_issue_mean = np.array(non_issue_tfidf.mean(axis=0)).flatten()
        sar_mean = np.array(sar_tfidf.mean(axis=0)).flatten()
        
        # Calculate discriminative ratio (with smoothing)
        epsilon = 1e-6
        
        # Green flags: high in non-issue, low in SAR
        green_ratio = non_issue_mean / (sar_mean + epsilon)
        green_indices = np.argsort(green_ratio)[::-1][:top_n]
        
        green_flags = pd.DataFrame({
            'term': feature_names[green_indices],
            'non_issue_score': non_issue_mean[green_indices],
            'sar_score': sar_mean[green_indices],
            'ratio': green_ratio[green_indices]
        })
        
        # Red flags: high in SAR, low in non-issue
        red_ratio = sar_mean / (non_issue_mean + epsilon)
        red_indices = np.argsort(red_ratio)[::-1][:top_n]
        
        red_flags = pd.DataFrame({
            'term': feature_names[red_indices],
            'sar_score': sar_mean[red_indices],
            'non_issue_score': non_issue_mean[red_indices],
            'ratio': red_ratio[red_indices]
        })
        
        return green_flags, red_flags
    
    def get_term_contexts(self, narratives, term, max_examples=5):
        """
        Get example contexts where a term appears
        """
        examples = []
        term_pattern = re.compile(r'\b' + re.escape(term.lower()) + r'\b')
        
        for narrative in narratives:
            if pd.isna(narrative):
                continue
            text = str(narrative).lower()
            if term_pattern.search(text):
                # Extract sentence containing the term
                sentences = re.split(r'[.!?]+', text)
                for sent in sentences:
                    if term.lower() in sent:
                        examples.append(sent.strip())
                        if len(examples) >= max_examples:
                            return examples
        return examples


# Example usage:
"""
# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Initialize analyzer
analyzer = AMLDistinctiveTermsAnalyzer(
    max_features=5000,
    ngram_range=(1, 3),  # unigrams, bigrams, trigrams
    min_df=3  # term must appear in at least 3 documents
)

# Fit and transform
non_issue_tfidf, sar_tfidf = analyzer.fit_transform(
    non_issue_df['NARRATIVE'].values,
    sar_df['NARRATIVE'].values
)

# Extract distinctive terms
green_flags, red_flags = analyzer.extract_distinctive_terms(
    non_issue_tfidf, 
    sar_tfidf, 
    top_n=50
)

print("TOP GREEN FLAGS (False Positive Indicators):")
print(green_flags.head(20))
print("\n" + "="*80 + "\n")

print("TOP RED FLAGS (True Positive Indicators):")
print(red_flags.head(20))

# Get contexts for specific terms
print("\nExample contexts for top red flag:")
top_red_flag = red_flags.iloc[0]['term']
contexts = analyzer.get_term_contexts(sar_df['NARRATIVE'].values, top_red_flag)
print(f"\nTerm: '{top_red_flag}'")
for i, ctx in enumerate(contexts[:3], 1):
    print(f"{i}. {ctx}")

# Save results
green_flags.to_csv('green_flags_tfidf.csv', index=False)
red_flags.to_csv('red_flags_tfidf.csv', index=False)
"""