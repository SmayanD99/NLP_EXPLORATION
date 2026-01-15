import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AMLTopicAnalyzer:
    """
    Discover latent topics in false positive and true positive narratives
    to understand common patterns and reasoning
    """
    
    def __init__(self, n_topics=10, max_features=3000, ngram_range=(1, 2), min_df=5):
        self.n_topics = n_topics
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        self.lda_model = None
        
    def preprocess_text(self, text):
        """Text preprocessing"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit_lda(self, narratives, label=''):
        """
        Fit LDA topic model on narratives
        """
        # Preprocess
        clean_texts = [self.preprocess_text(t) for t in narratives]
        
        # Vectorize
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=0.85,
            stop_words='english'
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(clean_texts)
        
        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=50,
            learning_method='online',
            n_jobs=-1
        )
        
        doc_topics = self.lda_model.fit_transform(doc_term_matrix)
        
        return doc_topics, doc_term_matrix
    
    def get_top_terms_per_topic(self, n_terms=15):
        """
        Extract top terms for each topic
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            top_weights = [topic[i] for i in top_indices]
            
            topics.append({
                'topic_id': topic_idx,
                'top_terms': top_terms,
                'weights': top_weights
            })
        
        return topics
    
    def get_topic_distribution(self, doc_topics):
        """
        Get distribution of documents across topics
        """
        # Get dominant topic for each document
        dominant_topics = np.argmax(doc_topics, axis=1)
        topic_counts = pd.Series(dominant_topics).value_counts().sort_index()
        
        # Get average topic weights
        avg_weights = doc_topics.mean(axis=0)
        
        return topic_counts, avg_weights
    
    def assign_topic_labels(self, topics):
        """
        Suggest interpretable labels based on top terms
        (Manual refinement recommended)
        """
        for topic in topics:
            terms = topic['top_terms'][:5]
            topic['suggested_label'] = ' + '.join(terms[:3])
        
        return topics
    
    def get_representative_documents(self, narratives, doc_topics, topic_id, top_n=5):
        """
        Get documents most representative of a specific topic
        """
        topic_weights = doc_topics[:, topic_id]
        top_doc_indices = topic_weights.argsort()[-top_n:][::-1]
        
        representatives = []
        for idx in top_doc_indices:
            representatives.append({
                'index': idx,
                'topic_weight': topic_weights[idx],
                'narrative': narratives[idx][:500]  # First 500 chars
            })
        
        return representatives
    
    def compare_topic_distributions(self, doc_topics_1, doc_topics_2, 
                                   label_1='Non-Issue', label_2='SAR'):
        """
        Compare topic distributions between two datasets
        """
        avg_1 = doc_topics_1.mean(axis=0)
        avg_2 = doc_topics_2.mean(axis=0)
        
        comparison = pd.DataFrame({
            'topic_id': range(self.n_topics),
            f'{label_1}_weight': avg_1,
            f'{label_2}_weight': avg_2,
            'difference': avg_2 - avg_1
        })
        
        comparison['dominant_in'] = comparison['difference'].apply(
            lambda x: label_2 if x > 0 else label_1
        )
        
        return comparison.sort_values('difference', ascending=False)


def run_comprehensive_topic_analysis(non_issue_df, sar_df):
    """
    Run complete topic modeling analysis on both datasets
    """
    results = {}
    
    # 1. Analyze Non-Issue Narratives (Green Flags)
    print("="*80)
    print("ANALYZING FALSE POSITIVE (NON-ISSUE) NARRATIVES")
    print("="*80)
    
    analyzer_ni = AMLTopicAnalyzer(n_topics=8, max_features=3000)
    doc_topics_ni, dtm_ni = analyzer_ni.fit_lda(
        non_issue_df['NARRATIVE'].values, 
        label='Non-Issue'
    )
    
    topics_ni = analyzer_ni.get_top_terms_per_topic(n_terms=15)
    topics_ni = analyzer_ni.assign_topic_labels(topics_ni)
    
    print("\nTOPICS IN FALSE POSITIVES (Green Flag Patterns):")
    for topic in topics_ni:
        print(f"\nTopic {topic['topic_id']}: {topic['suggested_label']}")
        print(f"Top terms: {', '.join(topic['top_terms'][:10])}")
    
    topic_dist_ni, avg_weights_ni = analyzer_ni.get_topic_distribution(doc_topics_ni)
    print(f"\nTopic distribution:\n{topic_dist_ni}")
    
    results['non_issue'] = {
        'analyzer': analyzer_ni,
        'topics': topics_ni,
        'doc_topics': doc_topics_ni,
        'distribution': topic_dist_ni
    }
    
    # 2. Analyze SAR Narratives (Red Flags)
    print("\n" + "="*80)
    print("ANALYZING TRUE POSITIVE (SAR) NARRATIVES")
    print("="*80)
    
    analyzer_sar = AMLTopicAnalyzer(n_topics=8, max_features=3000)
    doc_topics_sar, dtm_sar = analyzer_sar.fit_lda(
        sar_df['NARRATIVE'].values,
        label='SAR'
    )
    
    topics_sar = analyzer_sar.get_top_terms_per_topic(n_terms=15)
    topics_sar = analyzer_sar.assign_topic_labels(topics_sar)
    
    print("\nTOPICS IN TRUE POSITIVES (Red Flag Patterns):")
    for topic in topics_sar:
        print(f"\nTopic {topic['topic_id']}: {topic['suggested_label']}")
        print(f"Top terms: {', '.join(topic['top_terms'][:10])}")
    
    topic_dist_sar, avg_weights_sar = analyzer_sar.get_topic_distribution(doc_topics_sar)
    print(f"\nTopic distribution:\n{topic_dist_sar}")
    
    results['sar'] = {
        'analyzer': analyzer_sar,
        'topics': topics_sar,
        'doc_topics': doc_topics_sar,
        'distribution': topic_dist_sar
    }
    
    # 3. Get representative examples
    print("\n" + "="*80)
    print("REPRESENTATIVE EXAMPLES")
    print("="*80)
    
    print("\nExample from most common Non-Issue topic:")
    most_common_ni = topic_dist_ni.index[0]
    reps_ni = analyzer_ni.get_representative_documents(
        non_issue_df['NARRATIVE'].values,
        doc_topics_ni,
        most_common_ni,
        top_n=2
    )
    for i, rep in enumerate(reps_ni, 1):
        print(f"\n{i}. (weight: {rep['topic_weight']:.3f})")
        print(rep['narrative'])
    
    print("\n\nExample from most common SAR topic:")
    most_common_sar = topic_dist_sar.index[0]
    reps_sar = analyzer_sar.get_representative_documents(
        sar_df['NARRATIVE'].values,
        doc_topics_sar,
        most_common_sar,
        top_n=2
    )
    for i, rep in enumerate(reps_sar, 1):
        print(f"\n{i}. (weight: {rep['topic_weight']:.3f})")
        print(rep['narrative'])
    
    return results


# Example usage:
"""
# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run comprehensive analysis
results = run_comprehensive_topic_analysis(non_issue_df, sar_df)

# Access specific results
non_issue_topics = results['non_issue']['topics']
sar_topics = results['sar']['topics']

# Save topic information
ni_topics_df = pd.DataFrame([
    {
        'topic_id': t['topic_id'],
        'label': t['suggested_label'],
        'top_terms': ', '.join(t['top_terms'][:10])
    }
    for t in non_issue_topics
])
ni_topics_df.to_csv('non_issue_topics.csv', index=False)

sar_topics_df = pd.DataFrame([
    {
        'topic_id': t['topic_id'],
        'label': t['suggested_label'],
        'top_terms': ', '.join(t['top_terms'][:10])
    }
    for t in sar_topics
])
sar_topics_df.to_csv('sar_topics.csv', index=False)

# Get document-topic assignments for further analysis
doc_topic_assignments_ni = pd.DataFrame(results['non_issue']['doc_topics'])
doc_topic_assignments_ni['dominant_topic'] = doc_topic_assignments_ni.idxmax(axis=1)
non_issue_df['topic'] = doc_topic_assignments_ni['dominant_topic']
non_issue_df.to_csv('non_issue_with_topics.csv', index=False)
"""