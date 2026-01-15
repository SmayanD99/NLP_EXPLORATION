import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class DiscriminativeEmbeddingMiner:
    """
    Uses embedding space geometry to discover discriminative phrases automatically.
    
    Method: 
    1. Create TF-IDF embeddings for all narratives
    2. Find the direction in embedding space that maximally separates SAR from non-issue
    3. Project all phrases onto this discriminative direction
    4. Phrases with extreme projections are your red/green flags
    
    This discovers patterns like:
    - Phrases that "pull" narratives toward SAR space
    - Phrases that "pull" narratives toward non-issue space
    
    No prior knowledge needed - the algorithm finds what matters.
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 4), min_df=5):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None
        self.discriminative_direction = None
        self.phrase_scores = None
        
    def preprocess_text(self, text):
        """Minimal preprocessing to preserve meaning"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Keep structure words
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit_discriminative_space(self, non_issue_narratives, sar_narratives):
        """
        Learn the discriminative direction in embedding space
        """
        # Preprocess
        non_issue_clean = [self.preprocess_text(t) for t in non_issue_narratives]
        sar_clean = [self.preprocess_text(t) for t in sar_narratives]
        
        all_texts = non_issue_clean + sar_clean
        
        # Create TF-IDF embeddings
        print("Creating TF-IDF embeddings...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=0.85,
            sublinear_tf=True,
            norm='l2'
        )
        
        # Fit on all data
        all_vectors = self.vectorizer.fit_transform(all_texts)
        
        # Separate classes
        n_non_issue = len(non_issue_clean)
        non_issue_vectors = all_vectors[:n_non_issue]
        sar_vectors = all_vectors[n_non_issue:]
        
        # Calculate class centroids
        non_issue_centroid = np.array(non_issue_vectors.mean(axis=0)).flatten()
        sar_centroid = np.array(sar_vectors.mean(axis=0)).flatten()
        
        # The discriminative direction is the vector connecting centroids
        self.discriminative_direction = sar_centroid - non_issue_centroid
        self.discriminative_direction = self.discriminative_direction / (
            np.linalg.norm(self.discriminative_direction) + 1e-10
        )
        
        print(f"Learned discriminative direction in {len(self.discriminative_direction)}-dimensional space")
        
        return non_issue_vectors, sar_vectors
    
    def score_phrases(self, non_issue_vectors, sar_vectors):
        """
        Score each phrase by its projection onto the discriminative direction
        """
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Get the TF-IDF matrix columns (one per phrase)
        # Each column is the distribution of that phrase across documents
        phrase_vectors = self.vectorizer.transform([''] * len(feature_names))
        
        # For each phrase, calculate its discriminative score
        scores = []
        
        for idx, phrase in enumerate(feature_names):
            # Get phrase vector from vocabulary
            phrase_vec = np.zeros(len(feature_names))
            phrase_vec[idx] = 1.0
            
            # Project onto discriminative direction
            projection = np.dot(phrase_vec, self.discriminative_direction)
            
            # Calculate how often this phrase appears in each class
            non_issue_freq = np.array(non_issue_vectors[:, idx].sum())
            sar_freq = np.array(sar_vectors[:, idx].sum())
            
            # Calculate mean TF-IDF when phrase is present
            non_issue_mean = np.array(non_issue_vectors[:, idx].mean())
            sar_mean = np.array(sar_vectors[:, idx].mean())
            
            # Discriminative power: difference in means
            discriminative_power = sar_mean - non_issue_mean
            
            # Statistical significance using Mann-Whitney U test
            non_issue_values = np.array(non_issue_vectors[:, idx].toarray()).flatten()
            sar_values = np.array(sar_vectors[:, idx].toarray()).flatten()
            
            if non_issue_values.sum() > 0 and sar_values.sum() > 0:
                stat, p_value = mannwhitneyu(
                    sar_values, 
                    non_issue_values, 
                    alternative='two-sided'
                )
            else:
                p_value = 1.0
            
            scores.append({
                'phrase': phrase,
                'projection': projection,
                'discriminative_power': discriminative_power,
                'non_issue_freq': float(non_issue_freq),
                'sar_freq': float(sar_freq),
                'non_issue_mean_tfidf': float(non_issue_mean),
                'sar_mean_tfidf': float(sar_mean),
                'p_value': p_value
            })
        
        self.phrase_scores = pd.DataFrame(scores)
        
        # Filter for statistical significance
        self.phrase_scores = self.phrase_scores[self.phrase_scores['p_value'] < 0.01]
        
        return self.phrase_scores
    
    def get_red_flags(self, top_n=100, min_freq=10):
        """
        Get phrases that strongly indicate SAR (positive discriminative power)
        """
        if self.phrase_scores is None:
            raise ValueError("Must call score_phrases first")
        
        red_flags = self.phrase_scores[
            (self.phrase_scores['discriminative_power'] > 0) &
            (self.phrase_scores['sar_freq'] >= min_freq)
        ].copy()
        
        # Score combines discriminative power with frequency
        red_flags['red_flag_score'] = (
            red_flags['discriminative_power'] * 
            np.log1p(red_flags['sar_freq'])
        )
        
        red_flags = red_flags.sort_values('red_flag_score', ascending=False).head(top_n)
        
        return red_flags
    
    def get_green_flags(self, top_n=100, min_freq=10):
        """
        Get phrases that strongly indicate non-issue (negative discriminative power)
        """
        if self.phrase_scores is None:
            raise ValueError("Must call score_phrases first")
        
        green_flags = self.phrase_scores[
            (self.phrase_scores['discriminative_power'] < 0) &
            (self.phrase_scores['non_issue_freq'] >= min_freq)
        ].copy()
        
        # Score combines discriminative power with frequency
        green_flags['green_flag_score'] = (
            abs(green_flags['discriminative_power']) * 
            np.log1p(green_flags['non_issue_freq'])
        )
        
        green_flags = green_flags.sort_values('green_flag_score', ascending=False).head(top_n)
        
        return green_flags
    
    def find_phrase_clusters(self, flags_df, n_clusters=10):
        """
        Group similar phrases together to find thematic patterns
        """
        phrases = flags_df['phrase'].values
        
        # Get phrase embeddings
        phrase_vectors = self.vectorizer.transform(phrases)
        
        # Simple clustering using cosine similarity
        similarity_matrix = cosine_similarity(phrase_vectors)
        
        # For each phrase, find its nearest neighbors
        clusters = []
        used_phrases = set()
        
        for i, phrase in enumerate(phrases):
            if phrase in used_phrases:
                continue
            
            # Find similar phrases
            similarities = similarity_matrix[i]
            similar_indices = np.argsort(similarities)[::-1][1:6]  # Top 5 similar
            
            cluster_phrases = [phrase]
            for idx in similar_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    cluster_phrases.append(phrases[idx])
                    used_phrases.add(phrases[idx])
            
            if len(cluster_phrases) > 1:
                clusters.append({
                    'cluster_id': len(clusters),
                    'representative': phrase,
                    'phrases': cluster_phrases,
                    'avg_score': flags_df.iloc[[i] + list(similar_indices)].iloc[:len(cluster_phrases)].iloc[:, -1].mean()
                })
                used_phrases.add(phrase)
            
            if len(clusters) >= n_clusters:
                break
        
        return clusters
    
    def get_phrase_contexts(self, narratives, phrase, max_examples=5):
        """
        Get example contexts where phrase appears
        """
        examples = []
        phrase_pattern = re.compile(r'\b' + re.escape(phrase.lower()) + r'\b')
        
        for narrative in narratives:
            if pd.isna(narrative):
                continue
            text = str(narrative).lower()
            if phrase_pattern.search(text):
                # Get surrounding context (100 chars before and after)
                match = phrase_pattern.search(text)
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()
                examples.append(context)
                if len(examples) >= max_examples:
                    break
        
        return examples


def run_discriminative_embedding_discovery(non_issue_df, sar_df):
    """
    Run automatic discriminative phrase discovery
    """
    print("="*80)
    print("AUTOMATIC DISCRIMINATIVE PHRASE DISCOVERY")
    print("="*80)
    print("\nUsing embedding space geometry to discover red/green flags")
    print("No prior knowledge or assumptions required\n")
    
    miner = DiscriminativeEmbeddingMiner(
        max_features=10000,
        ngram_range=(1, 4),  # Unigrams to 4-grams
        min_df=5
    )
    
    # Fit discriminative space
    print("1. Learning discriminative embedding space...")
    non_issue_vectors, sar_vectors = miner.fit_discriminative_space(
        non_issue_df['NARRATIVE'].values,
        sar_df['NARRATIVE'].values
    )
    
    # Score all phrases
    print("2. Scoring all phrases by discriminative power...")
    phrase_scores = miner.score_phrases(non_issue_vectors, sar_vectors)
    
    print(f"   Found {len(phrase_scores)} statistically significant phrases")
    
    # Extract red flags
    print("\n3. Extracting red flags (SAR indicators)...")
    red_flags = miner.get_red_flags(top_n=100, min_freq=10)
    
    print("\n" + "="*80)
    print("TOP RED FLAGS (Automatically Discovered)")
    print("="*80)
    print("\nPhrases that strongly indicate SAR filing:\n")
    
    for idx, row in red_flags.head(30).iterrows():
        print(f"{row['phrase']:40s} | "
              f"SAR freq: {row['sar_freq']:>5.0f} | "
              f"Non-issue freq: {row['non_issue_freq']:>5.0f} | "
              f"Score: {row['red_flag_score']:.3f}")
    
    # Find phrase clusters for red flags
    print("\n4. Clustering red flags into thematic groups...")
    red_clusters = miner.find_phrase_clusters(red_flags, n_clusters=8)
    
    print("\n" + "="*80)
    print("RED FLAG THEMATIC CLUSTERS")
    print("="*80)
    
    for cluster in red_clusters:
        print(f"\nCluster {cluster['cluster_id'] + 1}: {cluster['representative']}")
        print(f"  Related phrases: {', '.join(cluster['phrases'][1:4])}")
        print(f"  Avg score: {cluster['avg_score']:.3f}")
    
    # Extract green flags
    print("\n5. Extracting green flags (non-issue indicators)...")
    green_flags = miner.get_green_flags(top_n=100, min_freq=10)
    
    print("\n" + "="*80)
    print("TOP GREEN FLAGS (Automatically Discovered)")
    print("="*80)
    print("\nPhrases that strongly indicate non-issue disposition:\n")
    
    for idx, row in green_flags.head(30).iterrows():
        print(f"{row['phrase']:40s} | "
              f"Non-issue freq: {row['non_issue_freq']:>5.0f} | "
              f"SAR freq: {row['sar_freq']:>5.0f} | "
              f"Score: {row['green_flag_score']:.3f}")
    
    # Find phrase clusters for green flags
    print("\n6. Clustering green flags into thematic groups...")
    green_clusters = miner.find_phrase_clusters(green_flags, n_clusters=8)
    
    print("\n" + "="*80)
    print("GREEN FLAG THEMATIC CLUSTERS")
    print("="*80)
    
    for cluster in green_clusters:
        print(f"\nCluster {cluster['cluster_id'] + 1}: {cluster['representative']}")
        print(f"  Related phrases: {', '.join(cluster['phrases'][1:4])}")
        print(f"  Avg score: {cluster['avg_score']:.3f}")
    
    # Get examples for top flags
    print("\n" + "="*80)
    print("EXAMPLE CONTEXTS")
    print("="*80)
    
    if len(red_flags) > 0:
        top_red = red_flags.iloc[0]['phrase']
        print(f"\nTop red flag: '{top_red}'")
        print("Example SAR contexts:")
        contexts = miner.get_phrase_contexts(sar_df['NARRATIVE'].values, top_red, max_examples=3)
        for i, ctx in enumerate(contexts, 1):
            print(f"\n{i}. ...{ctx}...")
    
    if len(green_flags) > 0:
        top_green = green_flags.iloc[0]['phrase']
        print(f"\n\nTop green flag: '{top_green}'")
        print("Example non-issue contexts:")
        contexts = miner.get_phrase_contexts(non_issue_df['NARRATIVE'].values, top_green, max_examples=3)
        for i, ctx in enumerate(contexts, 1):
            print(f"\n{i}. ...{ctx}...")
    
    # Save results
    red_flags.to_csv('auto_discovered_red_flags.csv', index=False)
    green_flags.to_csv('auto_discovered_green_flags.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ Results saved to:")
    print("  - auto_discovered_red_flags.csv")
    print("  - auto_discovered_green_flags.csv")
    print(f"✓ Discovered {len(red_flags)} red flags and {len(green_flags)} green flags")
    
    return {
        'red_flags': red_flags,
        'green_flags': green_flags,
        'red_clusters': red_clusters,
        'green_clusters': green_clusters,
        'miner': miner
    }


# Example usage:
"""
# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run automatic discovery
results = run_discriminative_embedding_discovery(non_issue_df, sar_df)

# Analyze results
print(f"\nTop 5 red flag phrases:")
print(results['red_flags'].head(5)[['phrase', 'red_flag_score']])

print(f"\nTop 5 green flag phrases:")
print(results['green_flags'].head(5)[['phrase', 'green_flag_score']])

# Get examples for any specific phrase
miner = results['miner']
specific_phrase = results['red_flags'].iloc[0]['phrase']
examples = miner.get_phrase_contexts(
    sar_df['NARRATIVE'].values, 
    specific_phrase,
    max_examples=10
)
print(f"\nAll contexts for '{specific_phrase}':")
for ex in examples:
    print(f"  - {ex}")
"""
