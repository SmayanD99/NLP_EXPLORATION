import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import chi2_contingency
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer

class SentenceLevelSemanticMiner:
    """
    Advanced technique: Instead of analyzing entire narratives, break them into sentences
    and find SENTENCE-LEVEL patterns that distinguish SAR from non-issue.
    
    This discovers:
    1. Specific sentence types that appear more in SARs (semantic fingerprints)
    2. Sentence sequences that indicate investigation quality
    3. Key reasoning sentences that flip decisions
    4. Hidden linguistic markers at granular level
    
    Why this is powerful:
    - A SAR narrative might have 10 sentences, only 2-3 are truly discriminative
    - This finds WHICH sentences matter, not just that the narrative is a SAR
    - Discovers micro-patterns that document-level analysis misses
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=0.75):
        print(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.sentence_clusters = None
        
    def extract_sentences(self, narratives, labels):
        """
        Extract all sentences with their source labels
        """
        sentences = []
        sentence_labels = []
        sentence_sources = []
        
        for idx, narrative in enumerate(narratives):
            if pd.isna(narrative):
                continue
            
            text = str(narrative)
            
            # Split into sentences (simple approach)
            sents = re.split(r'[.!?]+', text)
            
            for sent in sents:
                sent = sent.strip()
                if len(sent) > 20:  # Skip very short sentences
                    sentences.append(sent)
                    sentence_labels.append(labels[idx])
                    sentence_sources.append(idx)
        
        return sentences, np.array(sentence_labels), np.array(sentence_sources)
    
    def cluster_sentences(self, sentences, sentence_labels, n_clusters=100, method='kmeans'):
        """
        Cluster sentences by semantic similarity
        This groups sentences that mean similar things
        """
        print(f"Generating embeddings for {len(sentences)} sentences...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embeddings = normalize(embeddings, norm='l2')
        
        # Cluster
        print(f"Clustering sentences using {method}...")
        
        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            cluster_centers = clusterer.cluster_centers_
            
        elif method == 'dbscan':
            clusterer = DBSCAN(
                eps=0.3,
                min_samples=5,
                metric='cosine'
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Calculate cluster centers manually for DBSCAN
            unique_clusters = np.unique(cluster_labels)
            unique_clusters = unique_clusters[unique_clusters >= 0]
            cluster_centers = []
            
            for cluster_id in unique_clusters:
                mask = cluster_labels == cluster_id
                center = embeddings[mask].mean(axis=0)
                cluster_centers.append(center)
            
            cluster_centers = np.array(cluster_centers) if cluster_centers else None
        
        # Analyze each cluster
        clusters = []
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_sents = np.array(sentences)[mask]
            cluster_sent_labels = sentence_labels[mask]
            
            # Calculate statistics
            n_sar = cluster_sent_labels.sum()
            n_non_issue = len(cluster_sent_labels) - n_sar
            sar_proportion = n_sar / len(cluster_sent_labels) if len(cluster_sent_labels) > 0 else 0
            
            # Get representative sentence (closest to centroid)
            if cluster_centers is not None:
                cluster_embeddings = embeddings[mask]
                if method == 'kmeans':
                    center = cluster_centers[cluster_id]
                else:
                    center_idx = cluster_id if cluster_id < len(cluster_centers) else 0
                    center = cluster_centers[center_idx]
                
                distances = cosine_similarity(cluster_embeddings, center.reshape(1, -1)).flatten()
                representative_idx = np.argmax(distances)
                representative_sent = cluster_sents[representative_idx]
            else:
                representative_sent = cluster_sents[0] if len(cluster_sents) > 0 else ""
            
            # Statistical significance
            total_sar = sentence_labels.sum()
            total_non_issue = len(sentence_labels) - total_sar
            
            contingency = np.array([
                [n_sar, n_non_issue],
                [total_sar - n_sar, total_non_issue - n_non_issue]
            ])
            
            if contingency.min() >= 5:
                chi2, p_value, _, _ = chi2_contingency(contingency)
            else:
                p_value = 1.0
            
            clusters.append({
                'cluster_id': int(cluster_id),
                'size': int(mask.sum()),
                'n_sar': int(n_sar),
                'n_non_issue': int(n_non_issue),
                'sar_proportion': sar_proportion,
                'representative_sentence': representative_sent,
                'p_value': p_value,
                'sentences': cluster_sents.tolist()
            })
        
        self.sentence_clusters = pd.DataFrame(clusters)
        self.sentence_embeddings = embeddings
        self.cluster_labels = cluster_labels
        
        return self.sentence_clusters, embeddings, cluster_labels
    
    def find_discriminative_sentence_types(self, min_size=10, min_purity=0.7):
        """
        Find sentence clusters that strongly indicate SAR or non-issue
        """
        if self.sentence_clusters is None:
            raise ValueError("Must run cluster_sentences first")
        
        # Filter for significant clusters
        significant = self.sentence_clusters[
            (self.sentence_clusters['size'] >= min_size) &
            (self.sentence_clusters['p_value'] < 0.01)
        ]
        
        # SAR indicators
        sar_indicators = significant[
            significant['sar_proportion'] > min_purity
        ].sort_values('sar_proportion', ascending=False)
        
        # Non-issue indicators
        non_issue_indicators = significant[
            significant['sar_proportion'] < (1 - min_purity)
        ].sort_values('sar_proportion')
        
        return sar_indicators, non_issue_indicators
    
    def discover_sentence_sequences(self, narratives, labels, window_size=3):
        """
        Find common sequences of sentence types that appear in narratives
        
        This discovers patterns like:
        - SAR narratives often have: [observation] -> [verification attempt] -> [failure] -> [conclusion]
        - Non-issue: [observation] -> [verification success] -> [explanation] -> [closure]
        """
        sequences = []
        
        for narrative, label in zip(narratives, labels):
            if pd.isna(narrative):
                continue
            
            text = str(narrative)
            sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
            
            if len(sents) < window_size:
                continue
            
            # Generate embeddings for this narrative's sentences
            sent_embeddings = self.model.encode(sents, show_progress_bar=False)
            
            # Map each sentence to its nearest cluster
            if self.sentence_clusters is not None and hasattr(self, 'cluster_labels'):
                sent_cluster_ids = []
                
                for sent_emb in sent_embeddings:
                    # Find nearest cluster center
                    similarities = cosine_similarity(
                        sent_emb.reshape(1, -1),
                        self.sentence_embeddings
                    ).flatten()
                    
                    # Get cluster of most similar sentence
                    nearest_idx = np.argmax(similarities)
                    cluster_id = self.cluster_labels[nearest_idx]
                    sent_cluster_ids.append(cluster_id)
                
                # Extract sequences
                for i in range(len(sent_cluster_ids) - window_size + 1):
                    sequence = tuple(sent_cluster_ids[i:i+window_size])
                    sequences.append({
                        'sequence': sequence,
                        'label': label
                    })
        
        # Analyze sequences
        sar_sequences = Counter([s['sequence'] for s in sequences if s['label'] == 1])
        non_issue_sequences = Counter([s['sequence'] for s in sequences if s['label'] == 0])
        
        # Find discriminative sequences
        discriminative = []
        
        all_sequences = set(sar_sequences.keys()) | set(non_issue_sequences.keys())
        
        for seq in all_sequences:
            sar_count = sar_sequences.get(seq, 0)
            ni_count = non_issue_sequences.get(seq, 0)
            total = sar_count + ni_count
            
            if total >= 5:
                sar_prop = sar_count / total
                
                discriminative.append({
                    'sequence': seq,
                    'sar_count': sar_count,
                    'non_issue_count': ni_count,
                    'sar_proportion': sar_prop
                })
        
        return pd.DataFrame(discriminative).sort_values('sar_proportion', ascending=False)
    
    def find_key_sentences(self, narratives, labels, top_n=50):
        """
        Find individual sentences that are most predictive of SAR vs non-issue
        These are the "smoking gun" sentences
        """
        sentences, sentence_labels, sentence_sources = self.extract_sentences(narratives, labels)
        
        print(f"Analyzing {len(sentences)} sentences for discriminative power...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=True
        )
        
        embeddings = normalize(embeddings, norm='l2')
        
        # Calculate class centroids
        sar_mask = sentence_labels == 1
        non_issue_mask = sentence_labels == 0
        
        sar_centroid = embeddings[sar_mask].mean(axis=0)
        non_issue_centroid = embeddings[non_issue_mask].mean(axis=0)
        
        # Discriminative direction
        disc_direction = sar_centroid - non_issue_centroid
        disc_direction = disc_direction / (np.linalg.norm(disc_direction) + 1e-10)
        
        # Project all sentences
        projections = np.dot(embeddings, disc_direction)
        
        # Find sentences with extreme projections
        sar_sentences_indices = np.where(sentence_labels == 1)[0]
        ni_sentences_indices = np.where(sentence_labels == 0)[0]
        
        # Top SAR sentences (high projection, from SAR narratives)
        sar_projections = projections[sar_sentences_indices]
        top_sar_indices = sar_sentences_indices[np.argsort(sar_projections)[-top_n:]]
        
        # Top non-issue sentences (low projection, from non-issue narratives)
        ni_projections = projections[ni_sentences_indices]
        top_ni_indices = ni_sentences_indices[np.argsort(ni_projections)[:top_n]]
        
        key_sentences = {
            'sar': [
                {
                    'sentence': sentences[idx],
                    'projection': projections[idx],
                    'source_narrative_idx': sentence_sources[idx]
                }
                for idx in reversed(top_sar_indices)
            ],
            'non_issue': [
                {
                    'sentence': sentences[idx],
                    'projection': projections[idx],
                    'source_narrative_idx': sentence_sources[idx]
                }
                for idx in top_ni_indices
            ]
        }
        
        return key_sentences


def run_sentence_level_mining(non_issue_df, sar_df, narrative_col='NARRATIVE'):
    """
    Run comprehensive sentence-level semantic mining
    """
    print("="*80)
    print("SENTENCE-LEVEL SEMANTIC PATTERN MINING")
    print("="*80)
    print("\nAnalyzing narratives at the sentence level to find micro-patterns\n")
    
    miner = SentenceLevelSemanticMiner(
        model_name='all-MiniLM-L6-v2',
        similarity_threshold=0.75
    )
    
    # Prepare data
    all_narratives = np.concatenate([
        non_issue_df[narrative_col].values,
        sar_df[narrative_col].values
    ])
    
    labels = np.concatenate([
        np.zeros(len(non_issue_df)),
        np.ones(len(sar_df))
    ])
    
    # Extract sentences
    print("1. Extracting sentences from all narratives...")
    sentences, sentence_labels, sentence_sources = miner.extract_sentences(all_narratives, labels)
    print(f"   Extracted {len(sentences)} sentences")
    
    # Cluster sentences
    print("\n2. Clustering sentences by semantic similarity...")
    clusters_df, embeddings, cluster_labels = miner.cluster_sentences(
        sentences,
        sentence_labels,
        n_clusters=80,
        method='kmeans'
    )
    
    print(f"   Found {len(clusters_df)} sentence clusters")
    
    # Find discriminative sentence types
    print("\n3. Finding discriminative sentence types...")
    sar_indicators, ni_indicators = miner.find_discriminative_sentence_types(
        min_size=15,
        min_purity=0.7
    )
    
    print("\n" + "="*80)
    print("SAR INDICATOR SENTENCE TYPES")
    print("="*80)
    print(f"\nFound {len(sar_indicators)} sentence types that strongly indicate SAR:\n")
    
    for _, cluster in sar_indicators.head(15).iterrows():
        print(f"\nCluster {cluster['cluster_id']} "
              f"({cluster['size']} sentences, {cluster['sar_proportion']*100:.0f}% from SARs)")
        print(f"  Representative: \"{cluster['representative_sentence']}\"")
        print(f"  Distribution: {cluster['n_sar']} SAR, {cluster['n_non_issue']} non-issue")
    
    print("\n" + "="*80)
    print("NON-ISSUE INDICATOR SENTENCE TYPES")
    print("="*80)
    print(f"\nFound {len(ni_indicators)} sentence types that strongly indicate non-issue:\n")
    
    for _, cluster in ni_indicators.head(15).iterrows():
        print(f"\nCluster {cluster['cluster_id']} "
              f"({cluster['size']} sentences, {cluster['sar_proportion']*100:.0f}% from SARs)")
        print(f"  Representative: \"{cluster['representative_sentence']}\"")
        print(f"  Distribution: {cluster['n_sar']} SAR, {cluster['n_non_issue']} non-issue")
    
    # Find key individual sentences
    print("\n4. Finding most discriminative individual sentences...")
    key_sentences = miner.find_key_sentences(all_narratives, labels, top_n=20)
    
    print("\n" + "="*80)
    print("KEY SAR SENTENCES (Most Prototypical)")
    print("="*80)
    
    for i, sent_data in enumerate(key_sentences['sar'][:10], 1):
        print(f"\n{i}. (score: {sent_data['projection']:.3f})")
        print(f"   \"{sent_data['sentence']}\"")
    
    print("\n" + "="*80)
    print("KEY NON-ISSUE SENTENCES (Most Prototypical)")
    print("="*80)
    
    for i, sent_data in enumerate(key_sentences['non_issue'][:10], 1):
        print(f"\n{i}. (score: {sent_data['projection']:.3f})")
        print(f"   \"{sent_data['sentence']}\"")
    
    # Discover sentence sequences
    print("\n5. Discovering sentence sequences...")
    sequences = miner.discover_sentence_sequences(all_narratives, labels, window_size=3)
    
    if len(sequences) > 0:
        print("\n" + "="*80)
        print("DISCRIMINATIVE SENTENCE SEQUENCES")
        print("="*80)
        
        sar_sequences = sequences[sequences['sar_proportion'] > 0.7].head(10)
        if len(sar_sequences) > 0:
            print("\nSequences common in SAR narratives:")
            for _, seq in sar_sequences.iterrows():
                print(f"\n  Sequence: {seq['sequence']}")
                print(f"  SAR: {seq['sar_count']}, Non-issue: {seq['non_issue_count']}")
                print(f"  SAR proportion: {seq['sar_proportion']*100:.0f}%")
        
        ni_sequences = sequences[sequences['sar_proportion'] < 0.3].head(10)
        if len(ni_sequences) > 0:
            print("\nSequences common in non-issue narratives:")
            for _, seq in ni_sequences.iterrows():
                print(f"\n  Sequence: {seq['sequence']}")
                print(f"  SAR: {seq['sar_count']}, Non-issue: {seq['non_issue_count']}")
                print(f"  SAR proportion: {seq['sar_proportion']*100:.0f}%")
    
    # Save results
    print("\n6. Saving results...")
    
    sar_indicators.to_csv('sar_sentence_types.csv', index=False)
    ni_indicators.to_csv('non_issue_sentence_types.csv', index=False)
    
    # Save key sentences
    pd.DataFrame(key_sentences['sar']).to_csv('key_sar_sentences.csv', index=False)
    pd.DataFrame(key_sentences['non_issue']).to_csv('key_non_issue_sentences.csv', index=False)
    
    if len(sequences) > 0:
        sequences.to_csv('sentence_sequences.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ Results saved:")
    print("  - sar_sentence_types.csv")
    print("  - non_issue_sentence_types.csv")
    print("  - key_sar_sentences.csv")
    print("  - key_non_issue_sentences.csv")
    if len(sequences) > 0:
        print("  - sentence_sequences.csv")
    
    return {
        'miner': miner,
        'clusters': clusters_df,
        'sar_indicators': sar_indicators,
        'non_issue_indicators': ni_indicators,
        'key_sentences': key_sentences,
        'sequences': sequences if len(sequences) > 0 else None
    }


# Example usage:
"""
# Install required library:
# pip install sentence-transformers

import pandas as pd

# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run sentence-level mining
results = run_sentence_level_mining(non_issue_df, sar_df)

# Examine SAR indicator sentence types
print("\nTop SAR sentence types:")
print(results['sar_indicators'].head(10))

# Look at specific clusters
cluster_5_sentences = results['clusters'][
    results['clusters']['cluster_id'] == 5
]['sentences'].iloc[0]

print(f"\nAll sentences in cluster 5:")
for sent in cluster_5_sentences[:10]:
    print(f"  - {sent}")

# Check key individual sentences
print("\nMost discriminative SAR sentences:")
for sent in results['key_sentences']['sar'][:5]:
    print(f"  {sent['sentence']}")
"""