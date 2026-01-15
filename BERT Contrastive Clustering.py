import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Note: This requires sentence-transformers library
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

class BERTContrastiveClusterAnalyzer:
    """
    Uses BERT embeddings to discover semantic patterns that distinguish SAR from non-issue.
    
    Advanced techniques:
    1. Sentence-level embeddings capture MEANING not just keywords
    2. Contrastive learning finds what makes SARs semantically different
    3. Hierarchical clustering discovers narrative archetypes
    4. Semantic centroids identify "prototypical" SAR vs non-issue narratives
    
    This finds patterns like:
    - "Verification failure narratives" (expressed in 20 different ways)
    - "Evasive customer behavior narratives" (semantic concept, not keywords)
    - "Legitimate business justification narratives"
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', batch_size=32):
        """
        Initialize with a sentence transformer model.
        
        Models to consider:
        - 'all-MiniLM-L6-v2': Fast, good quality (default)
        - 'all-mpnet-base-v2': Higher quality, slower
        - 'paraphrase-MiniLM-L6-v2': Good for paraphrase detection
        """
        print(f"Loading BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embeddings = None
        self.labels = None
        
    def generate_embeddings(self, narratives, show_progress=True):
        """
        Generate BERT embeddings for all narratives
        """
        print(f"Generating embeddings for {len(narratives)} narratives...")
        
        # Clean narratives
        clean_narratives = [
            str(text) if not pd.isna(text) else "" 
            for text in narratives
        ]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            clean_narratives,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2')
        
        return embeddings
    
    def fit(self, non_issue_narratives, sar_narratives):
        """
        Generate embeddings for both classes
        """
        # Generate embeddings
        non_issue_embeddings = self.generate_embeddings(non_issue_narratives)
        sar_embeddings = self.generate_embeddings(sar_narratives)
        
        # Combine
        self.embeddings = np.vstack([non_issue_embeddings, sar_embeddings])
        self.labels = np.concatenate([
            np.zeros(len(non_issue_embeddings)),
            np.ones(len(sar_embeddings))
        ])
        
        self.non_issue_embeddings = non_issue_embeddings
        self.sar_embeddings = sar_embeddings
        
        return self.embeddings, self.labels
    
    def find_discriminative_direction(self):
        """
        Find the direction in embedding space that maximally separates classes
        Similar to Linear Discriminant Analysis (LDA)
        """
        # Calculate class centroids
        non_issue_centroid = self.non_issue_embeddings.mean(axis=0)
        sar_centroid = self.sar_embeddings.mean(axis=0)
        
        # Discriminative direction
        discriminative_vector = sar_centroid - non_issue_centroid
        discriminative_vector = discriminative_vector / (np.linalg.norm(discriminative_vector) + 1e-10)
        
        # Project all embeddings onto this direction
        non_issue_projections = np.dot(self.non_issue_embeddings, discriminative_vector)
        sar_projections = np.dot(self.sar_embeddings, discriminative_vector)
        
        return discriminative_vector, non_issue_projections, sar_projections
    
    def discover_narrative_archetypes(self, n_clusters=15, method='hierarchical'):
        """
        Discover narrative archetypes using clustering on embeddings
        These are semantic patterns that repeat across narratives
        """
        print(f"\nDiscovering narrative archetypes using {method} clustering...")
        
        if method == 'hdbscan':
            # HDBSCAN automatically finds number of clusters
            clusterer = HDBSCAN(
                min_cluster_size=30,
                min_samples=10,
                metric='euclidean',
                cluster_selection_epsilon=0.5
            )
            cluster_labels = clusterer.fit_predict(self.embeddings)
            
        else:  # hierarchical
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            cluster_labels = clusterer.fit_predict(self.embeddings)
        
        # Analyze clusters
        clusters = []
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude noise (-1)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_labels_class = self.labels[cluster_mask]
            
            # Calculate cluster statistics
            n_sar = cluster_labels_class.sum()
            n_non_issue = len(cluster_labels_class) - n_sar
            sar_purity = n_sar / len(cluster_labels_class) if len(cluster_labels_class) > 0 else 0
            
            # Calculate cluster centroid
            centroid = cluster_embeddings.mean(axis=0)
            
            clusters.append({
                'cluster_id': int(cluster_id),
                'size': int(cluster_mask.sum()),
                'n_sar': int(n_sar),
                'n_non_issue': int(n_non_issue),
                'sar_purity': sar_purity,
                'centroid': centroid,
                'mask': cluster_mask
            })
        
        return pd.DataFrame(clusters), cluster_labels
    
    def extract_prototypical_narratives(self, narratives, n_prototypes=5):
        """
        Find the most "prototypical" narratives for each class
        These are narratives closest to class centroids
        """
        non_issue_centroid = self.non_issue_embeddings.mean(axis=0)
        sar_centroid = self.sar_embeddings.mean(axis=0)
        
        # Find narratives closest to centroids
        non_issue_distances = cdist(
            self.non_issue_embeddings,
            non_issue_centroid.reshape(1, -1),
            metric='cosine'
        ).flatten()
        
        sar_distances = cdist(
            self.sar_embeddings,
            sar_centroid.reshape(1, -1),
            metric='cosine'
        ).flatten()
        
        # Get top prototypes
        non_issue_indices = np.argsort(non_issue_distances)[:n_prototypes]
        sar_indices = np.argsort(sar_distances)[:n_prototypes]
        
        non_issue_narratives_array = np.array(narratives[:len(self.non_issue_embeddings)])
        sar_narratives_array = np.array(narratives[len(self.non_issue_embeddings):])
        
        prototypes = {
            'non_issue': [
                {
                    'narrative': non_issue_narratives_array[idx],
                    'distance': non_issue_distances[idx]
                }
                for idx in non_issue_indices
            ],
            'sar': [
                {
                    'narrative': sar_narratives_array[idx],
                    'distance': sar_distances[idx]
                }
                for idx in sar_indices
            ]
        }
        
        return prototypes
    
    def find_semantic_outliers(self, class_label='sar', percentile=95):
        """
        Find narratives that are semantic outliers within their class
        These might be unusual cases worth investigating
        """
        if class_label == 'sar':
            embeddings = self.sar_embeddings
            centroid = embeddings.mean(axis=0)
        else:
            embeddings = self.non_issue_embeddings
            centroid = embeddings.mean(axis=0)
        
        # Calculate distances from centroid
        distances = cdist(embeddings, centroid.reshape(1, -1), metric='cosine').flatten()
        
        # Find outliers
        threshold = np.percentile(distances, percentile)
        outlier_indices = np.where(distances >= threshold)[0]
        
        return outlier_indices, distances[outlier_indices]
    
    def discover_boundary_cases(self, narratives, n_cases=20):
        """
        Find narratives that lie on the decision boundary
        These are ambiguous cases that are hardest to classify
        """
        # Calculate distance to both centroids
        non_issue_centroid = self.non_issue_embeddings.mean(axis=0)
        sar_centroid = self.sar_embeddings.mean(axis=0)
        
        dist_to_non_issue = cdist(self.embeddings, non_issue_centroid.reshape(1, -1), metric='cosine').flatten()
        dist_to_sar = cdist(self.embeddings, sar_centroid.reshape(1, -1), metric='cosine').flatten()
        
        # Boundary cases have similar distance to both centroids
        distance_difference = np.abs(dist_to_non_issue - dist_to_sar)
        
        # Get top boundary cases
        boundary_indices = np.argsort(distance_difference)[:n_cases]
        
        boundary_cases = []
        narratives_array = np.array(narratives)
        
        for idx in boundary_indices:
            boundary_cases.append({
                'index': int(idx),
                'narrative': narratives_array[idx],
                'true_label': 'SAR' if self.labels[idx] == 1 else 'Non-Issue',
                'dist_to_non_issue': dist_to_non_issue[idx],
                'dist_to_sar': dist_to_sar[idx],
                'ambiguity_score': distance_difference[idx]
            })
        
        return pd.DataFrame(boundary_cases)


def run_bert_contrastive_analysis(non_issue_df, sar_df, narrative_col='NARRATIVE'):
    """
    Run comprehensive BERT-based contrastive analysis
    """
    print("="*80)
    print("BERT CONTRASTIVE CLUSTERING ANALYSIS")
    print("="*80)
    print("\nUsing transformer embeddings to discover semantic patterns\n")
    
    # Initialize analyzer
    analyzer = BERTContrastiveClusterAnalyzer(
        model_name='all-MiniLM-L6-v2',  # Fast and effective
        batch_size=32
    )
    
    # Generate embeddings
    print("1. Generating BERT embeddings...")
    embeddings, labels = analyzer.fit(
        non_issue_df[narrative_col].values,
        sar_df[narrative_col].values
    )
    
    print(f"   Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Find discriminative direction
    print("\n2. Finding discriminative direction in embedding space...")
    disc_vector, ni_proj, sar_proj = analyzer.find_discriminative_direction()
    
    print(f"   Non-issue projection mean: {ni_proj.mean():.3f} (±{ni_proj.std():.3f})")
    print(f"   SAR projection mean: {sar_proj.mean():.3f} (±{sar_proj.std():.3f})")
    print(f"   Separation: {abs(sar_proj.mean() - ni_proj.mean()):.3f}")
    
    # Discover narrative archetypes
    print("\n3. Discovering narrative archetypes...")
    clusters_df, cluster_labels = analyzer.discover_narrative_archetypes(
        n_clusters=15,
        method='hierarchical'
    )
    
    print("\n" + "="*80)
    print("NARRATIVE ARCHETYPES (Semantic Clusters)")
    print("="*80)
    
    # Separate into SAR-dominant and non-issue-dominant clusters
    sar_clusters = clusters_df[clusters_df['sar_purity'] > 0.7].sort_values('sar_purity', ascending=False)
    non_issue_clusters = clusters_df[clusters_df['sar_purity'] < 0.3].sort_values('sar_purity')
    
    print(f"\nFound {len(clusters_df)} total clusters")
    print(f"  • {len(sar_clusters)} SAR-dominant clusters (>70% SAR)")
    print(f"  • {len(non_issue_clusters)} Non-Issue-dominant clusters (<30% SAR)")
    
    if len(sar_clusters) > 0:
        print("\nSAR-DOMINANT ARCHETYPES:")
        for _, cluster in sar_clusters.head(8).iterrows():
            print(f"\n  Cluster {cluster['cluster_id']}: "
                  f"{cluster['size']} narratives "
                  f"({cluster['sar_purity']*100:.0f}% SAR)")
            print(f"    SAR: {cluster['n_sar']}, Non-Issue: {cluster['n_non_issue']}")
    
    if len(non_issue_clusters) > 0:
        print("\nNON-ISSUE-DOMINANT ARCHETYPES:")
        for _, cluster in non_issue_clusters.head(8).iterrows():
            print(f"\n  Cluster {cluster['cluster_id']}: "
                  f"{cluster['size']} narratives "
                  f"({cluster['sar_purity']*100:.0f}% SAR)")
            print(f"    SAR: {cluster['n_sar']}, Non-Issue: {cluster['n_non_issue']}")
    
    # Extract prototypical narratives
    print("\n4. Extracting prototypical narratives...")
    all_narratives = np.concatenate([
        non_issue_df[narrative_col].values,
        sar_df[narrative_col].values
    ])
    
    prototypes = analyzer.extract_prototypical_narratives(all_narratives, n_prototypes=3)
    
    print("\n" + "="*80)
    print("PROTOTYPICAL NARRATIVES")
    print("="*80)
    
    print("\nMost prototypical NON-ISSUE narratives (closest to non-issue centroid):")
    for i, proto in enumerate(prototypes['non_issue'], 1):
        print(f"\n{i}. (distance: {proto['distance']:.3f})")
        print(f"   {str(proto['narrative'])[:300]}...")
    
    print("\n\nMost prototypical SAR narratives (closest to SAR centroid):")
    for i, proto in enumerate(prototypes['sar'], 1):
        print(f"\n{i}. (distance: {proto['distance']:.3f})")
        print(f"   {str(proto['narrative'])[:300]}...")
    
    # Find boundary cases
    print("\n5. Finding boundary cases (ambiguous narratives)...")
    boundary_cases = analyzer.discover_boundary_cases(all_narratives, n_cases=10)
    
    print("\n" + "="*80)
    print("BOUNDARY CASES (Ambiguous Narratives)")
    print("="*80)
    print("\nThese narratives are equidistant from both class centroids:\n")
    
    for _, case in boundary_cases.iterrows():
        print(f"\nTrue label: {case['true_label']} (Ambiguity: {case['ambiguity_score']:.3f})")
        print(f"  {str(case['narrative'])[:250]}...")
    
    # Find semantic outliers
    print("\n6. Finding semantic outliers...")
    sar_outliers, _ = analyzer.find_semantic_outliers('sar', percentile=95)
    ni_outliers, _ = analyzer.find_semantic_outliers('non_issue', percentile=95)
    
    print(f"\n  Found {len(sar_outliers)} SAR outliers (unusual SAR narratives)")
    print(f"  Found {len(ni_outliers)} Non-Issue outliers (unusual non-issue narratives)")
    
    # Save results
    print("\n7. Saving results...")
    
    # Add cluster assignments to original dataframes
    non_issue_df_copy = non_issue_df.copy()
    sar_df_copy = sar_df.copy()
    
    non_issue_df_copy['cluster'] = cluster_labels[:len(non_issue_df)]
    sar_df_copy['cluster'] = cluster_labels[len(non_issue_df):]
    
    non_issue_df_copy['projection'] = ni_proj
    sar_df_copy['projection'] = sar_proj
    
    non_issue_df_copy.to_csv('non_issue_with_clusters.csv', index=False)
    sar_df_copy.to_csv('sar_with_clusters.csv', index=False)
    
    clusters_df.to_csv('narrative_archetypes.csv', index=False)
    boundary_cases.to_csv('boundary_cases.csv', index=False)
    
    print("\n" + "="*80)
    print("✓ Results saved:")
    print("  - non_issue_with_clusters.csv (narratives with cluster assignments)")
    print("  - sar_with_clusters.csv (narratives with cluster assignments)")
    print("  - narrative_archetypes.csv (cluster statistics)")
    print("  - boundary_cases.csv (ambiguous cases)")
    
    return {
        'analyzer': analyzer,
        'clusters': clusters_df,
        'prototypes': prototypes,
        'boundary_cases': boundary_cases,
        'sar_clusters': sar_clusters,
        'non_issue_clusters': non_issue_clusters
    }


# Example usage:
"""
# Install required library first:
# pip install sentence-transformers

import pandas as pd

# Load your data
non_issue_df = pd.read_csv('non_issue_alerts.csv')
sar_df = pd.read_csv('sar_alerts.csv')

# Run BERT analysis
results = run_bert_contrastive_analysis(non_issue_df, sar_df)

# Examine specific clusters
sar_cluster_0 = sar_df[results['analyzer'].cluster_labels == 0]
print(f"\nNarratives in SAR cluster 0:")
print(sar_cluster_0['NARRATIVE'].head(10))

# Look at boundary cases - these are your hardest to classify
print("\nBoundary cases that might need manual review:")
print(results['boundary_cases'])

# Find which cluster a new narrative belongs to
new_narrative = "Customer made large deposit..."
new_embedding = results['analyzer'].generate_embeddings([new_narrative])
# Then use cosine similarity to find nearest cluster centroid
"""