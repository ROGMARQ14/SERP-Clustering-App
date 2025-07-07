import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from typing import Dict, List, Tuple

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def optimized_cluster_keywords_by_serp(
    input_file: str = "serp_results.json",
    output_file: str = "clustered_keywords.csv",
    similarity_threshold: float = 0.3,
    use_advanced_clustering: bool = True
) -> pd.DataFrame:
    """
    Optimized keyword clustering with multiple algorithm options and caching
    """

    # Load SERP results with error handling
    try:
        with open(input_file, 'r') as f:
            serp_results = json.load(f)
    except FileNotFoundError:
        st.error(f"File {input_file} not found")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in {input_file}")
        return pd.DataFrame()

    if not serp_results:
        st.warning("No SERP results found")
        return pd.DataFrame()

    keywords = list(serp_results.keys())
    n = len(keywords)

    progress_bar = st.progress(0)
    status_text = st.empty()

    if use_advanced_clustering:
        # Use TF-IDF vectorization for better semantic understanding
        status_text.text("Creating TF-IDF vectors from SERP URLs...")
        clusters = _advanced_clustering_with_tfidf(serp_results, keywords, similarity_threshold)
        progress_bar.progress(0.6)
    else:
        # Original URL overlap method (optimized)
        status_text.text("Calculating URL similarity matrix...")
        clusters = _url_overlap_clustering(serp_results, keywords, similarity_threshold)
        progress_bar.progress(0.6)

    # Create output dataframe
    status_text.text("Creating output dataframe...")
    df = _create_clustered_dataframe(clusters, serp_results)
    progress_bar.progress(0.8)

    # Sort and save results
    df = df.sort_values(by=["Cluster Size", "Hub", "Keyword"], ascending=[False, True, True])
    df.to_csv(output_file, index=False)

    progress_bar.progress(1.0)
    status_text.text(f"✅ Clustered {len(df)} keywords into {len(clusters)} clusters")

    return df

def _advanced_clustering_with_tfidf(
    serp_results: Dict[str, List[str]], 
    keywords: List[str], 
    similarity_threshold: float
) -> List[Dict]:
    """
    Advanced clustering using TF-IDF vectorization of SERP URLs
    This provides better semantic understanding than simple URL overlap
    """

    # Create documents from SERP URLs for each keyword
    documents = []
    for keyword in keywords:
        urls = serp_results[keyword]
        # Extract domain names and paths for better feature extraction
        url_features = []
        for url in urls:
            try:
                # Simple domain extraction (can be enhanced with urllib.parse)
                domain = url.split('/')[2] if len(url.split('/')) > 2 else url
                path = '/'.join(url.split('/')[3:]) if len(url.split('/')) > 3 else ''
                url_features.extend([domain, path])
            except:
                url_features.append(url)
        documents.append(' '.join(url_features))

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Apply clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(similarity_matrix)

    except Exception as e:
        st.warning(f"TF-IDF clustering failed: {e}. Falling back to URL overlap method.")
        return _url_overlap_clustering(serp_results, keywords, similarity_threshold)

    # Group keywords by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(keywords[i])

    # Convert to the expected format
    cluster_list = []
    for cluster_id, cluster_keywords in clusters.items():
        # Find the hub (keyword with most URL overlap with others in cluster)
        hub = _find_best_hub(cluster_keywords, serp_results)

        urls = set()
        for kw in cluster_keywords:
            urls.update(serp_results[kw])

        cluster_list.append({
            "hub": hub,
            "keywords": cluster_keywords,
            "urls": urls
        })

    return cluster_list

def _url_overlap_clustering(
    serp_results: Dict[str, List[str]], 
    keywords: List[str], 
    similarity_threshold: float
) -> List[Dict]:
    """
    Optimized version of original URL overlap clustering with parallel processing
    """
    n = len(keywords)

    # Parallelize similarity matrix calculation
    def calculate_similarity_row(i):
        row = np.zeros(n)
        urls1 = set(serp_results[keywords[i]])
        for j in range(n):
            if i == j:
                row[j] = 1.0
            else:
                urls2 = set(serp_results[keywords[j]])
                intersection = len(urls1.intersection(urls2))
                union = len(urls1.union(urls2))
                row[j] = intersection / union if union > 0 else 0.0
        return row

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        similarity_rows = list(executor.map(calculate_similarity_row, range(n)))

    similarity_matrix = np.array(similarity_rows)

    # Find hubs and create clusters (original algorithm)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    hub_scores = [(keywords[i], avg_similarities[i]) for i in range(n)]
    hub_scores.sort(key=lambda x: x[1], reverse=True)

    assigned = set()
    clusters = []

    for hub_keyword, hub_score in hub_scores:
        if hub_keyword in assigned:
            continue

        hub_idx = keywords.index(hub_keyword)
        cluster = {
            "hub": hub_keyword, 
            "keywords": [hub_keyword], 
            "urls": set(serp_results[hub_keyword])
        }
        assigned.add(hub_keyword)

        # Find similar keywords
        for j, other_keyword in enumerate(keywords):
            if (other_keyword not in assigned and 
                similarity_matrix[hub_idx][j] >= similarity_threshold):
                cluster["keywords"].append(other_keyword)
                cluster["urls"].update(serp_results[other_keyword])
                assigned.add(other_keyword)

        clusters.append(cluster)

    return clusters

def _find_best_hub(cluster_keywords: List[str], serp_results: Dict[str, List[str]]) -> str:
    """Find the keyword that best represents the cluster (most central)"""
    if len(cluster_keywords) == 1:
        return cluster_keywords[0]

    # Calculate which keyword has the most URL overlap with others
    best_hub = cluster_keywords[0]
    max_overlap = 0

    for candidate in cluster_keywords:
        candidate_urls = set(serp_results[candidate])
        total_overlap = 0

        for other in cluster_keywords:
            if other != candidate:
                other_urls = set(serp_results[other])
                overlap = len(candidate_urls.intersection(other_urls))
                total_overlap += overlap

        if total_overlap > max_overlap:
            max_overlap = total_overlap
            best_hub = candidate

    return best_hub

def _create_clustered_dataframe(clusters: List[Dict], serp_results: Dict[str, List[str]]) -> pd.DataFrame:
    """Create the output dataframe from cluster results"""
    rows = []
    for cluster in clusters:
        hub = cluster["hub"]
        urls_str = "\n".join(list(cluster["urls"])[:10])  # Top 10 URLs

        for keyword in cluster["keywords"]:
            rows.append({
                "Hub": hub,
                "Keyword": keyword,
                "Cluster Size": len(cluster["keywords"]),
                "URLs": urls_str
            })

    return pd.DataFrame(rows)

# For backwards compatibility
def cluster_keywords_by_serp(input_file="serp_results.json", output_file="clustered_keywords.csv", similarity_threshold=0.3):
    """Original function signature for backwards compatibility"""
    return optimized_cluster_keywords_by_serp(input_file, output_file, similarity_threshold)
