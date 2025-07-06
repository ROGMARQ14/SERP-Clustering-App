import streamlit as st
import pandas as pd
import openai
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import json
import requests
from collections import defaultdict
import networkx as nx

# --------------------
# Streamlit UI Setup
# --------------------
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("🧠 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get intent-based clusters using SERP overlap + semantic analysis.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    clustering_method = st.radio(
        "Clustering Method",
        ["SERP Overlap + Semantic", "SERP Overlap Only", "Semantic Only"],
        help="Choose how to cluster keywords"
    )
    
    if clustering_method != "Semantic Only":
        serper_api_key = st.text_input("Serper API Key", type="password", help="Get your API key from https://serper.dev")
        serp_weight = st.slider("SERP Weight", 0.0, 1.0, 0.7, 0.1, help="Weight for SERP overlap (vs semantic similarity)")
    else:
        serper_api_key = ""
        serp_weight = 0.0
    
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=1
        )
        
        sim_threshold = st.slider("Similarity Threshold", 0.1, 0.9, 0.3, 0.05)
        min_cluster_size = st.number_input("Min Cluster Size", 1, 10, 2)
        max_clusters = st.number_input("Max Clusters (0=auto)", 0, 100, 0)

# Main area
uploaded_file = st.file_uploader("Upload your keywords.csv file", type="csv")

# --------------------
# Helper Functions
# --------------------
def fetch_serp_results(keyword, serper_api_key):
    """Fetch top 10 SERP results for a keyword"""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": keyword, "num": 10}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get("organic", [])
            urls = [item.get("link") for item in organic_results][:10]
            return urls
        else:
            st.warning(f"SERP fetch failed for '{keyword}': {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"SERP fetch error for '{keyword}': {e}")
        return []

def calculate_serp_similarity(urls1, urls2):
    """Calculate Jaccard similarity between two URL lists"""
    if not urls1 or not urls2:
        return 0.0
    
    set1 = set(urls1)
    set2 = set(urls2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def get_embedding(text, client):
    """Get OpenAI embedding for text"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding failed for '{text}': {e}")
        return None

def find_hub_keywords(keywords, similarity_matrix, threshold=0.5):
    """Find hub keywords using graph-based approach"""
    G = nx.Graph()
    
    # Add edges for keywords with similarity above threshold
    for i in range(len(keywords)):
        for j in range(i + 1, len(keywords)):
            if similarity_matrix[i][j] >= threshold:
                G.add_edge(keywords[i], keywords[j], weight=similarity_matrix[i][j])
    
    # Calculate degree centrality to find hubs
    if G.number_of_nodes() == 0:
        return {}
    
    centrality = nx.degree_centrality(G)
    
    # Assign keywords to their best hub
    hub_assignments = {}
    components = list(nx.connected_components(G))
    
    for component in components:
        if len(component) == 1:
            # Single keyword cluster
            kw = list(component)[0]
            hub_assignments[kw] = kw
        else:
            # Find best hub in component
            component_centrality = {node: centrality[node] for node in component}
            hub = max(component_centrality, key=component_centrality.get)
            
            # Assign all keywords in component to this hub
            for kw in component:
                hub_assignments[kw] = hub
    
    # Handle unconnected keywords
    all_keywords = set(keywords)
    connected_keywords = set(hub_assignments.keys())
    unconnected = all_keywords - connected_keywords
    
    for kw in unconnected:
        hub_assignments[kw] = kw
    
    return hub_assignments

def generate_cluster_label(keywords, hub_keyword, client, model):
    """Generate intelligent cluster label based on keywords and search intent"""
    # Limit keywords for prompt to avoid token limits
    sample_keywords = keywords[:10] if len(keywords) > 10 else keywords
    
    prompt = f"""You are an SEO expert analyzing search intent. Given these keywords that appear in Google search results together:

Hub/Main Keyword: {hub_keyword}
Related Keywords: {', '.join(sample_keywords)}

Generate a concise 2-4 word label that captures the common search intent or topic.

Rules:
- Focus on the user's search intent (informational, navigational, transactional, commercial)
- Use industry-standard terminology
- Be specific enough to differentiate from other clusters
- Don't just repeat the hub keyword

Return only the label, nothing else."""

    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"{hub_keyword.title()} Services"

def cluster_keywords(keywords, serp_data, embeddings, method, serp_weight, threshold):
    """Cluster keywords using chosen method"""
    n = len(keywords)
    
    if method == "SERP Overlap Only":
        # Pure SERP similarity matrix
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = calculate_serp_similarity(
                        serp_data.get(keywords[i], []),
                        serp_data.get(keywords[j], [])
                    )
    
    elif method == "Semantic Only":
        # Pure semantic similarity
        similarity_matrix = cosine_similarity(embeddings)
    
    else:  # SERP Overlap + Semantic
        # Calculate SERP similarity
        serp_similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    serp_similarity[i][j] = 1.0
                else:
                    serp_similarity[i][j] = calculate_serp_similarity(
                        serp_data.get(keywords[i], []),
                        serp_data.get(keywords[j], [])
                    )
        
        # Calculate semantic similarity
        semantic_similarity = cosine_similarity(embeddings)
        
        # Weighted combination
        similarity_matrix = (serp_weight * serp_similarity + 
                           (1 - serp_weight) * semantic_similarity)
    
    # Find hub assignments
    hub_assignments = find_hub_keywords(keywords, similarity_matrix, threshold)
    
    return hub_assignments, similarity_matrix

# --------------------
# Main Processing
# --------------------
if st.button("🚀 Run Clustering") and uploaded_file and openai_api_key:
    if clustering_method != "Semantic Only" and not serper_api_key:
        st.error("Please provide a Serper API key for SERP-based clustering")
        st.stop()
    
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        
        # Find keyword column
        keyword_col = None
        for col in df.columns:
            if col.lower() in ['keyword', 'keywords', 'query', 'queries']:
                keyword_col = col
                break
        
        if not keyword_col:
            keyword_col = df.columns[0]
            st.warning(f"Using '{keyword_col}' as keyword column")
        
        # Extract unique keywords
        keywords = df[keyword_col].dropna().astype(str).str.strip().unique().tolist()
        keywords = [kw for kw in keywords if kw]  # Remove empty strings
        
        st.info(f"Processing {len(keywords)} unique keywords...")
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Fetch SERP data if needed
        serp_data = {}
        if clustering_method != "Semantic Only":
            status_text.text("Fetching SERP data...")
            for i, kw in enumerate(keywords):
                serp_data[kw] = fetch_serp_results(kw, serper_api_key)
                progress_bar.progress((i + 1) / (len(keywords) * 2))
                time.sleep(1.2)  # Rate limiting
        
        # Step 2: Generate embeddings
        status_text.text("Generating embeddings...")
        embeddings = []
        valid_keywords = []
        
        for i, kw in enumerate(keywords):
            emb = get_embedding(kw, client)
            if emb:
                embeddings.append(emb)
                valid_keywords.append(kw)
            
            if clustering_method != "Semantic Only":
                progress_bar.progress(0.5 + (i + 1) / (len(keywords) * 2))
            else:
                progress_bar.progress((i + 1) / len(keywords))
            
            time.sleep(0.5)  # Rate limiting
        
        # Step 3: Cluster keywords
        status_text.text("Clustering keywords...")
        hub_assignments, similarity_matrix = cluster_keywords(
            valid_keywords, serp_data, embeddings, 
            clustering_method, serp_weight, sim_threshold
        )
        
        # Step 4: Generate cluster labels
        status_text.text("Generating cluster labels...")
        cluster_groups = defaultdict(list)
        for kw, hub in hub_assignments.items():
            cluster_groups[hub].append(kw)
        
        # Filter small clusters if needed
        if min_cluster_size > 1:
            cluster_groups = {hub: kws for hub, kws in cluster_groups.items() 
                            if len(kws) >= min_cluster_size}
        
        # Generate results
        results = []
        for hub, kws in cluster_groups.items():
            label = generate_cluster_label(kws, hub, client, openai_model)
            
            # Add URLs column if SERP data available
            for kw in kws:
                row = {
                    "Cluster Label": label,
                    "Hub Keyword": hub,
                    "Keyword": kw,
                    "Cluster Size": len(kws)
                }
                
                if clustering_method != "Semantic Only":
                    urls = serp_data.get(kw, [])
                    row["Top 3 URLs"] = "\n".join(urls[:3])
                
                results.append(row)
        
        # Create final dataframe
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values(by=["Cluster Label", "Cluster Size"], 
                                      ascending=[True, False])
        
        # Display results
        progress_bar.progress(1.0)
        status_text.text("✅ Clustering complete!")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clusters", len(cluster_groups))
        with col2:
            st.metric("Clustered Keywords", len(final_df))
        with col3:
            st.metric("Avg Cluster Size", f"{len(final_df) / len(cluster_groups):.1f}")
        
        # Download button
        csv = final_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            data=csv,
            file_name="clustered_keywords.csv",
            mime="text/csv"
        )
        
        # Display results
        st.subheader("📊 Clustering Results")
        
        # Add filtering options
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_cluster = st.selectbox(
                "Filter by Cluster",
                ["All"] + sorted(final_df["Cluster Label"].unique())
            )
        
        # Filter dataframe
        display_df = final_df
        if selected_cluster != "All":
            display_df = final_df[final_df["Cluster Label"] == selected_cluster]
        
        st.dataframe(display_df, use_container_width=True, height=600)
        
        # Cluster overview
        st.subheader("📈 Cluster Overview")
        cluster_summary = final_df.groupby("Cluster Label").agg({
            "Keyword": "count",
            "Hub Keyword": "first"
        }).rename(columns={"Keyword": "Size"}).sort_values("Size", ascending=False)
        
        st.dataframe(cluster_summary, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
