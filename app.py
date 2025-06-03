import streamlit as st
import pandas as pd
import openai  # still imported if you want to test hybrid later
import requests
import time
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Page setup
st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")
st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API key, and generate content-ready keyword clusters based on SERP overlap.")

# Upload & API Inputs
uploaded_file = st.file_uploader("📤 Upload your keywords.csv file", type="csv")
serper_api = st.text_input("🔑 Serper API Key", type="password")
threshold = st.slider("🛠️ SERP Similarity Threshold (%)", 10, 100, 30) / 100

# Initialize session state
if 'final_df' not in st.session_state:
    st.session_state.final_df = None

progress_bar = st.progress(0)

# Clean cluster keywords for label generation
def clean_keywords(keywords):
    cleaned = []
    for kw in keywords:
        kw = kw.lower()
        kw = re.sub(r'\b(in|near|services|best|top|rated|local|new|ann arbor|contractors|companies|repair)\b', '', kw)
        kw = re.sub(r'\s+', ' ', kw).strip()
        cleaned.append(kw)
    return cleaned

# Extract most common 2-3 word phrase in a cluster
def extract_top_phrase(keywords):
    try:
        vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english').fit(keywords)
        X = vectorizer.transform(keywords)
        word_freq = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        return vocab[word_freq.argmax()] if len(vocab) > 0 else ""
    except:
        return ""

# Main logic
if st.button("🚀 Run Clustering") and uploaded_file and serper_api:
    st.info("Processing... Please wait.")
    keywords_df = pd.read_csv(uploaded_file)

    # Detect keyword column
    column_names = [col.lower().strip() for col in keywords_df.columns]
    keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in column_names), None)
    keyword_col = keywords_df.columns[column_names.index(keyword_col)] if keyword_col else keywords_df.columns[0]

    keywords = keywords_df[keyword_col].dropna().unique().tolist()

    # Fetch SERP results
    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}
    for i, keyword in enumerate(keywords):
        progress_bar.progress((i + 1) / len(keywords))
        try:
            response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": keyword})
            urls = [item.get("link") for item in response.json().get("organic", [])][:10]
        except Exception:
            urls = []
        serp_data[keyword] = urls
        time.sleep(1)

    # Jaccard similarity
    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

    # Cluster keywords
    clusters = []
    unclustered = set(keywords)
    while unclustered:
        hub = unclustered.pop()
        cluster = [hub]
        to_compare = list(unclustered)
        for kw in to_compare:
            if jaccard(serp_data[hub], serp_data[kw]) >= threshold:
                cluster.append(kw)
                unclustered.remove(kw)
        clusters.append(cluster)

    # Label clusters using CountVectorizer + fallback
    labeled_rows = []
    for i, cluster in enumerate(clusters):
        cleaned_cluster = clean_keywords(cluster)
        top_phrase = extract_top_phrase(cleaned_cluster)
        label = top_phrase if top_phrase else cluster[0]  # fallback to hub

        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": cluster[0],
                "Keyword": kw
            })

    # Final DataFrame
    final_df = pd.DataFrame(labeled_rows)
    st.session_state.final_df = final_df
    st.success("✅ Clustering complete!")

# Display & download
if st.session_state.final_df is not None:
    csv_data = st.session_state.final_df.to_csv(index=False, encoding="utf-8")
    st.download_button("📥 Download Clustered CSV", data=csv_data, file_name="final_clustered_keywords.csv", mime="text/csv")

    display_df = st.session_state.final_df.copy()
    display_df["Cluster Label"] = display_df["Cluster Label"].apply(lambda x: f"**{x}**")
    st.markdown("### 📊 Preview of Clustered Keywords")
    st.dataframe(display_df, use_container_width=True)
