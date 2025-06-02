import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")
st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate content-ready keyword clusters based on SERP overlap.")

# Session state initialization
if 'final_df' not in st.session_state:
    st.session_state.final_df = None

# File uploader
uploaded_file = st.file_uploader("📥 Upload your keywords.csv file", type="csv")

# API key inputs
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")

# Threshold slider
threshold = st.slider("🔧 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

# Clustering logic
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    keywords_df = pd.read_csv(uploaded_file)

    # Try detecting a keyword column or fallback to first column
    possible_cols = ["Keyword", "keyword", "KEYWORD", "Query", "query", "queries"]
    found_col = next((col for col in keywords_df.columns if col in possible_cols), keywords_df.columns[0])

    keywords = keywords_df[found_col].dropna().unique().tolist()

    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    progress_bar = st.progress(0)

    for i, keyword in enumerate(keywords):
        with st.spinner(f"Fetching SERP for: {keyword}"):
            response = requests.post("https://google.serper.dev/search", headers=headers, json={"q": keyword})
            if response.status_code == 200:
                data = response.json().get("organic", [])
                urls = [item.get("link") for item in data][:10]
                serp_data[keyword] = urls
            else:
                serp_data[keyword] = []
            time.sleep(1)
        progress_bar.progress((i + 1) / len(keywords))

    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

    clusters = []
    unclustered = set(serp_data.keys())
    while unclustered:
        hub = unclustered.pop()
        cluster = [hub]
        to_compare = list(unclustered)
        for kw in to_compare:
            if jaccard(serp_data[hub], serp_data[kw]) >= threshold:
                cluster.append(kw)
                unclustered.remove(kw)
        clusters.append(cluster)

    openai.api_key = openai_api
    labeled_rows = []
    for cluster in clusters:
        hub = cluster[0]
        prompt = f"Generate a short and meaningful SEO topic label for the following keywords:\n{cluster}"
        try:
            res = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            label = res.choices[0].message['content'].strip()
            if not label:
                label = f"Cluster: {hub}"
        except Exception:
            label = f"Cluster: {hub}"

        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw,
                "URLs": "\n".join(serp_data.get(kw, []))
            })

    final_df = pd.DataFrame(labeled_rows)
    st.session_state.final_df = final_df
    st.success("✅ Clustering completed!")

# Show results if they exist
if st.session_state.final_df is not None:
    st.dataframe(st.session_state.final_df)
    csv_data = st.session_state.final_df.to_csv(index=False, sep=',', encoding='utf-8')
    st.download_button(
        label="📥 Download Clustered CSV",
        data=csv_data,
        file_name="final_clustered_keywords.csv",
        mime="text/csv"
    )
