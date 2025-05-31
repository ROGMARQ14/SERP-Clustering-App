import streamlit as st
import pandas as pd
import json
import openai
import requests
import time

# Page setup
st.set_page_config(page_title="SERP Keyword Clustering", layout="wide")

st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV, enter your API keys, and generate content-ready keyword clusters based on SERP overlap.")

# Upload keywords CSV
uploaded_file = st.file_uploader("📥 Upload your keywords.csv file", type="csv")

# API keys input
serper_api = st.text_input("🔑 Serper API Key", type="password")
openai_api = st.text_input("🔑 OpenAI API Key", type="password")

# Similarity threshold
threshold = st.slider("🔧 SERP Similarity Threshold (%)", min_value=10, max_value=100, value=30) / 100

# Clustering trigger
if st.button("🚀 Run Clustering") and uploaded_file and serper_api and openai_api:
    st.info("Processing... Please wait.")
    
    # Read uploaded keywords
    keywords_df = pd.read_csv(uploaded_file)
    keywords = keywords_df['Keyword'].dropna().unique().tolist()

    headers = {"X-API-KEY": serper_api, "Content-Type": "application/json"}
    serp_data = {}

    # Fetch SERP results
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

    # Define Jaccard similarity
    def jaccard(set1, set2):
        return len(set(set1) & set(set2)) / len(set(set1) | set(set2)) if set1 or set2 else 0

    # Cluster keywords
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

    # Generate AI-based labels
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
        except Exception:
            label = "Unnamed Cluster"

        for kw in cluster:
            labeled_rows.append({
                "Cluster Label": label,
                "Hub": hub,
                "Keyword": kw,
                "URLs": "\n".join(serp_data.get(kw, []))
            })

    # Final output
    final_df = pd.DataFrame(labeled_rows)
    st.success("✅ Clustering completed!")
    st.download_button("📥 Download Clustered CSV", final_df.to_csv(index=False), file_name="final_clustered_keywords.csv", mime="text/csv")
    st.dataframe(final_df)
