import streamlit as st
import pandas as pd
import openai
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# --------------------
# Streamlit UI Setup
# --------------------
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("\U0001F9E0 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

uploaded_file = st.file_uploader("Upload your keywords.csv file", type="csv")
openai_api_key = st.text_input("OpenAI API Key", type="password")
sim_threshold = st.slider("Cosine Similarity Threshold", min_value=70, max_value=95, value=80)
progress_text = st.empty()
progress_bar = st.progress(0)

# --------------------
# Helper Functions
# --------------------
def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.warning(f"Embedding failed for '{text}': {e}")
        return None

def generate_label(keywords, client):
    prompt = f"""
You're an SEO assistant. Given the following keywords:
{keywords}
Return a short, generalized 2–4 word label that describes the group. Avoid long-tails or exact matches. Just return the label, nothing else.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return res.choices[0].message.content.strip().title()
    except Exception as e:
        st.warning(f"GPT label generation failed: {e}")
        return "Unlabeled Cluster"

# --------------------
# Clustering Logic
# --------------------
if st.button("Run Clustering") and uploaded_file and openai_api_key:
    st.info("Embedding and clustering... hold tight.")
    try:
        df = pd.read_csv(uploaded_file)
        column_names = [col.lower().strip() for col in df.columns]
        keyword_col = next((col for col in ['keyword', 'keywords', 'query', 'queries'] if col in column_names), df.columns[0])
        keyword_col = df.columns[column_names.index(keyword_col)]
        keywords = df[keyword_col].dropna().unique().tolist()

        client = openai.OpenAI(api_key=openai_api_key)

        embeddings = []
        valid_keywords = []
        for i, kw in enumerate(keywords):
            progress_text.text(f"Embedding {i+1}/{len(keywords)}: {kw}")
            emb = get_embedding(kw, client)
            if emb:
                embeddings.append(emb)
                valid_keywords.append(kw)
            progress_bar.progress((i + 1) / len(keywords))
            time.sleep(0.5)

        if len(embeddings) < 2:
            st.error("❌ Not enough valid embeddings were generated to proceed with clustering.")
            st.stop()

        st.success("✅ All embeddings generated. Proceeding to cluster...")

        similarity = cosine_similarity(embeddings)
        distance = 1 - similarity

        clustering = AgglomerativeClustering(
            metric='precomputed',
            linkage='average',
            distance_threshold=1 - (sim_threshold / 100),
            n_clusters=None
        ).fit(distance)

        labels = clustering.labels_
        df_clustered = pd.DataFrame({"Keyword": valid_keywords, "Cluster": labels})

        results = []
        total_clusters = df_clustered["Cluster"].nunique()

        for cluster_id in sorted(df_clustered["Cluster"].unique()):
            kws = df_clustered[df_clustered["Cluster"] == cluster_id]["Keyword"].tolist()
            label = generate_label(kws, client)
            for kw in kws:
                results.append({
                    "Topic Cluster": label,
                    "Cluster Size": len(kws),
                    "Keyword": kw
                })

        final_df = pd.DataFrame(results).sort_values(by=["Topic Cluster", "Keyword"])

        if final_df.empty:
            st.warning("⚠️ Clustering completed but no meaningful clusters were formed. Try reducing the similarity threshold.")
            st.stop()

        percent_clustered = round((len(final_df) / len(keywords)) * 100, 2)
        st.success(f"✅ Clustering complete! {percent_clustered}% of keywords clustered.")

        csv = final_df.to_csv(index=False, encoding="utf-8")
        st.download_button("Download Clustered CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv", key="download_csv")
        st.markdown("### 🔍 Final Clustered Output")
        st.dataframe(final_df, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong during clustering: {e}")
