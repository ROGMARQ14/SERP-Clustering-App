import streamlit as st
import pandas as pd
import openai
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import os
import subprocess

# Try to import custom modules
try:
    from fetch_serp import fetch_all_serp_results
    from label_clusters import label_keyword_clusters
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# --------------------
# Streamlit UI Setup
# --------------------
st.set_page_config(page_title="AI-Based Keyword Clustering Tool", layout="wide")
st.title("\U0001F9E0 AI-Based Keyword Clustering Tool")
st.markdown("Upload your keyword CSV and get semantic, search-intent-based clusters with smart labels.")

uploaded_file = st.file_uploader("Upload your keywords.csv file", type="csv")
openai_api_key = st.text_input("OpenAI API Key", type="password")
serper_api_key = st.text_input("Serper API Key (for SERP data)", type="password", help="Get your API key from https://serper.dev")

# Model and temperature settings
col1, col2 = st.columns(2)
with col1:
    openai_model = st.selectbox(
        "OpenAI Model for Labeling",
        ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Model used for generating cluster labels"
    )
with col2:
    label_temperature = st.slider(
        "Label Generation Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Higher values make labels more creative, lower values more focused"
    )

sim_threshold = st.slider("Cosine Similarity Threshold", min_value=70, max_value=95, value=80)
progress_text = st.empty()
progress_bar = st.progress(0)

# --------------------
# Helper Functions
# --------------------
def load_csv_safely(uploaded_file):
    """Safely load CSV with multiple fallback methods"""
    try:
        # First, try standard CSV loading
        df = pd.read_csv(uploaded_file)
        return df, None
    except pd.errors.ParserError as e:
        # Don't show the technical error, just handle it quietly
        uploaded_file.seek(0)
        
        try:
            # Try with different separators and error handling
            df = pd.read_csv(uploaded_file, sep=',', on_bad_lines='skip')
            st.info("⚠️ Some malformed lines were automatically skipped during CSV loading.")
            
            # If we only got 1 column, try other delimiters
            if df.shape[1] == 1:
                uploaded_file.seek(0)
                
                # Try semicolon delimiter
                try:
                    df_semi = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip')
                    if df_semi.shape[1] > 1:
                        st.info("✅ Using semicolon (;) as delimiter")
                        return df_semi, "semicolon_delimiter"
                except:
                    pass
                
                uploaded_file.seek(0)
                
                # Try tab delimiter
                try:
                    df_tab = pd.read_csv(uploaded_file, sep='\t', on_bad_lines='skip')
                    if df_tab.shape[1] > 1:
                        st.info("✅ Using tab as delimiter")
                        return df_tab, "tab_delimiter"
                except:
                    pass
                
                uploaded_file.seek(0)
                
                # If still single column, check if it's actually all keywords in one column
                st.info("📋 Single column detected - treating as keyword list")
            
            return df, "skipped_lines"
        except Exception as e2:
            st.warning(f"Alternative parsing method failed: {e2}")
            
            # Reset file pointer again
            uploaded_file.seek(0)
            
            try:
                # Try reading with no header and then clean
                df = pd.read_csv(uploaded_file, header=None, on_bad_lines='skip')
                st.info("⚠️ CSV loaded without headers. Please verify column structure.")
                return df, "no_header"
            except Exception as e3:
                return None, f"All parsing methods failed: {e3}"

def get_embedding(text, client):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        if "unsupported_country_region_territory" in str(e):
            st.error("🚫 **Regional Restriction**: OpenAI API is not available in your region.")
            st.markdown("""
            **Solutions:**
            1. Use a VPN to connect from a supported region (US, UK, Canada, etc.)
            2. Check your OpenAI account region settings
            3. Try a different API key from a supported region
            
            **Supported regions**: https://platform.openai.com/docs/supported-countries
            """)
        else:
            st.warning(f"Embedding failed for '{text}': {e}")
        return None

def generate_label(keywords, client, model, temperature):
    prompt = f"""
You're an SEO assistant. Given the following keywords:
{keywords}
Return a short, generalized 2–4 word label that describes the group. Avoid long-tails or exact matches. Just return the label, nothing else.
"""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
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
        # Load CSV with error handling
        df, load_status = load_csv_safely(uploaded_file)
        
        if df is None:
            st.error(f"❌ Failed to load CSV file: {load_status}")
            st.markdown("""
            **Troubleshooting tips:**
            1. Check if your CSV has consistent delimiters (commas)
            2. Look for extra commas or quotes in your data
            3. Ensure all rows have the same number of columns
            4. Try opening the CSV in a text editor to check line 157
            """)
            st.stop()
        
        # Display CSV info for debugging
        st.info(f"✅ CSV loaded successfully. Shape: {df.shape}")
        st.markdown("**Preview of loaded data:**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Find keyword column with better error handling
        if df.empty:
            st.error("❌ The loaded CSV is empty.")
            st.stop()
            
        column_names = [col.lower().strip() for col in df.columns]
        st.info(f"Available columns: {list(df.columns)}")
        
        # Look for keyword column
        keyword_col = None
        for potential_col in ['keyword', 'keywords', 'query', 'queries']:
            if potential_col in column_names:
                keyword_col = df.columns[column_names.index(potential_col)]
                break
        
        if keyword_col is None:
            # Use first column as fallback
            keyword_col = df.columns[0]
            st.warning(f"⚠️ No standard keyword column found. Using '{keyword_col}' as keyword column.")
        else:
            st.success(f"✅ Using '{keyword_col}' as keyword column.")
        
        # Special handling for single-column CSVs
        if df.shape[1] == 1:
            st.info("📋 Single column CSV detected. All data will be treated as keywords.")
            # Rename column to something more descriptive if it's unclear
            if keyword_col.lower() in ['0', 'unnamed: 0', 'column1'] or keyword_col.startswith('Unnamed'):
                df = df.rename(columns={keyword_col: 'Keywords'})
                keyword_col = 'Keywords'
                st.info(f"📝 Renamed column to '{keyword_col}' for clarity.")
        
        # Extract keywords with better cleaning
        keywords_raw = df[keyword_col].dropna().astype(str).str.strip()
        keywords_raw = keywords_raw[keywords_raw != '']  # Remove empty strings
        keywords = keywords_raw.unique().tolist()
        
        if len(keywords) == 0:
            st.error("❌ No valid keywords found in the selected column.")
            st.stop()
            
        st.info(f"Found {len(keywords)} unique keywords to process.")

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
            label = generate_label(kws, client, openai_model, label_temperature)
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

        # Store results in session state to prevent reset
        st.session_state.clustering_results = final_df
        st.session_state.clustering_complete = True

        percent_clustered = round((len(final_df) / len(keywords)) * 100, 2)
        st.success(f"✅ Clustering complete! {percent_clustered}% of keywords clustered.")

    except Exception as e:
        st.error(f"Something went wrong during clustering: {e}")
        st.markdown("""
        **Debug Information:**
        - Check your CSV file for formatting issues
        - Ensure consistent column structure
        - Look for special characters or encoding issues
        """)

# Display results and download button (outside the clustering logic)
if hasattr(st.session_state, 'clustering_complete') and st.session_state.clustering_complete:
    final_df = st.session_state.clustering_results
    
    # Create properly formatted CSV
    csv_data = final_df.to_csv(index=False, encoding='utf-8-sig')  # UTF-8 with BOM for Excel compatibility
    
    # Download button with unique key and proper CSV formatting
    st.download_button(
        label="📥 Download Clustered Keywords (CSV)",
        data=csv_data,
        file_name=f"clustered_keywords_{int(time.time())}.csv",
        mime="text/csv",
        key="download_clustered_csv",
        help="Download your clustered keywords as a CSV file"
    )
    
    st.markdown("### 🔍 Final Clustered Output")
    st.dataframe(final_df, use_container_width=True)
    
    # Option to clear results
    if st.button("🔄 Clear Results & Start Over"):
        if 'clustering_results' in st.session_state:
            del st.session_state.clustering_results
        if 'clustering_complete' in st.session_state:
            del st.session_state.clustering_complete
        st.rerun()
