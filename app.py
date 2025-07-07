import streamlit as st
import pandas as pd
import json
import os
import time
from pathlib import Path

# Import optimized modules
try:
    from fetch_serp import fetch_serp_data
    from cluster_keywords import cluster_keywords_by_serp
    from label_clusters import label_clusters
    from fetch_keyword_metrics import (
        fetch_keyword_metrics_dataforseo_with_competition,
        fetch_keyword_metrics_semrush,
        enrich_clustered_keywords
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Required modules not found: {e}")

# --------------------
# Configuration with Streamlit Secrets
# --------------------

def load_configuration():
    """Load configuration from Streamlit secrets with fallbacks"""
    try:
        # Primary: Use Streamlit secrets
        config = {
            'serper_api_key': st.secrets.get("SERPER_API_KEY", ""),
            'openai_api_key': st.secrets.get("OPENAI_API_KEY", ""),
            'dataforseo_login': st.secrets.get("dataforseo", {}).get("login", ""),
            'dataforseo_password': st.secrets.get("dataforseo", {}).get("password", ""),
            'dataforseo_location': st.secrets.get("dataforseo", {}).get("location_code", 2840),
            'semrush_key': st.secrets.get("semrush", {}).get("api_key", ""),
            'semrush_db': st.secrets.get("semrush", {}).get("database", "us"),
            # App settings
            'similarity_threshold': st.secrets.get("app_settings", {}).get("default_similarity_threshold", 0.3),
            'openai_model': st.secrets.get("app_settings", {}).get("default_openai_model", "gpt-4o-mini"),
            'temperature': st.secrets.get("app_settings", {}).get("default_temperature", 0.3)
        }

        if config['serper_api_key'] and config['openai_api_key']:
            st.success("✅ Configuration loaded from Streamlit secrets")
        else:
            st.warning("⚠️ Some API keys missing from secrets")

    except Exception as e:
        # Fallback: Use environment variables (for local development)
        st.warning("Using environment variables fallback")
        config = {
            'serper_api_key': os.getenv("SERPER_API_KEY", ""),
            'openai_api_key': os.getenv("OPENAI_API_KEY", ""),
            'dataforseo_login': os.getenv("DATAFORSEO_LOGIN", ""),
            'dataforseo_password': os.getenv("DATAFORSEO_PASSWORD", ""),
            'dataforseo_location': int(os.getenv("DATAFORSEO_LOCATION_CODE", "2840")),
            'semrush_key': os.getenv("SEMRUSH_API_KEY", ""),
            'semrush_db': os.getenv("SEMRUSH_DATABASE", "us"),
            'similarity_threshold': 0.3,
            'openai_model': "gpt-4o-mini",
            'temperature': 0.3
        }

    return config

# --------------------
# Streamlit UI Setup
# --------------------

st.set_page_config(
    page_title="SERP-Based Keyword Clustering Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 SERP-Based Keyword Clustering Tool (Optimized)")
st.markdown("Upload keywords and cluster them based on actual Google search result overlap with enhanced performance.")

# Load configuration
config = load_configuration()

# --------------------
# Sidebar Configuration
# --------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # Show configuration status
    with st.expander("🔐 Security Status"):
        if config['serper_api_key']:
            st.success("✅ Serper API Key loaded")
        else:
            st.error("❌ Serper API Key missing")

        if config['openai_api_key']:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key missing")

    # API Key overrides (only show if not in secrets)
    if not config['serper_api_key']:
        config['serper_api_key'] = st.text_input(
            "Serper API Key",
            type="password",
            help="Get your API key from https://serper.dev"
        )

    if not config['openai_api_key']:
        config['openai_api_key'] = st.text_input(
            "OpenAI API Key",
            type="password"
        )

    # Performance settings
    st.subheader("🚀 Performance Settings")
    use_async_processing = st.checkbox(
        "Use Async Processing", 
        value=True,
        help="Enable concurrent API requests for faster processing"
    )

    use_advanced_clustering = st.checkbox(
        "Use Advanced Clustering",
        value=True, 
        help="Use TF-IDF vectorization for better semantic clustering"
    )

    max_concurrent_requests = st.slider(
        "Max Concurrent Requests",
        min_value=1,
        max_value=20,
        value=10,
        help="Number of simultaneous API requests (higher = faster but more resource intensive)"
    )

    # Keyword metrics configuration
    st.subheader("📊 Keyword Metrics")
    metrics_service = st.selectbox(
        "Metrics API Service",
        ["None", "DataForSEO", "SEMrush"],
        help="Add search volume and competition data"
    )

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        similarity_threshold = st.slider(
            "SERP Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=config['similarity_threshold'],
            step=0.05,
            help="Higher = stricter clustering"
        )

        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )

        enable_caching = st.checkbox(
            "Enable Caching",
            value=True,
            help="Cache results to speed up repeated operations"
        )

# --------------------
# Main Processing Logic
# --------------------

uploaded_file = st.file_uploader("Upload keywords.csv", type=['csv'])

# Performance monitoring
if st.sidebar.checkbox("Show Performance Metrics", value=False):
    import psutil
    import time

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
    with col2:
        st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
    with col3:
        st.metric("Active Processes", len(psutil.pids()))

# Cost estimation for DataForSEO
if metrics_service == "DataForSEO" and uploaded_file:
    try:
        df_temp = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset file pointer
        num_keywords = len(df_temp)
        num_batches = (num_keywords + 999) // 1000
        estimated_cost = num_batches * 0.15

        st.info(f"""
        💰 **Estimated DataForSEO Cost:**
        - Keywords: {num_keywords:,}
        - Batches: {num_batches}
        - **Estimated Cost: ${estimated_cost:.2f}**
        """)
    except:
        pass

# Process button with enhanced error handling
if st.button("🚀 Start Optimized Clustering", type="primary") and uploaded_file:
    if not config['serper_api_key'] or not config['openai_api_key']:
        st.error("❌ Please provide both Serper and OpenAI API keys")
        st.stop()

    # Save uploaded file
    with open("keywords.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        start_time = time.time()

        # Load and validate keywords
        df_check = pd.read_csv("keywords.csv")
        total_keywords = len(df_check)

        if total_keywords == 0:
            st.error("No keywords found in uploaded file")
            st.stop()

        st.success(f"✅ Loaded {total_keywords:,} keywords from CSV")

        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📊 Processing", "📈 Results", "🔍 Analysis"])

        with tab1:
            # Step 1: Fetch SERP data
            st.subheader("🔍 Step 1: Fetching SERP Data")
            if use_async_processing:
                st.info("Using optimized async processing for faster results")
                serp_results = fetch_serp_data_optimized(df_check["Keyword"].tolist())
            else:
                st.info("Using standard processing")
                # Use original function with progress tracking
                with st.spinner("Fetching SERP data..."):
                    from fetch_serp import fetch_serp_data
                    fetch_serp_data(config['serper_api_key'])
                    with open("serp_results.json", "r") as f:
                        serp_results = json.load(f)

            if not serp_results:
                st.error("❌ Failed to fetch SERP data")
                st.stop()

            # Step 2: Cluster keywords
            st.subheader("🧮 Step 2: Clustering Keywords")

            if use_advanced_clustering:
                st.info("Using advanced TF-IDF clustering algorithm")
            else:
                st.info("Using URL overlap clustering algorithm")

            clustered_df = optimized_cluster_keywords_by_serp(
                input_file="serp_results.json",
                output_file="clustered_keywords.csv", 
                similarity_threshold=similarity_threshold,
                use_advanced_clustering=use_advanced_clustering
            )

            num_clusters = clustered_df['Hub'].nunique()
            avg_cluster_size = len(clustered_df) / num_clusters if num_clusters > 0 else 0

            st.success(f"✅ Created {num_clusters} clusters (avg size: {avg_cluster_size:.1f})")

            # Step 3: Generate labels
            st.subheader("🏷️ Step 3: Generating Cluster Labels")
            final_df = label_clusters(
                openai_api_key=config['openai_api_key'],
                input_file="clustered_keywords.csv",
                output_file="final_clustered_keywords.csv",
                model=openai_model,
                temperature=config['temperature']
            )

            # Calculate total processing time
            total_time = time.time() - start_time
            st.success(f"✅ Processing completed in {total_time:.1f} seconds!")

        with tab2:
            # Display results in the second tab
            st.subheader("📊 Clustering Results")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Keywords", len(final_df))
            with col2:
                st.metric("Clusters Created", final_df['Cluster Label'].nunique())
            with col3:
                st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
            with col4:
                coverage = (len(final_df) / total_keywords * 100)
                st.metric("Coverage", f"{coverage:.1f}%")

            # Download button
            csv_data = final_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_data,
                file_name="clustered_keywords_optimized.csv",
                mime="text/csv"
            )

            # Interactive data explorer
            st.subheader("🔍 Explore Clusters")
            selected_cluster = st.selectbox(
                "Select Cluster",
                ["All Clusters"] + sorted(final_df['Cluster Label'].unique().tolist())
            )

            if selected_cluster != "All Clusters":
                display_df = final_df[final_df['Cluster Label'] == selected_cluster]
            else:
                display_df = final_df

            st.dataframe(display_df, use_container_width=True, height=400)

        with tab3:
            # Analysis tab
            st.subheader("📈 Cluster Analysis")

            # Performance metrics
            processing_speed = total_keywords / total_time if total_time > 0 else 0
            st.metric("Processing Speed", f"{processing_speed:.1f} keywords/second")

            # Cluster size distribution
            cluster_sizes = final_df.groupby('Cluster Label').size()
            st.bar_chart(cluster_sizes.head(20))

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

    finally:
        # Clean up temporary files
        for file in ['keywords.csv', 'serp_results.json', 'clustered_keywords.csv']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass

elif not MODULES_AVAILABLE:
    st.error("❌ Cannot proceed without required modules")
elif not uploaded_file:
    st.info("👆 Please upload a CSV file containing keywords to begin")
else:
    st.warning("🔑 Please ensure all required API keys are configured")
