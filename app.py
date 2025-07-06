import streamlit as st
import pandas as pd
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules
try:
    from fetch_serp import fetch_serp_data
    from cluster_keywords import cluster_keywords_by_serp
    from label_clusters import label_clusters
    from fetch_keyword_metrics import (
        fetch_and_enrich, 
        fetch_keyword_metrics_dataforseo, 
        fetch_keyword_metrics_dataforseo_with_competition,  # Fixed: Added this import
        fetch_keyword_metrics_semrush, 
        enrich_clustered_keywords
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Required modules not found: {e}")
    st.error("Please ensure all .py files are in the same directory.")

# --------------------
# Streamlit UI Setup
# --------------------
st.set_page_config(page_title="SERP-Based Keyword Clustering Tool", layout="wide")
st.title("🔍 SERP-Based Keyword Clustering Tool")
st.markdown("Upload keywords and cluster them based on actual Google search result overlap.")

# Get environment variables with defaults
default_serper_key = os.getenv("SERPER_API_KEY", "")
default_openai_key = os.getenv("OPENAI_API_KEY", "")
default_dataforseo_login = os.getenv("DATAFORSEO_LOGIN", "")
default_dataforseo_password = os.getenv("DATAFORSEO_PASSWORD", "")
default_dataforseo_location = int(os.getenv("DATAFORSEO_LOCATION_CODE", "2840"))
default_semrush_key = os.getenv("SEMRUSH_API_KEY", "")
default_semrush_db = os.getenv("SEMRUSH_DATABASE", "us")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys with environment variable defaults
    serper_api_key = st.text_input(
        "Serper API Key", 
        type="password", 
        value=default_serper_key,
        help="Get your API key from https://serper.dev"
    )
    
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        value=default_openai_key
    )
    
    # Show if environment variables are loaded
    if default_serper_key or default_openai_key:
        st.success("✅ API keys loaded from environment")
    
    # Keyword metrics API selection
    st.subheader("📊 Keyword Metrics (Optional)")
    metrics_service = st.selectbox(
        "Metrics API Service",
        ["None", "DataForSEO", "SEMrush"],
        help="Add search volume and competition data"
    )
    
    if metrics_service == "DataForSEO":
        dataforseo_login = st.text_input(
            "DataForSEO Login Email", 
            type="password",
            value=default_dataforseo_login
        )
        dataforseo_password = st.text_input(
            "DataForSEO Password", 
            type="password",
            value=default_dataforseo_password
        )
        location_code = st.number_input(
            "Location Code", 
            value=default_dataforseo_location, 
            help="2840=US, 2826=UK, 2124=CA"
        )
        
        # Option to include competition/CPC data
        include_competition = st.checkbox(
            "Include Competition & CPC data", 
            value=False,
            help="Adds $0.075 per 1000 keywords (Google Ads endpoint)"
        )
        
    elif metrics_service == "SEMrush":
        semrush_api_key = st.text_input(
            "SEMrush API Key", 
            type="password",
            value=default_semrush_key
        )
        semrush_database = st.selectbox(
            "Database", 
            ["us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "mx"],
            index=["us", "uk", "ca", "au", "de", "fr", "es", "it", "br", "mx"].index(default_semrush_db)
        )
    
    with st.expander("Advanced Settings"):
        similarity_threshold = st.slider(
            "SERP Similarity Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.3,
            step=0.05,
            help="Higher = stricter clustering (fewer, tighter clusters)"
        )
        
        openai_model = st.selectbox(
            "OpenAI Model for Labels",
            ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
        
        label_temperature = st.slider(
            "Label Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )

# Main area
uploaded_file = st.file_uploader("Upload keywords.csv", type=['csv'])

# Show cost estimate for DataForSEO
if metrics_service == "DataForSEO" and uploaded_file:
    try:
        df_temp = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)  # Reset file pointer
        num_keywords = len(df_temp)
        num_batches = (num_keywords + 999) // 1000
        
        clickstream_cost = num_batches * 0.15
        google_ads_cost = num_batches * 0.075 if include_competition else 0
        total_cost = clickstream_cost + google_ads_cost
        
        st.info(f"""
        **💰 Estimated DataForSEO Cost:**
        - Keywords: {num_keywords}
        - Clickstream API: ${clickstream_cost:.2f} ({num_batches} batches × $0.15)
        {f'- Google Ads API: ${google_ads_cost:.2f} ({num_batches} batches × $0.075)' if include_competition else ''}
        - **Total: ${total_cost:.2f}**
        """)
    except:
        pass

# Information boxes
col1, col2 = st.columns(2)
with col1:
    st.info("""
    **How it works:**
    1. Fetches top 10 Google results for each keyword
    2. Clusters keywords that share similar search results
    3. Labels clusters based on search intent
    """)

with col2:
    st.warning("""
    **Note:** 
    - SERP fetching takes ~1.2 seconds per keyword
    - 100 keywords ≈ 2 minutes processing time
    """)

# Process button
if st.button("🚀 Start Clustering", type="primary") and uploaded_file and serper_api_key and openai_api_key:
    
    # Save uploaded file temporarily
    with open("keywords.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Check if keywords.csv exists and has content
        df_check = pd.read_csv("keywords.csv")
        total_keywords = len(df_check)
        st.success(f"✅ Loaded {total_keywords} keywords from CSV")
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_estimate = st.empty()
        
        # Step 1: Fetch SERP data
        status_text.text("🔍 Step 1/3: Fetching SERP data...")
        time_estimate.text(f"Estimated time: {total_keywords * 1.2 / 60:.1f} minutes")
        
        fetch_serp_data(serper_api_key)
        progress_bar.progress(0.4)
        
        # Check if SERP results were generated
        if os.path.exists("serp_results.json"):
            with open("serp_results.json", "r") as f:
                serp_data = json.load(f)
            st.success(f"✅ Fetched SERP data for {len(serp_data)} keywords")
        else:
            st.error("❌ Failed to fetch SERP data")
            st.stop()
        
        # Step 2: Cluster keywords
        status_text.text("🧮 Step 2/3: Clustering keywords based on SERP overlap...")
        
        clustered_df = cluster_keywords_by_serp(
            input_file="serp_results.json",
            output_file="clustered_keywords.csv",
            similarity_threshold=similarity_threshold
        )
        progress_bar.progress(0.7)
        
        # Display clustering stats
        num_clusters = clustered_df['Hub'].nunique()
        avg_cluster_size = len(clustered_df) / num_clusters
        st.success(f"✅ Created {num_clusters} clusters (avg size: {avg_cluster_size:.1f} keywords)")
        
        # Step 3: Generate labels
        status_text.text("🏷️ Step 3/4: Generating intelligent cluster labels...")
        
        final_df = label_clusters(
            openai_api_key=openai_api_key,
            input_file="clustered_keywords.csv",
            output_file="final_clustered_keywords.csv",
            model=openai_model,
            temperature=label_temperature
        )
        progress_bar.progress(0.85)
        
        # Step 4: Fetch keyword metrics (if selected)
        if metrics_service != "None":
            status_text.text("📊 Step 4/4: Fetching keyword metrics and calculating cluster priority...")
            
            try:
                # Prepare API credentials
                api_credentials = {}
                if metrics_service == "DataForSEO":
                    api_credentials = {
                        'login': dataforseo_login,
                        'password': dataforseo_password,
                        'location_code': location_code
                    }
                    # Fetch metrics directly
                    keywords = final_df['Keyword'].unique().tolist()
                    
                    if include_competition:
                        # Use combined endpoint for volume + competition/CPC
                        metrics = fetch_keyword_metrics_dataforseo_with_competition(
                            keywords, dataforseo_login, dataforseo_password, location_code
                        )
                    else:
                        # Use Clickstream endpoint for volume only
                        metrics = fetch_keyword_metrics_dataforseo(
                            keywords, dataforseo_login, dataforseo_password, location_code
                        )
                    
                elif metrics_service == "SEMrush":
                    api_credentials = {
                        'api_key': semrush_api_key,
                        'database': semrush_database
                    }
                    # Fetch metrics directly
                    keywords = final_df['Keyword'].unique().tolist()
                    metrics = fetch_keyword_metrics_semrush(keywords, semrush_api_key, semrush_database)
                
                # Enrich the dataframe with metrics
                final_df = enrich_clustered_keywords(
                    "final_clustered_keywords.csv",
                    metrics,
                    "enriched_clustered_keywords.csv"
                )
                
                st.success("✅ Keyword metrics added successfully!")
                
            except Exception as e:
                st.warning(f"⚠️ Could not fetch metrics: {e}")
                st.info("Continuing without keyword metrics...")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Clustering complete!")
        
        # Display results
        st.header("📊 Clustering Results")
        
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
        
        # Additional metrics if available
        if 'Search Volume' in final_df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                total_volume = final_df['Search Volume'].sum()
                st.metric("Total Search Volume", f"{total_volume:,}")
            with col2:
                avg_competition = final_df['Competition'].mean()
                st.metric("Avg Competition", f"{avg_competition:.2f}")
            with col3:
                total_value = final_df['Keyword Value'].sum() if 'Keyword Value' in final_df.columns else 0
                st.metric("Total Keyword Value", f"${total_value:,.0f}")
        
        # Download button
        csv_data = final_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results (CSV)",
            data=csv_data,
            file_name="enriched_clustered_keywords.csv" if metrics_service != "None" else "final_clustered_keywords.csv",
            mime="text/csv"
        )
        
        # Filter options
        st.subheader("🔍 Explore Clusters")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_cluster = st.selectbox(
                "Select Cluster",
                ["All Clusters"] + sorted(final_df['Cluster Label'].unique().tolist())
            )
        with col2:
            min_size_filter = st.number_input("Min Cluster Size", min_value=1, value=1)
        
        # Filter dataframe
        display_df = final_df.copy()
        if selected_cluster != "All Clusters":
            display_df = display_df[display_df['Cluster Label'] == selected_cluster]
        
        # Apply size filter
        cluster_sizes = display_df.groupby('Cluster Label')['Keyword'].count()
        valid_clusters = cluster_sizes[cluster_sizes >= min_size_filter].index
        display_df = display_df[display_df['Cluster Label'].isin(valid_clusters)]
        
        # Show filtered results
        if 'Cluster Priority' in display_df.columns:
            # Show with priority if metrics available
            display_columns = ['Cluster Label', 'Hub', 'Keyword', 'Search Volume', 'Competition', 'CPC', 'Cluster Priority', 'Cluster Size']
            display_columns = [col for col in display_columns if col in display_df.columns]
        else:
            # Show basic columns
            display_columns = ['Cluster Label', 'Hub', 'Keyword', 'Cluster Size']
        
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            height=500
        )
        
        # Cluster summary
        with st.expander("📈 Cluster Summary"):
            if 'Cluster Priority' in final_df.columns:
                # Enhanced summary with metrics
                summary_df = final_df.groupby('Cluster Label').agg({
                    'Keyword': 'count',
                    'Hub': 'first',
                    'Search Volume': 'sum',
                    'Competition': 'mean',
                    'Cluster Priority': 'first'
                }).rename(columns={
                    'Keyword': 'Size',
                    'Search Volume': 'Total Volume'
                }).sort_values('Cluster Priority', ascending=False)
                
                summary_df['Competition'] = summary_df['Competition'].round(2)
                summary_df['Cluster Priority'] = summary_df['Cluster Priority'].round(0)
            else:
                # Basic summary
                summary_df = final_df.groupby('Cluster Label').agg({
                    'Keyword': 'count',
                    'Hub': 'first'
                }).rename(columns={'Keyword': 'Size'}).sort_values('Size', ascending=False)
            
            st.dataframe(summary_df, use_container_width=True)
            
            # Priority insights if metrics available
            if 'Cluster Priority' in final_df.columns:
                st.subheader("🎯 Top Priority Clusters")
                top_clusters = summary_df.nlargest(5, 'Cluster Priority')
                
                for idx, (cluster_label, row) in enumerate(top_clusters.iterrows(), 1):
                    st.write(f"**{idx}. {cluster_label}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"Volume: {row['Total Volume']:,}")
                    with col2:
                        st.write(f"Competition: {row['Competition']:.2f}")
                    with col3:
                        st.write(f"Priority Score: {row['Cluster Priority']:,.0f}")
        
        # Show sample URLs for selected cluster
        if selected_cluster != "All Clusters":
            with st.expander("🔗 Sample URLs from this cluster"):
                cluster_data = display_df[display_df['Cluster Label'] == selected_cluster].iloc[0]
                urls = cluster_data['URLs'].split('\n')[:5]  # Show top 5 URLs
                for url in urls:
                    st.write(f"- {url}")
        
        # Clean up temporary files
        for file in ['keywords.csv', 'serp_results.json', 'clustered_keywords.csv']:
            if os.path.exists(file):
                os.remove(file)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
        
        # Clean up on error
        for file in ['keywords.csv', 'serp_results.json', 'clustered_keywords.csv']:
            if os.path.exists(file):
                os.remove(file)

else:
    if not MODULES_AVAILABLE:
        st.error("Cannot proceed without required modules. Please ensure all Python files are in the same directory.")
    elif not uploaded_file:
        st.info("👆 Please upload a CSV file containing keywords to begin.")
    elif not serper_api_key:
        st.warning("🔑 Please enter your Serper API key in the sidebar.")
    elif not openai_api_key:
        st.warning("🔑 Please enter your OpenAI API key in the sidebar.")
