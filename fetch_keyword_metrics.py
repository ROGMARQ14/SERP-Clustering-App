import requests
import pandas as pd
import time
import json
from typing import List, Dict
import base64

def fetch_keyword_metrics_dataforseo(keywords: List[str], dataforseo_login: str, dataforseo_password: str, location_code: int = 2840):
    """
    Fetch search volume data from DataForSEO Clickstream endpoint
    Cost: $0.15 per request (up to 1000 keywords)
    
    Args:
        keywords: List of keywords to get metrics for
        dataforseo_login: DataForSEO login email
        dataforseo_password: DataForSEO password
        location_code: Location code (2840 = United States)
    
    Returns:
        Dict mapping keywords to their metrics
    """
    
    # Setup authentication
    cred = base64.b64encode(f"{dataforseo_login}:{dataforseo_password}".encode()).decode()
    headers = {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }
    
    # DataForSEO Clickstream endpoint for bulk search volume
    url = "https://api.dataforseo.com/v3/keywords_data/clickstream_data/bulk_search_volume/live"
    
    # Process in batches of 1000 (API limit) - CRITICAL for cost optimization
    batch_size = 1000
    all_metrics = {}
    total_requests = 0
    
    # Calculate total cost estimate
    num_batches = (len(keywords) + batch_size - 1) // batch_size
    estimated_cost = num_batches * 0.15
    print(f"💰 Estimated API cost: ${estimated_cost:.2f} ({num_batches} requests × $0.15)")
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"📊 Processing batch {batch_num}/{num_batches} ({len(batch)} keywords)...")
        
        # Prepare payload for Clickstream API
        payload = [{
            "location_code": location_code,
            "keywords": batch,
            "include_clickstream_data": False  # We only need volume, not full clickstream
        }]
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            total_requests += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("tasks") and len(data["tasks"]) > 0:
                    task = data["tasks"][0]
                    
                    if task.get("status_code") == 20000 and task.get("result"):
                        # Extract metrics from result
                        for item in task["result"]:
                            keyword = item.get("keyword", "")
                            
                            # Clickstream API returns different structure
                            all_metrics[keyword] = {
                                "search_volume": item.get("search_volume", 0),
                                # Note: Clickstream endpoint doesn't provide competition/CPC
                                # You'll need Google Ads endpoint for those metrics
                                "competition": 0,  
                                "cpc": 0,
                                "last_updated": item.get("last_updated", "")
                            }
                    else:
                        print(f"⚠️ Task error in batch {batch_num}: {task.get('status_message', 'Unknown error')}")
                
                print(f"✅ Batch {batch_num} complete - fetched data for {len(batch)} keywords")
                
            else:
                print(f"❌ API error for batch {batch_num}: {response.status_code}")
                print(f"Response: {response.text[:500]}...")  # First 500 chars of error
                
        except Exception as e:
            print(f"❌ Exception in batch {batch_num}: {e}")
        
        # Rate limiting - DataForSEO allows 2000 requests/minute, but let's be conservative
        if batch_num < num_batches:
            time.sleep(0.5)
    
    print(f"💰 Total API cost: ${total_requests * 0.15:.2f} ({total_requests} requests)")
    print(f"📊 Retrieved metrics for {len(all_metrics)} keywords out of {len(keywords)} requested")
    
    return all_metrics


def fetch_keyword_metrics_dataforseo_with_competition(keywords: List[str], dataforseo_login: str, dataforseo_password: str, location_code: int = 2840):
    """
    Fetch search volume from Clickstream + competition/CPC from Google Ads endpoint
    Combined cost: $0.15 (clickstream) + $0.075 (google ads) = $0.225 per 1000 keywords
    
    Args:
        keywords: List of keywords to get metrics for
        dataforseo_login: DataForSEO login email
        dataforseo_password: DataForSEO password
        location_code: Location code (2840 = United States)
    
    Returns:
        Dict mapping keywords to their metrics including volume, competition, and CPC
    """
    
    # First get search volume from Clickstream (more accurate)
    volume_metrics = fetch_keyword_metrics_dataforseo(keywords, dataforseo_login, dataforseo_password, location_code)
    
    # Then get competition and CPC from Google Ads endpoint
    cred = base64.b64encode(f"{dataforseo_login}:{dataforseo_password}".encode()).decode()
    headers = {
        'Authorization': f'Basic {cred}',
        'Content-Type': 'application/json'
    }
    
    # Google Ads endpoint for competition and CPC
    url = "https://api.dataforseo.com/v3/keywords_data/google_ads/search_volume/live"
    
    batch_size = 1000
    num_batches = (len(keywords) + batch_size - 1) // batch_size
    print(f"\n💰 Fetching competition/CPC data - Additional cost: ${num_batches * 0.075:.2f}")
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        payload = [{
            "location_code": location_code,
            "language_code": "en",
            "keywords": batch
        }]
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("tasks") and data["tasks"][0].get("result"):
                    for item in data["tasks"][0]["result"]:
                        keyword = item.get("keyword", "")
                        
                        # Update metrics with competition and CPC
                        if keyword in volume_metrics:
                            volume_metrics[keyword].update({
                                "competition": item.get("competition", 0),
                                "competition_level": item.get("competition_level", ""),
                                "cpc": item.get("cpc", 0)
                            })
                
                print(f"✅ Competition/CPC batch {batch_num}/{num_batches} complete")
                
        except Exception as e:
            print(f"⚠️ Error fetching competition/CPC for batch {batch_num}: {e}")
        
        if batch_num < num_batches:
            time.sleep(0.5)
    
    return volume_metrics


def fetch_keyword_metrics_semrush(keywords: List[str], semrush_api_key: str, database: str = "us"):
    """
    Fetch keyword metrics from SEMrush API
    
    Args:
        keywords: List of keywords
        semrush_api_key: SEMrush API key
        database: Country database (us, uk, ca, etc.)
    
    Returns:
        Dict mapping keywords to metrics
    """
    
    base_url = "https://api.semrush.com/"
    all_metrics = {}
    
    for keyword in keywords:
        params = {
            "type": "phrase_this",
            "key": semrush_api_key,
            "phrase": keyword,
            "database": database,
            "export_columns": "Ph,Nq,Cp,Co,Nr,Td"  # Keyword, Volume, CPC, Competition, Results, Trend
        }
        
        try:
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200 and response.text.strip():
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    # Parse the response
                    headers = lines[0].split(';')
                    values = lines[1].split(';')
                    
                    data = dict(zip(headers, values))
                    
                    all_metrics[keyword] = {
                        "search_volume": int(data.get("Nq", 0)),
                        "competition": float(data.get("Co", 0)),
                        "cpc": float(data.get("Cp", 0)),
                        "keyword_difficulty": float(data.get("Td", 0)) if "Td" in data else None,
                        "serp_results": int(data.get("Nr", 0)) if "Nr" in data else None
                    }
                else:
                    all_metrics[keyword] = {"search_volume": 0, "competition": 0, "cpc": 0}
                    
            print(f"✅ Fetched metrics for: {keyword}")
            
        except Exception as e:
            print(f"❌ Error fetching {keyword}: {e}")
            all_metrics[keyword] = {"search_volume": 0, "competition": 0, "cpc": 0}
        
        # Rate limiting - adjust based on your API plan
        time.sleep(0.5)
    
    return all_metrics


def enrich_clustered_keywords(
    clustered_file: str,
    metrics: Dict[str, Dict],
    output_file: str = "enriched_clustered_keywords.csv"
):
    """
    Add metrics to clustered keywords and calculate cluster-level metrics
    
    Args:
        clustered_file: Path to clustered keywords CSV
        metrics: Dictionary of keyword metrics
        output_file: Path for enriched output
    
    Returns:
        Enriched DataFrame
    """
    
    # Load clustered keywords
    df = pd.read_csv(clustered_file)
    
    # Add individual keyword metrics
    df['Search Volume'] = df['Keyword'].map(lambda x: metrics.get(x, {}).get('search_volume', 0))
    df['Competition'] = df['Keyword'].map(lambda x: metrics.get(x, {}).get('competition', 0))
    df['CPC'] = df['Keyword'].map(lambda x: metrics.get(x, {}).get('cpc', 0))
    
    # Check if we have competition/CPC data
    has_competition_data = df['Competition'].sum() > 0 or df['CPC'].sum() > 0
    
    if has_competition_data:
        # Calculate keyword value score with competition and CPC
        df['Keyword Value'] = (
            df['Search Volume'] * 
            df['CPC'] * 
            (1 - df['Competition'])  # Higher value for lower competition
        ).round(2)
    else:
        # Simple value based on volume only
        df['Keyword Value'] = df['Search Volume']
    
    # Calculate cluster-level metrics
    if has_competition_data:
        cluster_metrics = df.groupby('Cluster Label').agg({
            'Search Volume': ['sum', 'mean', 'max'],
            'Competition': 'mean',
            'CPC': 'mean',
            'Keyword Value': 'sum',
            'Keyword': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_metrics.columns = [
            'Cluster Volume (Total)', 'Cluster Volume (Avg)', 'Cluster Volume (Max)',
            'Cluster Competition', 'Cluster CPC', 'Cluster Value', 'Cluster Size'
        ]
        
        # Calculate cluster priority score with full metrics
        cluster_metrics['Cluster Priority'] = (
            cluster_metrics['Cluster Volume (Total)'] * 
            cluster_metrics['Cluster CPC'] * 
            (1 - cluster_metrics['Cluster Competition']) *
            (1 + cluster_metrics['Cluster Size'] / 100)  # Bonus for larger clusters
        ).round(0)
    else:
        # Simplified metrics without competition/CPC
        cluster_metrics = df.groupby('Cluster Label').agg({
            'Search Volume': ['sum', 'mean', 'max'],
            'Keyword Value': 'sum',
            'Keyword': 'count'
        }).round(2)
        
        # Flatten column names
        cluster_metrics.columns = [
            'Cluster Volume (Total)', 'Cluster Volume (Avg)', 'Cluster Volume (Max)',
            'Cluster Value', 'Cluster Size'
        ]
        
        # Simplified priority based on volume and size only
        cluster_metrics['Cluster Priority'] = (
            cluster_metrics['Cluster Volume (Total)'] * 
            (1 + cluster_metrics['Cluster Size'] / 100)
        ).round(0)
    
    # Merge cluster metrics back to main dataframe
    merge_columns = ['Cluster Priority', 'Cluster Volume (Total)']
    if has_competition_data:
        merge_columns.append('Cluster Competition')
    
    df = df.merge(
        cluster_metrics[merge_columns], 
        left_on='Cluster Label', 
        right_index=True, 
        how='left'
    )
    
    # Sort by cluster priority and then by search volume within clusters
    df = df.sort_values(
        by=['Cluster Priority', 'Cluster Label', 'Search Volume'], 
        ascending=[False, True, False]
    )
    
    # Reorder columns for better readability
    if has_competition_data:
        column_order = [
            'Cluster Label', 'Hub', 'Keyword', 
            'Search Volume', 'Competition', 'CPC', 'Keyword Value',
            'Cluster Size', 'Cluster Priority', 'Cluster Volume (Total)', 
            'Cluster Competition', 'URLs'
        ]
    else:
        column_order = [
            'Cluster Label', 'Hub', 'Keyword', 
            'Search Volume', 'Keyword Value',
            'Cluster Size', 'Cluster Priority', 'Cluster Volume (Total)', 
            'URLs'
        ]
    
    # Only include columns that exist
    final_columns = [col for col in column_order if col in df.columns]
    df = df[final_columns]
    
    # Save enriched data
    df.to_csv(output_file, index=False)
    
    print(f"✅ Enriched data saved to {output_file}")
    print(f"📊 Top 5 Priority Clusters:")
    print(cluster_metrics.nlargest(5, 'Cluster Priority')[['Cluster Priority', 'Cluster Volume (Total)', 'Cluster Size']])
    
    return df


# Example usage function
def fetch_and_enrich(clustered_file: str, api_service: str, api_credentials: Dict):
    """
    Main function to fetch metrics and enrich clustered keywords
    
    Args:
        clustered_file: Path to your clustered keywords CSV
        api_service: 'dataforseo' or 'semrush'
        api_credentials: Dict with API credentials
    """
    
    # Load keywords from clustered file
    df = pd.read_csv(clustered_file)
    keywords = df['Keyword'].unique().tolist()
    
    print(f"📊 Fetching metrics for {len(keywords)} keywords...")
    
    # Fetch metrics based on selected service
    if api_service == 'dataforseo':
        metrics = fetch_keyword_metrics_dataforseo(
            keywords,
            api_credentials['login'],
            api_credentials['password'],
            api_credentials.get('location_code', 2840)
        )
    elif api_service == 'semrush':
        metrics = fetch_keyword_metrics_semrush(
            keywords,
            api_credentials['api_key'],
            api_credentials.get('database', 'us')
        )
    else:
        raise ValueError(f"Unknown API service: {api_service}")
    
    # Save metrics for debugging/caching
    with open('keyword_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Enrich clustered keywords with metrics
    enriched_df = enrich_clustered_keywords(clustered_file, metrics)
    
    return enriched_df


if __name__ == "__main__":
    # Example usage
    api_credentials = {
        # For DataForSEO
        'login': 'your_email@domain.com',
        'password': 'your_password',
        'location_code': 2840  # US
        
        # For SEMrush
        # 'api_key': 'your_semrush_api_key',
        # 'database': 'us'
    }
    
    enriched_df = fetch_and_enrich(
        'final_clustered_keywords.csv',
        'dataforseo',  # or 'semrush'
        api_credentials
    )
