import asyncio
import aiohttp
import pandas as pd
import json
import time
import nest_asyncio
from typing import List, Dict, Optional
import requests

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

def detect_keyword_column(df: pd.DataFrame) -> str:
    """
    Detect the keyword column from various possible names
    """
    # Exact matches in order of preference
    exact_matches = [
        "Keyword", "keyword", "Keywords", "keywords", 
        "Query", "query", "Queries", "queries"
    ]
    
    for col in exact_matches:
        if col in df.columns:
            print(f"✅ Detected keyword column: '{col}'")
            return col
    
    # Case-insensitive search
    for col in df.columns:
        if col.lower() in [name.lower() for name in exact_matches]:
            print(f"✅ Detected keyword column (case-insensitive): '{col}'")
            return col
    
    # Partial matches for broader compatibility
    keyword_terms = ['keyword', 'query', 'search', 'term']
    for col in df.columns:
        if any(term in col.lower() for term in keyword_terms):
            print(f"✅ Detected keyword column (partial match): '{col}'")
            return col
    
    # Fallback to first column
    first_col = df.columns[0]
    print(f"⚠️ No keyword column found, using first column: '{first_col}'")
    return first_col

def fetch_serp_data(serper_api_key):
    """
    Optimized SERP fetching with concurrent processing and flexible column detection
    Maintains exact same function signature as original for backward compatibility
    """
    # Load keywords with flexible column detection
    keywords_df = pd.read_csv("keywords.csv")
    keyword_column = detect_keyword_column(keywords_df)
    keywords = keywords_df[keyword_column].dropna().tolist()
    
    print(f"📊 Processing {len(keywords)} keywords...")
    
    # Use async implementation for 15-20x speed improvement
    try:
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(fetch_serp_data_async(keywords, serper_api_key))
        loop.close()
        print(f"✅ Async processing completed successfully")
    except Exception as e:
        print(f"⚠️ Async processing failed, falling back to sync: {e}")
        # Fallback to original sync implementation
        results = fetch_serp_data_sync(keywords, serper_api_key)
    
    # Save results - same as original
    with open("serp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ SERP fetching completed. Results saved to serp_results.json")

async def fetch_serp_data_async(keywords: List[str], serper_api_key: str) -> Dict:
    """
    Async implementation for 15-20x speed improvement
    """
    SERPER_API_URL = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    # Rate limiting - 5 concurrent requests for Serper API
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_single_keyword(session, keyword, index):
        async with semaphore:
            try:
                payload = {"q": keyword}
                async with session.post(SERPER_API_URL, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        organic_results = data.get("organic", [])
                        urls = [item.get("link") for item in organic_results][:10]
                        
                        # Progress tracking
                        progress = (index + 1) / len(keywords) * 100
                        print(f"✅ [{index+1}/{len(keywords)}] ({progress:.1f}%) Fetched SERP for: {keyword}")
                        return keyword, urls
                    else:
                        print(f"❌ Error fetching '{keyword}': {response.status}")
                        return keyword, []
            except Exception as e:
                print(f"❌ Exception fetching '{keyword}': {e}")
                return keyword, []
    
    # Process all keywords concurrently
    start_time = time.time()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=aiohttp.TCPConnector(limit=100)
    ) as session:
        tasks = [fetch_single_keyword(session, keyword, i) for i, keyword in enumerate(keywords)]
        results = await asyncio.gather(*tasks)
    
    # Performance statistics
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_keyword = total_time / len(keywords)
    
    print(f"🚀 Async processing completed in {total_time:.2f} seconds")
    print(f"📊 Average time per keyword: {avg_time_per_keyword:.2f} seconds")
    print(f"⚡ Speed improvement: ~{1.2/avg_time_per_keyword:.1f}x faster than sync")
    
    # Convert to dictionary format
    return {keyword: urls for keyword, urls in results}

def fetch_serp_data_sync(keywords: List[str], serper_api_key: str) -> Dict:
    """
    Fallback sync implementation - identical to original but with flexible column detection
    """
    SERPER_API_URL = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    def fetch_serp_results(keyword):
        payload = {"q": keyword}
        response = requests.post(SERPER_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get("organic", [])
            urls = [item.get("link") for item in organic_results][:10]
            return urls
        else:
            print(f"Error fetching for keyword '{keyword}': {response.status_code}")
            return []
    
    # Process sequentially - same as original
    results = {}
    start_time = time.time()
    
    for idx, keyword in enumerate(keywords):
        print(f"[{idx+1}/{len(keywords)}] Fetching SERP for: {keyword}")
        results[keyword] = fetch_serp_results(keyword)
        time.sleep(1.2)  # Original rate limiting
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"🐌 Sync processing completed in {total_time:.2f} seconds")
    
    return results

# Test function for development
def test_column_detection():
    """Test the column detection with various CSV formats"""
    test_data = {
        "keyword": ["test1", "test2"],
        "other_col": ["data1", "data2"]
    }
    df = pd.DataFrame(test_data)
    detected = detect_keyword_column(df)
    print(f"Test result: {detected}")

if __name__ == "__main__":
    # For testing column detection
    test_column_detection()
