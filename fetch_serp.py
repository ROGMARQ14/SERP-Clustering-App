import asyncio
import aiohttp
import pandas as pd
import json
import time
import nest_asyncio
import requests
from typing import List, Dict, Tuple

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

def detect_keyword_column(df: pd.DataFrame) -> str:
    """Detect keyword column from various naming conventions"""
    # Exact matches in priority order
    exact_matches = ["Keyword", "keyword", "Keywords", "keywords", "Query", "query", "Queries", "queries"]
    
    for col in exact_matches:
        if col in df.columns:
            print(f"✅ Detected keyword column: '{col}'")
            return col
    
    # Case-insensitive matches
    df_columns_lower = [col.lower() for col in df.columns]
    for target in ["keyword", "keywords", "query", "queries"]:
        for i, col_lower in enumerate(df_columns_lower):
            if target == col_lower:
                actual_col = df.columns[i]
                print(f"✅ Detected keyword column (case-insensitive): '{actual_col}'")
                return actual_col
    
    # Partial matches for columns containing keyword-related terms
    keyword_terms = ["keyword", "query", "search", "term"]
    for col in df.columns:
        if any(term in col.lower() for term in keyword_terms):
            print(f"✅ Detected keyword column (partial match): '{col}'")
            return col
    
    # Fallback to first column
    if len(df.columns) > 0:
        first_col = df.columns[0]
        print(f"⚠️ Using first column as keywords: '{first_col}'")
        return first_col
    
    raise ValueError("No columns found in CSV file")

class AsyncSerpFetcher:
    """Asynchronous SERP fetcher with rate limiting"""
    
    def __init__(self, serper_api_key: str, max_concurrent: int = 5):
        self.serper_api_key = serper_api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.base_url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_requests = 0
        self.start_time = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.headers
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def fetch_single_keyword(self, keyword: str) -> Tuple[str, List[str]]:
        """Fetch SERP results for a single keyword with error handling"""
        async with self.semaphore:  # Rate limiting
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    payload = {"q": keyword}
                    
                    async with self.session.post(self.base_url, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            organic_results = data.get("organic", [])
                            urls = [item.get("link") for item in organic_results][:10]
                            urls = [url for url in urls if url is not None]
                            
                            self.completed_requests += 1
                            
                            # Progress update
                            if self.total_requests > 0:
                                progress = (self.completed_requests + self.failed_requests) / self.total_requests
                                print(f"Progress: {progress:.1%} - {keyword}")
                            
                            return keyword, urls
                            
                        elif response.status == 429:
                            # Rate limited - exponential backoff
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"Rate limited for '{keyword}', waiting {wait_time:.1f}s")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        else:
                            print(f"HTTP {response.status} for '{keyword}'")
                            
                except Exception as e:
                    print(f"Exception for '{keyword}': {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
            
            # All retries failed
            self.failed_requests += 1
            print(f"❌ Failed to fetch SERP for: {keyword}")
            return keyword, []
    
    async def fetch_batch(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Fetch SERP results for all keywords concurrently"""
        self.total_requests = len(keywords)
        self.completed_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        
        print(f"🚀 Starting async SERP fetch for {len(keywords)} keywords")
        print(f"⚡ Using {self.max_concurrent} concurrent connections")
        
        # Create tasks for all keywords
        tasks = [self.fetch_single_keyword(keyword) for keyword in keywords]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        serp_data = {}
        for result in results:
            if not isinstance(result, Exception):
                keyword, urls = result
                serp_data[keyword] = urls
        
        # Performance summary
        elapsed_time = time.time() - self.start_time
        total_processed = self.completed_requests + self.failed_requests
        
        print(f"\n📊 Performance Summary:")
        print(f"   • Total time: {elapsed_time:.1f} seconds")
        print(f"   • Success rate: {(self.completed_requests/total_processed)*100:.1f}%")
        print(f"   • Average time per keyword: {elapsed_time/total_processed:.2f} seconds")
        print(f"   • Speed improvement: ~{1.2/max(elapsed_time/total_processed, 0.01):.1f}x faster")
        
        return serp_data

def fetch_serp_data_sync(keywords: List[str], serper_api_key: str) -> Dict[str, List[str]]:
    """Fallback synchronous SERP fetcher - identical to original"""
    print("⚠️ Using fallback synchronous mode")
    
    SERPER_API_URL = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    def fetch_serp_results(keyword: str) -> List[str]:
        payload = {"q": keyword}
        try:
            response = requests.post(SERPER_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                organic_results = data.get("organic", [])
                urls = [item.get("link") for item in organic_results][:10]
                return [url for url in urls if url is not None]
            else:
                print(f"Error fetching for keyword '{keyword}': {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception fetching for keyword '{keyword}': {e}")
            return []
    
    results = {}
    for idx, keyword in enumerate(keywords):
        print(f"[{idx+1}/{len(keywords)}] Fetching SERP for: {keyword}")
        results[keyword] = fetch_serp_results(keyword)
        if idx < len(keywords) - 1:
            time.sleep(1.2)  # Original rate limiting
    
    return results

def fetch_serp_data(serper_api_key: str):
    """
    Main function - maintains exact same signature as original
    Automatically detects keyword column and uses async processing with sync fallback
    """
    try:
        # Load keywords with flexible column detection
        print("📂 Loading keywords from CSV...")
        keywords_df = pd.read_csv("keywords.csv")
        
        # Detect the correct keyword column
        keyword_column = detect_keyword_column(keywords_df)
        
        # Extract keywords using the detected column
        keywords = keywords_df[keyword_column].dropna().tolist()
        
        if not keywords:
            raise ValueError("No keywords found in the CSV file")
            
        print(f"📋
