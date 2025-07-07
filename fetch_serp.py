import asyncio
import aiohttp
import pandas as pd
import json
import time
from typing import List, Dict
import streamlit as st
from datetime import datetime

async def fetch_serp_results_async(session: aiohttp.ClientSession, keyword: str, serper_api_key: str, semaphore: asyncio.Semaphore) -> tuple:
    """
    Fetch SERP results for a single keyword with rate limiting and error handling
    """
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": keyword}
    
    async with semaphore:  # Rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        organic_results = data.get("organic", [])
                        urls = [item.get("link") for item in organic_results][:10]
                        return keyword, urls
                    elif response.status == 429:  # Rate limited
                        wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"⚠️ Error {response.status} for keyword '{keyword}' (attempt {attempt + 1})")
                        if attempt == max_retries - 1:
                            return keyword, []
                        await asyncio.sleep(1.0)
            except asyncio.TimeoutError:
                print(f"⏱️ Timeout for keyword '{keyword}' (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return keyword, []
                await asyncio.sleep(1.0)
            except Exception as e:
                print(f"❌ Exception for keyword '{keyword}': {e}")
                if attempt == max_retries - 1:
                    return keyword, []
                await asyncio.sleep(1.0)
    
    return keyword, []

async def fetch_serp_data_async(serper_api_key: str, keywords: List[str], max_concurrent: int = 5) -> Dict[str, List[str]]:
    """
    Fetch SERP data for multiple keywords concurrently with progress tracking
    """
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
    
    # Create progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create tasks for all keywords
        tasks = [
            fetch_serp_results_async(session, keyword, serper_api_key, semaphore)
            for keyword in keywords
        ]
        
        # Process tasks and track progress
        results = {}
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            keyword, urls = await coro
            results[keyword] = urls
            completed += 1
            
            # Update progress
            progress = completed / len(keywords)
            elapsed_time = time.time() - start_time
            
            if completed > 0:
                avg_time_per_keyword = elapsed_time / completed
                remaining_keywords = len(keywords) - completed
                eta_seconds = remaining_keywords * avg_time_per_keyword
                eta_minutes = eta_seconds / 60
                
                progress_placeholder.progress(progress)
                status_placeholder.text(
                    f"Processed {completed}/{len(keywords)} keywords | "
                    f"Rate: {completed/elapsed_time:.1f} keywords/sec | "
                    f"ETA: {eta_minutes:.1f} minutes"
                )
    
    return results

def fetch_serp_data(serper_api_key: str):
    """
    Main function - drop-in replacement for existing code
    Maintains same function signature for compatibility
    """
    # Load keywords
    keywords_df = pd.read_csv("keywords.csv")
    keywords = keywords_df["Keyword"].dropna().tolist()
    
    print(f"🚀 Starting concurrent SERP fetching for {len(keywords)} keywords...")
    
    # Check if we're in Streamlit environment
    try:
        import streamlit as st
        st.info(f"🔍 Fetching SERP data for {len(keywords)} keywords using concurrent processing...")
        
        # Run async function
        if hasattr(asyncio, 'create_task'):
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Use nest_asyncio for Streamlit compatibility
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                pass
                
            results = loop.run_until_complete(
                fetch_serp_data_async(serper_api_key, keywords, max_concurrent=5)
            )
        else:
            # Fallback for older Python versions
            results = asyncio.run(fetch_serp_data_async(serper_api_key, keywords, max_concurrent=5))
            
    except ImportError:
        # Fallback to synchronous processing if Streamlit not available
        print("📊 Streamlit not available, using synchronous processing...")
        results = {}
        for idx, keyword in enumerate(keywords):
            print(f"[{idx+1}/{len(keywords)}] Fetching SERP for: {keyword}")
            # Use synchronous version as fallback
            import requests
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json"
            }
            payload = {"q": keyword}
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    organic_results = data.get("organic", [])
                    urls = [item.get("link") for item in organic_results][:10]
                    results[keyword] = urls
                else:
                    results[keyword] = []
            except Exception as e:
                print(f"❌ Error fetching {keyword}: {e}")
                results[keyword] = []
            time.sleep(1.2)
    
    # Save results
    with open("serp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ SERP fetching completed. Results saved to serp_results.json")
    print(f"📊 Successfully fetched data for {len([k for k, v in results.items() if v])} keywords")
    
    return results

# Backwards compatibility
if __name__ == "__main__":
    # Example usage
    import os
    serper_key = os.getenv("SERPER_API_KEY")
    if serper_key:
        fetch_serp_data(serper_key)
    else:
        print("❌ SERPER_API_KEY not found in environment variables")
