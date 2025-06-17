import requests
import pandas as pd
import json
import time
import sys

def fetch_serp_results(keyword, serper_api_key):
    """Fetch SERP results for a given keyword using Serper API"""
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": keyword
    }
    
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get("organic", [])
            urls = [item.get("link") for item in organic_results][:10]
            return urls
        else:
            print(f"Error fetching for keyword '{keyword}': {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception fetching for keyword '{keyword}': {e}")
        return []

def fetch_all_serp_results(keywords_file, serper_api_key, output_file="serp_results.json"):
    """Fetch SERP results for all keywords in the CSV file"""
    try:
        # Load keywords
        keywords_df = pd.read_csv(keywords_file)
        
        # Try to find keyword column
        keyword_column = None
        for col in ['Keyword', 'Keywords', 'keyword', 'keywords', 'Query', 'query']:
            if col in keywords_df.columns:
                keyword_column = col
                break
        
        if keyword_column is None:
            keyword_column = keywords_df.columns[0]  # Use first column as fallback
            
        keywords = keywords_df[keyword_column].dropna().tolist()
        
        if not keywords:
            print("❌ No keywords found in the CSV file.")
            return False
            
        print(f"📋 Found {len(keywords)} keywords to process")
        
        # Fetch SERP results
        results = {}
        for idx, keyword in enumerate(keywords):
            print(f"[{idx+1}/{len(keywords)}] Fetching SERP for: {keyword}")
            results[keyword] = fetch_serp_results(keyword, serper_api_key)
            time.sleep(1.2)  # To respect rate limits (1–2 req/sec max)
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ SERP fetching completed. Results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing SERP data: {e}")
        return False

# Command line interface
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_serp.py <keywords_file> <serper_api_key> [output_file]")
        print("Example: python fetch_serp.py keywords.csv your_api_key_here serp_results.json")
        sys.exit(1)
    
    keywords_file = sys.argv[1]
    serper_api_key = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "serp_results.json"
    
    success = fetch_all_serp_results(keywords_file, serper_api_key, output_file)
    sys.exit(0 if success else 1)
