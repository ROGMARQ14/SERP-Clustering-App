import requests
import pandas as pd
import json
import time

# Load keywords
keywords_df = pd.read_csv("keywords.csv")
keywords = keywords_df["Keyword"].dropna().tolist()

# Serper API details
SERPER_API_KEY = "eb089cc1bad537ae8a8b5d5ac2d3ecf9e101142a"
SERPER_API_URL = "https://google.serper.dev/search"

headers = {
    "X-API-KEY": SERPER_API_KEY,
    "Content-Type": "application/json"
}

# Function to fetch SERP results
def fetch_serp_results(keyword):
    payload = {
        "q": keyword
    }
    response = requests.post(SERPER_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        organic_results = data.get("organic", [])
        urls = [item.get("link") for item in organic_results][:10]
        return urls
    else:
        print(f"Error fetching for keyword '{keyword}': {response.status_code}")
        return []

# Main loop
results = {}
for idx, keyword in enumerate(keywords):
    print(f"[{idx+1}/{len(keywords)}] Fetching SERP for: {keyword}")
    results[keyword] = fetch_serp_results(keyword)
    time.sleep(1.2)  # To respect rate limits (1–2 req/sec max)

# Save results
with open("serp_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ SERP fetching completed. Results saved to serp_results.json")
