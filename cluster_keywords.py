import json
import pandas as pd
from collections import defaultdict
import numpy as np

def cluster_keywords_by_serp(input_file="serp_results.json", 
                           output_file="clustered_keywords.csv", 
                           similarity_threshold=0.3):
    """
    Cluster keywords based on SERP URL overlap using hub-and-spoke method
    """
    # Load SERP results
    with open(input_file, 'r') as f:
        serp_results = json.load(f)
    
    keywords = list(serp_results.keys())
    n = len(keywords)
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                urls1 = set(serp_results[keywords[i]])
                urls2 = set(serp_results[keywords[j]])
                
                intersection = len(urls1.intersection(urls2))
                union = len(urls1.union(urls2))
                
                similarity = intersection / union if union > 0 else 0.0
                similarity_matrix[i][j] = similarity
    
    # Find hubs (keywords with highest average similarity to others)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    hub_scores = [(keywords[i], avg_similarities[i]) for i in range(n)]
    hub_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Assign keywords to clusters
    assigned = set()
    clusters = []
    
    for hub_keyword, hub_score in hub_scores:
        if hub_keyword in assigned:
            continue
            
        hub_idx = keywords.index(hub_keyword)
        cluster = {"hub": hub_keyword, "keywords": [hub_keyword], "urls": set(serp_results[hub_keyword])}
        assigned.add(hub_keyword)
        
        # Find similar keywords
        for j, other_keyword in enumerate(keywords):
            if other_keyword not in assigned and similarity_matrix[hub_idx][j] >= similarity_threshold:
                cluster["keywords"].append(other_keyword)
                cluster["urls"].update(serp_results[other_keyword])
                assigned.add(other_keyword)
        
        clusters.append(cluster)
    
    # Create output dataframe
    rows = []
    for cluster in clusters:
        hub = cluster["hub"]
        urls_str = "\n".join(list(cluster["urls"])[:10])  # Top 10 URLs
        
        for keyword in cluster["keywords"]:
            rows.append({
                "Hub": hub,
                "Keyword": keyword,
                "Cluster Size": len(cluster["keywords"]),
                "URLs": urls_str
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Cluster Size", "Hub", "Keyword"], ascending=[False, True, True])
    df.to_csv(output_file, index=False)
    
    print(f"✅ Clustered {len(df)} keywords into {len(clusters)} clusters")
    print(f"📊 Average cluster size: {len(df) / len(clusters):.1f}")
    print(f"💾 Results saved to {output_file}")
    
    return df

if __name__ == "__main__":
    cluster_keywords_by_serp()
