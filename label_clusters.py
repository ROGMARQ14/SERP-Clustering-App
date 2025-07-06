import openai
import pandas as pd
from collections import defaultdict
import time

def label_clusters(openai_api_key, input_file="clustered_keywords.csv", output_file="final_clustered_keywords.csv", model="gpt-4o-mini", temperature=0.3):
    # Set up OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Load the clustered keywords file
    df = pd.read_csv(input_file)
    
    # Group keywords by Hub
    hub_groups = defaultdict(list)
    for _, row in df.iterrows():
        hub_groups[row['Hub']].append(row['Keyword'])
    
    # Generate cluster labels using GPT with improved prompting
    hub_to_label = {}
    
    for hub, keywords in hub_groups.items():
        # Sample keywords if too many
        sample_keywords = keywords[:15] if len(keywords) > 15 else keywords
        
        prompt = f"""You are an SEO expert analyzing search intent. Given these keywords that cluster together based on Google search results:

Hub/Main Keyword: {hub}
Related Keywords: {', '.join(sample_keywords)}
Total Keywords in Cluster: {len(keywords)}

Analyze the search intent and generate a concise 2-4 word label that:
1. Captures the common user intent (informational, transactional, navigational, commercial)
2. Uses industry-standard SEO terminology
3. Is specific enough to differentiate from other clusters
4. Focuses on what users are trying to find/accomplish

Examples of good labels:
- "Local Repair Services" (for local service keywords)
- "Product Comparisons" (for comparison keywords)
- "DIY Guides" (for how-to keywords)
- "Commercial Solutions" (for B2B service keywords)

Return only the label, nothing else."""
    
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            label = response.choices[0].message.content.strip()
            hub_to_label[hub] = label
            print(f"✅ Labeled cluster for hub: '{hub}' → {label}")
        except Exception as e:
            print(f"❌ Error labeling hub '{hub}': {e}")
            # Fallback label based on hub
            if any(term in hub.lower() for term in ['service', 'repair', 'contractor']):
                hub_to_label[hub] = f"{hub.title()} Services"
            elif any(term in hub.lower() for term in ['best', 'top', 'review']):
                hub_to_label[hub] = f"{hub.title()} Reviews"
            else:
                hub_to_label[hub] = hub.title()
    
        time.sleep(1.1)  # Respect rate limit
    
    # Add the cluster labels to the DataFrame
    df["Cluster Label"] = df["Hub"].map(hub_to_label)
    
    # Add cluster metrics
    cluster_sizes = df.groupby("Hub").size().to_dict()
    df["Cluster Size"] = df["Hub"].map(cluster_sizes)
    
    # Reorder and save output
    df = df[["Cluster Label", "Hub", "Keyword", "Cluster Size", "URLs"]]
    df = df.sort_values(by=["Cluster Size", "Cluster Label", "Keyword"], ascending=[False, True, True])
    df.to_csv(output_file, index=False)
    
    print(f"✅ Final file saved as {output_file}")
    print(f"📊 Total clusters: {len(hub_to_label)}")
    print(f"📊 Total keywords: {len(df)}")
    
    return df
