import openai
import pandas as pd
from collections import defaultdict
import time

# Set your OpenAI API key
openai.api_key = "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"

# Load the clustered keywords file
df = pd.read_csv("clustered_keywords.csv")

# Group keywords by Hub
hub_groups = defaultdict(list)
for _, row in df.iterrows():
    hub_groups[row['Hub']].append(row['Keyword'])

# Generate cluster labels using GPT
hub_to_label = {}

for hub, keywords in hub_groups.items():
    prompt = (
        f"Given the following list of search keywords:\n\n"
        f"{keywords}\n\n"
        "What is a short, content-marketing-friendly topic label for this group?\n"
        "Avoid generic names like 'Cluster 1'. Return just the label."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        label = response.choices[0].message['content'].strip()
        hub_to_label[hub] = label
        print(f"✅ Labeled cluster for hub: '{hub}' → {label}")
    except Exception as e:
        print(f"❌ Error labeling hub '{hub}': {e}")
        hub_to_label[hub] = "Unknown Cluster"

    time.sleep(1.1)  # Respect rate limit

# Add the cluster labels to the DataFrame
df["Cluster Label"] = df["Hub"].map(hub_to_label)

# Reorder and save output
df = df[["Cluster Label", "Hub", "Keyword", "URLs"]]
df.to_csv("final_clustered_keywords.csv", index=False)

print("✅ Final file saved as final_clustered_keywords.csv")
