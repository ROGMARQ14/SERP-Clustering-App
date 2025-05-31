# 🔍 SERP-Based Keyword Clustering Tool

This tool clusters keywords based on shared SERP results using a user-defined similarity threshold. It outputs results in a Hub & Spoke format and generates AI-powered cluster labels — ideal for SEO content planning.

---

## 📁 Project Structure

| File/Folder                     | Purpose                                              |
|--------------------------------|------------------------------------------------------|
| `keywords.csv`                 | Input file with keywords                            |
| `fetch_serp.py`               | Fetch top 10 SERP URLs per keyword via Serper API   |
| `serp_results.json`           | JSON file storing fetched SERP results              |
| `cluster_keywords.py`         | Clusters keywords based on SERP similarity          |
| `clustered_keywords.csv`      | Clusters output (before AI labeling)                |
| `label_clusters.py`           | Labels clusters using OpenAI GPT                    |
| `final_clustered_keywords.csv`| Final output with cluster labels                    |
| `requirements.txt`            | Dependencies file                                   |

---

## ⚙️ Setup Instructions

### 1. Create & Activate Virtual Environment

```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Or on Mac/Linux
source venv/bin/activate
```

---

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 🔑 API Keys Required

Before running scripts, make sure you have:
- **Serper API Key** – https://serper.dev
- **OpenAI API Key** – https://platform.openai.com

These keys will be inserted directly in the scripts when prompted.

---

## 🚀 How to Use the Tool

### STEP 1: Add Keywords
Edit the `keywords.csv` file with your list of keywords.  
The file should look like:

```csv
Keyword
asphalt repair near me
driveway sealing companies
asphalt driveway cost
...
```

---

### STEP 2: Fetch SERP Results

```bash
python fetch_serp.py
```

This will:
- Read keywords from `keywords.csv`
- Query Serper API to fetch top 10 Google results for each
- Save them to `serp_results.json`

---

### STEP 3: Cluster the Keywords

```bash
python cluster_keywords.py
```

This will:
- Compare keyword SERP result sets using similarity
- Group keywords into clusters
- Save as `clustered_keywords.csv`

> 🔧 You can adjust the similarity threshold in the script (default is 0.3 or 30%)

---

### STEP 4: Generate Cluster Labels

```bash
python label_clusters.py
```

This script uses GPT (via OpenAI API) to assign a meaningful name to each cluster.

Output: `final_clustered_keywords.csv`

---

## 📤 Output Format

| Cluster Label                         | Hub                              | Keyword                          | URLs                           |
|--------------------------------------|----------------------------------|----------------------------------|--------------------------------|
| "Expert Asphalt Services in Ann Arbor" | expert asphalt repair services in ann arbor | driveway sealing companies | url1<br>url2<br>url3...        |

---

## 📝 Notes

- **Google Sheets–friendly**: Output supports line breaks and filter-ready columns.
- **You can adjust clustering threshold** in `cluster_keywords.py`.
- **You can open `final_clustered_keywords.csv` in Excel or Google Sheets** and apply:
  - **Wrap text** on URLs column
  - **Create filter** on header row
  - **Freeze 1st row** for easy scrolling

---

## 📩 Support

Developed by **Pranjal Shukla**  
Need help or want to add more features? Let’s connect!