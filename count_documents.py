import json
import numpy as np

file_path = "/Users/mrarnav69/Documents/AnchorSum/data/multi_news_500_samples.json"

with open(file_path, "r") as f:
    data = json.load(f)

doc_counts = []
for instance in data:
    text = instance.get("document", "")
    # The separator seems to be |||||
    parts = text.split("|||||")
    doc_counts.append(len(parts))

avg_docs = np.mean(doc_counts)
median_docs = np.median(doc_counts)
min_docs = np.min(doc_counts)
max_docs = np.max(doc_counts)

print(f"Total instances: {len(data)}")
print(f"Average number of documents per instance: {avg_docs:.2f}")
print(f"Median: {median_docs}")
print(f"Min: {min_docs}")
print(f"Max: {max_docs}")
