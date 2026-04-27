import json

with open("/Users/mrarnav69/Documents/AnchorSum/scratch_doc_334.txt", "r") as f:
    source_docs = f.read()

json_path = "/Users/mrarnav69/Documents/AnchorSum/example_334_analysis.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Change key name to be clearer
data["human_reference_summary"] = data.pop("reference_documents", "")
data["source_documents"] = source_docs

with open(json_path, "w") as f:
    json.dump(data, f, indent=4)
