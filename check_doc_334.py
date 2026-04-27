import json

with open("/Users/mrarnav69/Documents/AnchorSum/data/multi_news_500_samples.json", "r") as f:
    data = json.load(f)

# If it's a list, get the 334th element (or check if example_id is present)
for i, item in enumerate(data):
    # Multi-news usually has no 'example_id' originally, we probably assigned index as id.
    # Let's check both index 334 and item.get('example_id') == 334
    if i == 334 or item.get('example_id') == 334:
        print(f"Index {i}")
        doc = item.get('document', '')
        print("Document length:", len(doc))
        print("Starts with:", doc[:500])
        with open("/Users/mrarnav69/Documents/AnchorSum/scratch_doc_334.txt", "w") as f2:
            f2.write(doc)
        break
