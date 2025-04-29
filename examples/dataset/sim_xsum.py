import json
import random
import uuid
input_path = "examples/dataset/data/xsum/xsum_dataset_similar_docs_top50_250403_windows.json"
output_path = "examples/dataset/data/xsum/benchmark_xsum.json"

data = json.load(open(input_path, "r", encoding="utf-8"))


all_docs = data["all_documents"]
all_paris = data["similar_docs"]


new_data = []


for item in all_paris:
    # id1 = item["id"]
    id2 = item["simi_top1"]
    
    # target_doc = all_docs[str(id1)]
    candidate_doc = all_docs[str(id2)]
    
    new_data.append({
        "uuid": str(uuid.uuid4()),
        "candidate_doc": candidate_doc["document"],
        "target_doc": item["document"],
        "answer": item["summary"]
    })

new_data = random.sample(new_data, 512)
print(len(new_data))
json.dump(new_data, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
