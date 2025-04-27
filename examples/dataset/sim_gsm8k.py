import json
import random
input_path = "examples/dataset/data/gsm8k/gsm8k_dataset_similar_docs_top5.json"
output_path = "examples/dataset/data/gsm8k/benchmark_gsm8k.json"

data = json.load(open(input_path, "r", encoding="utf-8"))


all_docs = data["all_data"]
all_paris = data["similar_pairs"]


new_data = []


for key, item in all_paris.items():
    id1 = item["id"]
    # id2 = item["cosine_similarity_top5"][1]["id"]
    
    target_doc = all_docs[str(id1)]
    # candidate_doc = all_docs[id1]
    
    new_data.append({
        "candidate_doc": "",
        "target_doc": target_doc["question"],
        "answer": target_doc["answer"]
    })

new_data = random.sample(new_data, 128)
print(len(new_data))
json.dump(new_data, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
