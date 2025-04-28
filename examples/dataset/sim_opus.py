import json

input_path = "examples/dataset/data/opus/zero_score_samples.json"
output_path = "examples/dataset/data/opus/benchmark_opus.json"

data = json.load(open(input_path, "r", encoding="utf-8"))


all_docs = data["all_translations"]
all_paris = data["similar_pairs"]


new_data = []


for item in all_paris:
    id1 = item["id"]
    id2 = item["reused_top1_w31"]["id"]
    
    target_doc = all_docs[id1]
    candidate_doc = all_docs[id2]
    
    new_data.append({
        "candidate_doc": candidate_doc["zh"],
        "target_doc": target_doc["zh"],
        "answer": target_doc["en"]
    })
    
print(len(new_data))
json.dump(new_data, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)


output_path = "examples/dataset/data/opus/opus_test.json"

test_data = []

for item in new_data:
    test_data.append(item["target_doc"])
    
json.dump(test_data, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

