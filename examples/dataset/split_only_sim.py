import json
import hashlib


def split_text(json_path: str, new_json_path: str):
    data = json.load(open(json_path))
    new_data = []
    for item in data:
        text1 = item["text1"]
        text2 = item["text2"]
        hash1 = hashlib.md5(text1.encode()).hexdigest()
        hash2 = hashlib.md5(text2.encode()).hexdigest()
        if hash1 != hash2:
            # print(item)
            new_data.append(item)
    json.dump(new_data, open(new_json_path, "w"), ensure_ascii=False,indent=4)

if __name__ == "__main__":

    # wild_path = "examples/dataset/data/wild_chat_sim.json"
    # new_wild_path = "examples/dataset/data/wild_chat_sim_only_similarity.json"
    # split_text(wild_path, new_wild_path)
    
    # sharegpt_path = "examples/dataset/data/sharegpt90k_sim.json"
    # new_sharegpt_path = "examples/dataset/data/sharegpt90k_sim_only_similarity.json"
    # split_text(sharegpt_path, new_sharegpt_path)



    wild_path = "examples/dataset/data/lmsys_chat_1m_sim.json"
    new_wild_path = "examples/dataset/data/lmsys_chat_1m_sim_only_similarity.json"
    split_text(wild_path, new_wild_path)
