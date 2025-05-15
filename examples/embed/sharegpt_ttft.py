import json
import random
from collections import Counter
from transformers import AutoTokenizer

data = json.load(open("/root/code/vllm_plus/examples/dataset/data/sharegpt/sharegpt90k_similar_250331.json"))
similar_docs = data["similar_pairs"]
all_texts = data["all_texts"]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 从similar_docs中筛选所有符合条件的文本对
filtered_pairs_high = []  # 相似度在0.9-0.98之间的文本对
filtered_pairs_medium = []  # 相似度在0.7-0.9之间的文本对

for pair in similar_docs:
    # 要求当前文本长度超过1000
    if len(all_texts[str(pair["id"])]) > 2048 and  len(all_texts[str(pair["id"])]) < 3144:
        # 检查high_token_reused_top5中前两个的similarity
        if len(pair["high_token_reused_top5"]) >= 2:
            first_sim = pair["high_token_reused_top5"][0]["similarity"]
            second_sim = pair["high_token_reused_top5"][1]["similarity"]
            
            # 根据相似度范围分类
            if 0.9 < first_sim <= 0.98 and 0.9 < second_sim <= 0.98:
                filtered_pairs_high.append([all_texts[str(pair["id"])],all_texts[str(pair["high_token_reused_top5"][1]["id"])]])
            elif 0.7 < first_sim <= 0.9 and 0.7 < second_sim <= 0.9:
                filtered_pairs_medium.append([all_texts[str(pair["id"])],all_texts[str(pair["high_token_reused_top5"][1]["id"])]])

print(f"找到相似度在0.9-0.98之间的文本对数量: {len(filtered_pairs_high)}")
print(f"找到相似度在0.7-0.9之间的文本对数量: {len(filtered_pairs_medium)}")

# 从不同相似度范围的文本对中随机选择指定数量
num_high = 128
num_medium = 0

selected_pairs_high = random.sample(filtered_pairs_high, min(num_high, len(filtered_pairs_high))) if filtered_pairs_high else []
selected_pairs_medium = random.sample(filtered_pairs_medium, min(num_medium, len(filtered_pairs_medium))) if filtered_pairs_medium else []

# 合并所有选中的文本对
selected_pairs = selected_pairs_high + selected_pairs_medium

print(f"最终选择的文本对数量: {len(selected_pairs)}")
print(f"其中相似度在0.9-0.98之间的文本对数量: {len(selected_pairs_high)}")
print(f"其中相似度在0.7-0.9之间的文本对数量: {len(selected_pairs_medium)}")

# 提取当前文本和候选文本
text_triplets = {
    "candidates": [],
    "targets": []
}
for pair in selected_pairs:
    current_text = pair[0]
    candidate1 = pair[1]
    text_triplets["candidates"].append(candidate1)
    text_triplets["targets"].append(current_text)
    
# candidates 需要去重
text_triplets["candidates"] = list(set(text_triplets["candidates"]))    

print(len(text_triplets["candidates"]))
print(len(text_triplets["targets"]))
# json.dump(selected_pairs, open("examples/dataset/data/sharegpt/sharegpt90k_filtered_pairs.json", "w"), indent=4, ensure_ascii=False)
json.dump(text_triplets, open("examples/dataset/data/sharegpt/sharegpt90k_throught_benchmark_high_sim.json", "w"), indent=4, ensure_ascii=False)

