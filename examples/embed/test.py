import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 1. 加载数据
df = pd.read_csv("boolq_similar_pairs.csv")

# 2. 加载模型
model_name = "Qwen/Qwen2.5-7B-Instruct"  # 确认官方模型名称
device = "cuda:1" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. 定义预测函数
def get_answer(question):
    prompt = f"""请根据以下问题给出答案（只需回答True/False）：
问题：{question}
答案："""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=10,
        temperature=0.01,  # 降低随机性
        do_sample=True
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取最终答案
    return "True" if "True" in answer else "False" if "False" in answer else "Unknown"

# 4. 批量处理预测
results = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        ans1 = get_answer(row["text1"])
        ans2 = get_answer(row["text2"])
        results.append((ans1, ans2))
    except Exception as e:
        print(f"Error processing row {idx}: {str(e)}")
        results.append(("Error", "Error"))

# 5. 保存结果
df["answer1"] = [r[0] for r in results]
df["answer2"] = [r[1] for r in results]
df.to_csv("boolq_model_results.csv", index=False)

# 6. 统计差异
mismatch_count = sum(1 for r in results if r[0] != r[1])
total_pairs = len(results) - sum(1 for r in results if "Error" in r)

print(f"总有效数据对: {total_pairs}")
print(f"答案不一致的对数: {mismatch_count}")
print(f"不一致比例: {mismatch_count/total_pairs:.2%}")