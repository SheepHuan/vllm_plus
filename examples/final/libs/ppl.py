import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
def calculate_nll(text, model, tokenizer):
    # 分词并转换为模型输入格式
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(model.device)
    input_ids=inputs.input_ids
    # 获取模型的输出 logits
    with torch.no_grad():
        outputs = model(**inputs,max_new_tokens=1)
    logits = outputs.logits

    # 计算每个 token 的条件概率对数
    nll = 0.0
    for i in range(1, input_ids.size(1)):  # 从第2个token开始计算（自回归预测）
        # 获取当前token的logits和真实token ID
        current_logits = logits[:, i-1, :]  # 模型预测第i个token的logits基于前i-1个token
        current_token_id = input_ids[:, i]

        # 计算softmax概率
        probs = torch.softmax(current_logits, dim=-1)
        token_prob = probs[:, current_token_id].squeeze()

        # 累加负对数概率
        nll -= torch.log(token_prob).item()
    # print(f"NLL of '{text}': {nll:.2f}")
    return nll

def calculate_ppl(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]
    
    # 批量分词
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    batch_size = input_ids.size(0)
    
    # 获取模型的输出logits
    with torch.no_grad():
        outputs = model(**inputs, max_new_tokens=1)
    logits = outputs.logits

    # 计算每个序列的PPL
    ppls = []
    for b in range(batch_size):
        nll = 0.0
        seq_len = attention_mask[b].sum().item() - 1  # 减1是因为我们不计算第一个token
        
        # 计算当前序列中每个token的条件概率对数
        for i in range(1, seq_len + 1):  # 从第2个token开始计算
            current_logits = logits[b, i-1, :]
            current_token_id = input_ids[b, i]
            probs = torch.softmax(current_logits, dim=-1)
            token_prob = probs[current_token_id]
            nll -= torch.log(token_prob).item()
        
        # 计算困惑度
        ppl = math.exp(nll / seq_len)
        ppls.append(ppl)
    return ppls

if __name__ == "__main__":
    # 加载模型和分词器
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 示例句子
    calculate_nll( "I love China.", model, tokenizer)
    calculate_nll( "I love dog.", model, tokenizer)
    calculate_nll( "I love !!!.", model, tokenizer)