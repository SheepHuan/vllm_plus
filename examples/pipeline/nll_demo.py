import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

def calculate_ppl(prompt,text, model, tokenizer: AutoTokenizer):
    # 分词并转换为模型输入格式
    inputs = tokenizer(prompt+"\n"+text, return_tensors="pt", add_special_tokens=True).to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.size(1)
    decode_seq_len = len(tokenizer.encode(text,add_special_tokens=False))
    
    
    # 获取模型的输出logits
    with torch.no_grad():
        outputs = model(**inputs, max_new_tokens=1)
    logits = outputs.logits

    # 计算每个token的条件概率对数
    nll = 0.0
    cumulative_ppls = []  # 存储每个位置的累计PPL
    for i in range(1, input_ids.size(1)):  # 从第2个token开始计算
        current_logits = logits[:, i-1, :]
        current_token_id = input_ids[:, i]
        probs = torch.softmax(current_logits, dim=-1)
        token_prob = probs[:, current_token_id].squeeze()
        nll -= torch.log(token_prob).item()
        
        # 计算当前位置的累计PPL
        current_ppl = math.exp(nll / i)  # 使用当前位置作为分母
        cumulative_ppls.append(current_ppl)

    # 返回每个位置的累计PPL和最终PPL
    return cumulative_ppls[-decode_seq_len:], math.exp(nll / seq_len)


if __name__ == "__main__":
    # 加载模型和分词器
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda:0")

    fc_data = json.load(open("examples/ppl_analysis/gsm8k_benchmark_full_compute.json","r"))
    naive_data = json.load(open("examples/ppl_analysis/gsm8k_benchmark_naive.json","r"))
    kvshare_prefill_data = json.load(open("examples/ppl_analysis/gsm8k_benchmark_kvshare-prefill.json","r"))
    kvshare_decode_data = json.load(open("examples/ppl_analysis/gsm8k_benchmark_kvshare-decode.json","r"))
    
    assert len(fc_data) == len(naive_data) == len(kvshare_prefill_data) == len(kvshare_decode_data)
    
    # 用于存储每个位置的PPL值
    position_ppls = {
        'full_compute': defaultdict(list),
        'naive': defaultdict(list),
        'kvshare_prefill': defaultdict(list),
        'kvshare_decode': defaultdict(list)
    }
    num = 64
    # 遍历所有样本
    for fc, naive, kvshare_prefill, kvshare_decode in zip(fc_data[:num], naive_data[:num], kvshare_prefill_data[:num], kvshare_decode_data[:num]):
        target_doc = fc["target_doc"]
        # 计算每个模式的PPL
        for mode, data in [
            ('full_compute', fc['full_compute_output']),
            ('naive', naive['partial_compute_output']),
            ('kvshare_prefill', kvshare_prefill['partial_compute_output']),
            ('kvshare_decode', kvshare_decode['partial_compute_output'])
        ]:
            # for pos, output in enumerate(data):
            cumulative_ppls, final_ppl = calculate_ppl(target_doc,data, model, tokenizer)
            # 存储每个位置的累计PPL
            for token_pos, ppl in enumerate(cumulative_ppls):
                position_ppls[mode][token_pos].append(ppl)
    
    # 计算每个位置的平均PPL和方差
    results = {}
    for mode in position_ppls:
        results[mode] = {
            'mean_ppl': [],
            'std_ppl': [],
            'positions': []
        }
        
        for pos in sorted(position_ppls[mode].keys()):
            ppls = position_ppls[mode][pos]
            results[mode]['positions'].append(pos)
            results[mode]['mean_ppl'].append(np.mean(ppls))
            results[mode]['std_ppl'].append(np.std(ppls))
    
    # 保存结果
    with open('ppl_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    

    # 创建图表
    plt.figure(figsize=(6, 6), dpi=300)
    
    # 设置颜色方案
    colors = {
        'full_compute': '#1f77b4',  # 蓝色
        'naive': '#2ca02c',  # 绿色
        'kvshare_prefill': '#d62728',  # 红色
        'kvshare_decode': '#ff7f0e'  # 橙色
    }
    
    # 设置线型和标记
    line_styles = {
        'full_compute': '-',
        'naive': '--',
        'kvshare_prefill': '-.',
        'kvshare_decode': ':'
    }

    # 设置图例标签映射
    legend_labels = {
        'full_compute': 'Full Recompute',
        'naive': 'Naive',
        'kvshare_prefill': 'KVShare Recomp in Prefill',
        'kvshare_decode': 'KVShare Recomp in Prefill & Decode'
    }
    
    # 绘制均值
    for mode in ['full_compute', 'naive', 'kvshare_prefill', 'kvshare_decode']:
        plt.plot(results[mode]['positions'], 
                results[mode]['mean_ppl'],
                label=legend_labels[mode],
                color=colors[mode],
                linestyle=line_styles[mode],
                marker='o',
                markersize=4,
                linewidth=1.5,
                alpha=0.8)
    
    # 设置坐标轴
    plt.xlabel('Decode Token Position', fontsize=12, labelpad=10)
    plt.ylabel('PPL', fontsize=12, labelpad=10)
    
    # 设置刻度
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 设置图例
    plt.legend(fontsize=10, 
              frameon=True,
              fancybox=True,
              framealpha=0.8,
              edgecolor='gray',
              loc='best')
    
    # 设置背景
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('white')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('ppl_analysis_plot.png', 
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    print("分析完成，结果已保存到 ppl_analysis_results.json 和 ppl_analysis_plot.png")
    
    # 我想要分析一个图，横轴是decode token的位置，我想要观察不同的计算模式下，在相同的decode位置的平均PPL值和，方差PPL
    for fc,naive,kvshare_prefill,kvshare_decode in zip(fc_data,naive_data,kvshare_prefill_data,kvshare_decode_data):
        
        fc_output = fc["full_compute_output"]
        naive_output = naive["partial_compute_output"]
        kvshare_prefill_output = kvshare_prefill["partial_compute_output"]
        kvshare_decode_output = kvshare_decode["partial_compute_output"]
        
        
        
        