import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager 
from tqdm import tqdm

font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
import json
from typing import List
import time
def get_key_value(model:LLM,prompt: List[str],save_dir:str):
    # template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # prompt = template.format(prompt=prompt)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    output = model.generate(prompt, sampling_params)
    print(output[0].outputs[0].text)
    llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    
    past_key_values = []
    num_layer = len(llm_layers)
    for j in range(num_layer):
        hack_kv = llm_layers[j].self_attn.hack_kv
        temp_key_cache = hack_kv[0].clone()
        temp_value_cache = hack_kv[1].clone()
        # print(temp_key_cache.shape)
        past_key_values.append([temp_key_cache,temp_value_cache])
    os.makedirs(save_dir,exist_ok=True)
    kv_save_path = os.path.join(save_dir,"kv.pth")
    token_save_path = os.path.join(save_dir,"token.json")
    prompt_token_ids = output[0].prompt_token_ids
    torch.save(past_key_values,kv_save_path)
    json.dump(prompt_token_ids,open(token_save_path,"w"))

    return past_key_values

def plot_prefill_efficiency(json_path:str,prefill_time_path:str):
    model = LLM(model="Qwen/Qwen2.5-7B-Instruct",
                dtype="bfloat16",
                multi_step_stream_outputs=False,enforce_eager=True,disable_async_output_proc=False,enable_prefix_caching=False,
                device="cuda:0"
                )
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    data = json.load(open(json_path))
    new_data = []
    for item in tqdm(data):
        target_text = item["target_text"]
        full_compute_time = []
        for _ in range(15):
            output = model.generate(target_text,sampling_params,use_tqdm=False)
            full_compute_time.append(output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time)
        item["full_compute_time"] = np.mean(full_compute_time[5:]) * 1000
        new_data.append(item)
    json.dump(new_data,open(prefill_time_path,"w"),indent=4,ensure_ascii=False)


def plot_embedding_efficiency(json_path:str,embedding_time_path:str):
    from sentence_transformers import SentenceTransformer
    device = "cuda:1"
    model = SentenceTransformer("all-MiniLM-L6-v2",local_files_only=True).to(device).to(torch.bfloat16)
    data = json.load(open(json_path))
    new_data = []
    for item in tqdm(data):
        target_text = item["target_text"]
        embedding_time = []
        for _ in range(15):
            start_time = time.time()
            embedding = model.encode(target_text)
            end_time = time.time()
            embedding_time.append(end_time - start_time)
        item["embedding_time"] = np.mean(embedding_time[5:]) * 1000
        new_data.append(item)
    json.dump(new_data,open(embedding_time_path,"w"),indent=4,ensure_ascii=False)

def plot_edit_efficiency(json_path:str,edit_time_path:str):
    from edit2 import find_text_differences,apply_text_changes
    data = json.load(open(json_path))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    clusters_data = data["clusters"]
    
    plot_data_x = []  # total text length (source + target)
    plot_data_y = []  # processing time (ms)

    edit_items = []
    for key,cluster in tqdm(clusters_data.items()):
        size = cluster["size"]
        members = cluster["members"]
        texts = [member["text"] for member in members]
        
        source_text = texts[0]
        for target_text in texts[1:]:
            target_tokens = tokenizer.encode(target_text)
            source_tokens = tokenizer.encode(source_text)
            start_time = time.time()
            diff_report = find_text_differences(tokenizer, source_tokens, target_tokens)
            modified_tokens = apply_text_changes(source_tokens, target_tokens, diff_report, tokenizer)
            end_time = time.time()
            processing_time = end_time - start_time
            # check if the modified tokens are the same as the target tokens
         
            edit_items.append({
                "source_text":source_text,
                "target_text":target_text,
                "edit_time":processing_time * 1000
            })
            # plot_data_x.append(len(source_tokens)+len(target_tokens))
            # plot_data_y.append(processing_time*1000)  # 转换为毫秒
    json.dump(edit_items,open(edit_time_path,"w"),indent=4,ensure_ascii=False)

def plot_comparison_curves(embedding_time_path: str, edit_time_path: str , prefill_time_path:str):
    """
    对比编辑时间和嵌入时间的性能曲线
    Args:
        embedding_time_path: 嵌入时间数据文件路径
        edit_time_path: 编辑时间数据文件路径
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # 加载数据
    embedding_data = json.load(open(embedding_time_path))
    edit_data = json.load(open(edit_time_path))
    prefill_data = json.load(open(prefill_time_path))
    # 提取数据点
    embedding_lengths = [len(tokenizer.encode(item["target_text"])) for item in embedding_data]
    embedding_times = [item["embedding_time"] for item in embedding_data]
    
    edit_lengths = [len(tokenizer.encode(item["target_text"])) 
                   for item in edit_data]
    edit_times = [item["edit_time"] for item in edit_data]
    
    prefill_lengths = [len(tokenizer.encode(item["target_text"])) 
                   for item in prefill_data]
    prefill_times = [item["full_compute_time"] for item in prefill_data]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制嵌入时间散点和拟合曲线
    plt.scatter(embedding_lengths, embedding_times, 
               alpha=0.5, 
               label='Embedding Samples', 
               color='#4876FF', 
               s=15)
    
    # 绘制编辑时间散点和拟合曲线
    plt.scatter(edit_lengths, edit_times, 
               alpha=0.5, 
               label='Text Edit Samples', 
               color='#EEB4B4', 
               s=15)
    
    plt.scatter(prefill_lengths, prefill_times, 
               alpha=0.5, 
               label='Prefill Samples', 
               color='#B4EEB4', 
               s=15)
    
    # 对两组数据进行多项式拟合
    for data_x, data_y, name, color,dim in [
        (embedding_lengths, embedding_times, 'Embedding Time Curve', 'blue',1),
        (edit_lengths, edit_times, 'KVEdit Time Curve', 'red',1),
        (prefill_lengths, prefill_times, 'Prefill Time Curve', 'green',1)
    ]:
        # 排序数据点
        sorted_indices = np.argsort(data_x)
        x_sorted = np.array(data_x)[sorted_indices]
        y_sorted = np.array(data_y)[sorted_indices]
        
        # 多项式拟合
        coefficients = np.polyfit(x_sorted, y_sorted, dim)
        poly = np.poly1d(coefficients)
        
        # 生成平滑曲线
        x_smooth = np.linspace(min(x_sorted), max(x_sorted), 1000)
        y_smooth = poly(x_smooth)
        
        # 计算R²值
        y_mean = np.mean(y_sorted)
        ss_tot = np.sum((y_sorted - y_mean) ** 2)
        ss_res = np.sum((y_sorted - poly(x_sorted)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        plt.plot(x_smooth, y_smooth, 
                color=color, 
                label=f'{name}',
                linewidth=2)
    
    plt.xlabel('Sequence Length (tokens)')
    plt.ylabel('Latency (ms)')
    plt.title('Performance Analysis: Text Edit vs Embedding')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.yscale('log')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('examples/pipeline/images/performance_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_comparison_curves(
        "examples/dataset/data/similar/instruction_wildv2/embedding_time.json",
        "examples/dataset/data/similar/instruction_wildv2/edit_time.json",
        "examples/dataset/data/similar/instruction_wildv2/prefill_time.json"
    )
    
    
    