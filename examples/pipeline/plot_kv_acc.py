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


def analyze_token_reuse_rate(json_path:str,error_edit_path:str):
    from edit2 import find_text_differences,apply_text_changes
    data = json.load(open(json_path))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    clusters_data = data["clusters"]
    
    plot_data_x = []  # total text length (source + target)
    plot_data_y = []  # processing time (ms)
    error_count = 0
    error_items = []
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
            if modified_tokens != target_tokens:
                error_count += 1
                error_items.append({
                    "source_text":source_text,
                    "target_text":target_text,
                    "modified_text": tokenizer.decode(modified_tokens),
                    # "target_tokens":target_tokens
                })
            
            
            plot_data_x.append(len(source_tokens)+len(target_tokens))
            plot_data_y.append(processing_time*1000)  # 转换为毫秒
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(plot_data_x, plot_data_y, 
               alpha=0.5,        # 设置透明度
               s=20,            # 点的大小
               c=plot_data_y,   # 使用处理时间作为颜色
               cmap='viridis')  # 使用viridis颜色映射
    
    # 设置y轴为对数刻度
    plt.yscale('log')
    
    # 添加颜色条
    plt.colorbar(label='Processing Time (ms)')
    
    # 设置轴标签
    plt.xlabel('Total Text Length (tokens)')
    plt.ylabel('Processing Time (ms) - Log Scale')
    
    # 设置标题
    plt.title('Text Length vs Processing Time (Log Scale)')
    
    # 添加网格（对数刻度下的网格）
    plt.grid(True, linestyle='--', alpha=0.7, which='both')  # 'both' 表示主要和次要网格线都显示
    
    # 保存图片
    plt.savefig('examples/pipeline/images/edit_time_log.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error count: {error_count}")
    print(f"Error rate: {error_count/len(plot_data_x):.4f}")
    print(f"Average processing time: {np.mean(plot_data_y):.2f} ms")
    print(f"Max processing time: {max(plot_data_y):.2f} ms")
    print(f"Min processing time: {min(plot_data_y):.2f} ms")
    
    json.dump(error_items,open(error_edit_path,"w"),indent=4,ensure_ascii=False)

if __name__ == "__main__":
    
    
    analyze_token_reuse_rate("examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters.json",
                             "examples/dataset/data/similar/instruction_wildv2/error_edit.json")
    
    # plot_kv_acc("examples/pipeline/data/kv")
    pass
    