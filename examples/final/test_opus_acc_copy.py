import json
from libs.pipeline import KVShareNewPipeline
from libs.edit import KVEditor
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import os
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from matplotlib import pyplot as plt
from evaluate import load
import seaborn as sns
from scipy.stats import zscore
import math
import re
import evaluate
import matplotlib
from matplotlib import font_manager 
import traceback
import multiprocessing as mp
from functools import partial
from typing import List

# download the font files and save in this fold
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

qwen_template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nTranslate the following text from Chinese to English:\n{text}\n<|im_end|>\n<|im_start|>assistant\n"""
llama3_template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Translate the following text from Chinese to English:\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

OPUS_KVCACHE_DIR="examples/pipeline/kvcache/opus"

def generate_output_data(input_path: str, output_path: str, model_name = "Qwen/Qwen2.5-7B-Instruct", batch_size=4, max_model_len=8192,max_generate_len=512):
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device, max_model_len)
    tokenizer = pipeline.model.get_tokenizer()
    meteor_metric = evaluate.load('meteor')
    
    # 加载数据
    with open(input_path, "r") as f:
        dataset = json.load(f)
    
    all_translations = dataset["all_translations"]
    similar_pairs = dataset["similar_pairs"]
    # 过滤相似度过高的样本
    similar_pairs = [pair for pair in similar_pairs if pair["reused_top1_w31"]["similarity"] < 0.9]
    
    # 检查是否存在已处理的数据
    processed_pairs = []
    if os.path.exists(output_path):
        processed_pairs = []
    
    print(f"处理后样本数量: {len(similar_pairs)}")
    # 随机采样样本
    # similar_pairs = random.sample(similar_pairs, 400)
    
    processed_results = []
    
    # 批量处理数据
    for batch_start in tqdm(range(0, len(similar_pairs), batch_size), desc="Processing batches"):
        try:
            batch_items = similar_pairs[batch_start:batch_start + batch_size]
            
            # 准备批次数据
            batch_target_prompts = []
            batch_source_prompts = []
            batch_references = []
            valid_batch_items = []
            
            for item in batch_items:
                source_text = all_translations[str(item["id"])]["zh"]
                target_text = all_translations[str(item["id"])]["en"]
                # reused_source_text = source_text
                reused_source_text = all_translations[str(item["reused_top1_w31"]["id"])]["zh"]
                # 检查token长度
                source_tokens = tokenizer.encode(template.format(text=source_text))
                reused_tokens = tokenizer.encode(template.format(text=reused_source_text))
                
                if len(source_tokens) <= max_model_len and len(reused_tokens) <= max_model_len:
                    batch_target_prompts.append(template.format(text=source_text))
                    batch_source_prompts.append(template.format(text=reused_source_text))
                    batch_references.append(target_text)
                    valid_batch_items.append(item)
            
            if not valid_batch_items:
                continue
                     
            # 获取source的KV缓存
            source_kv_cache, source_outputs, source_req_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=1),
                batch_source_prompts
            )
            
            real_target_kv_cache,real_target_outputs,real_target_req_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=1),
                batch_target_prompts
            )
            
            source_token_ids = [output.prompt_token_ids for output in source_outputs]
            max_request_id = max(int(req_output.request_id) for req_output in real_target_outputs)
            
            # 对不同window size进行处理
            for window_size in [1,3,7,11,15]:
                # 批量KV编辑
                target_kv_cache, reused_indices, unreused_indices, selected_tokens, target_slices = KVEditor.batch_kvedit(
                    [tokenizer.encode(prompt) for prompt in batch_target_prompts],
                    source_token_ids,
                    source_kv_cache,
                    window_size=window_size
                )
                # merge reused_indices
                reused_token_index_each_item = []
                unreused_token_index_each_item = []
                # reused_prefix_len = 0
                for idx, (resued_item_indices,unreused_item_indices) in enumerate(zip(reused_indices,unreused_indices)):
                    reused_token_index_each_item.append(torch.tensor(resued_item_indices) + target_slices[idx][0])
                    unreused_token_index_each_item.append(torch.tensor(unreused_item_indices) + target_slices[idx][0])

                
                target_req_ids = [max_request_id + 1 + idx for idx in range(len(batch_target_prompts))]
                
                # 批量partial计算
                target_outputs,batch_partial_kvcache = KVShareNewPipeline.partial_compute(
                    pipeline.model,
                    SamplingParams(temperature=0, max_tokens=max_generate_len),
                    batch_target_prompts,
                    target_kv_cache,
                    reused_indices,
                    unreused_indices,
                    selected_tokens,
                    target_slices,
                    target_req_ids
                )
                
                max_request_id = max(int(req_output.request_id) for req_output in target_outputs)
                
                # 处理输出结果
                for idx, item in enumerate(valid_batch_items):
                    try:
                        generated_text = target_outputs[idx].outputs[0].text
                        item["reused_top1_w31"][f"output_w{window_size}"] = generated_text
                        item["reused_top1_w31"][f"meteor_w{window_size}"] = meteor_metric.compute(
                            predictions=[generated_text], 
                            references=[batch_references[idx]]
                        )
                        key_error_reused = torch.abs(real_target_kv_cache[:,0,reused_token_index_each_item[idx],:] - batch_partial_kvcache[:,0,reused_token_index_each_item[idx],:])
                        value_error_reused = torch.abs(real_target_kv_cache[:,1,reused_token_index_each_item[idx],:] - batch_partial_kvcache[:,1,reused_token_index_each_item[idx],:])
                
                        # # 保存reused位置的KV误差
                        item["reused_top1_w31"][f"key_error_layer_reused_w{window_size}"] = torch.mean(key_error_reused[:,:,:],dim=[1,2]).cpu().numpy().tolist()
                        item["reused_top1_w31"][f"value_error_layer_reused_w{window_size}"] = torch.mean(value_error_reused[:,:,:],dim=[1,2]).cpu().numpy().tolist()
                        item["reused_top1_w31"][f"key_error_token_reused_w{window_size}"] = torch.mean(key_error_reused[:,:,:],dim=[0,2]).cpu().numpy().tolist()
                        item["reused_top1_w31"][f"value_error_token_reused_w{window_size}"] = torch.mean(value_error_reused[:,:,:],dim=[0,2]).cpu().numpy().tolist()
                        
                        # 计算unreused位置的KV误差
                        key_error_unreused = torch.abs(real_target_kv_cache[:,0,unreused_token_index_each_item[idx],:] - batch_partial_kvcache[:,0,unreused_token_index_each_item[idx],:])
                        value_error_unreused = torch.abs(real_target_kv_cache[:,1,unreused_token_index_each_item[idx],:] - batch_partial_kvcache[:,1,unreused_token_index_each_item[idx],:])

                        # # 保存unreused位置的KV误差
                        item["reused_top1_w31"][f"key_error_layer_unreused_w{window_size}"] = torch.mean(key_error_unreused[:,:,:],dim=[1,2]).cpu().numpy().tolist()
                        item["reused_top1_w31"][f"value_error_layer_unreused_w{window_size}"] = torch.mean(value_error_unreused[:,:,:],dim=[1,2]).cpu().numpy().tolist()
                        item["reused_top1_w31"][f"key_error_token_unreused_w{window_size}"] = torch.mean(key_error_unreused[:,:,:],dim=[0,2]).cpu().numpy().tolist()
                        item["reused_top1_w31"][f"value_error_token_unreused_w{window_size}"] = torch.mean(value_error_unreused[:,:,:],dim=[0,2]).cpu().numpy().tolist()
                        
                    except Exception as e:
                        print(f"处理输出结果时出错: {str(e)}")
                        continue
            
            processed_results.extend(valid_batch_items)
            
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            traceback.print_exc()
            continue
            
    dataset["similar_pairs"] = processed_results + processed_pairs
    json.dump(dataset, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
def plot_bleu_acc(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)
    bleu_acc = []
    for item in data:
        bleu_acc.append(item["bleu"])
    plt.plot(bleu_acc)
    plt.show()

def plot_bleu_comparison(input_path: str, save_path: str = "examples/pipeline/images/opus_bleu_comparison.png"):
    """对比全量计算、相似度top1和重用token top1的BLEU分数分布"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 收集三种方法的BLEU分数
    full_compute_bleu = []
    similarity_top1_bleu = []
    reused_token_top1_bleu = []
    
    for item in data:
        try:
            if item["bleu"]["bleu"] >= 0.001:
                full_compute_bleu.append(item["bleu"]["bleu"])
            
            # 相似度top1的BLEU
            if item["cosine_similarity_top5"] and len(item["cosine_similarity_top5"]) > 0:
                similarity_top1_bleu.append(item["cosine_similarity_top5"][0]["bleu"]["bleu"])
            
            # 重用token top1的BLEU
            if item["reused_token_num_top5"] and len(item["reused_token_num_top5"]) > 0:
                reused_token_top1_bleu.append(item["reused_token_num_top5"][0]["bleu"]["bleu"])
        except Exception as e:
            continue
    
    # 定义统一的颜色方案
    colors = {
        'Full Compute': 'red',      # 红色
        'Similarity Top1': 'green',    # 绿色
        'Reused Token Top1': 'blue'   # 蓝色
    }
    labels = list(colors.keys())
    
    # 创建单个图
    plt.figure(figsize=(4, 3))
    
    # 添加ChatGPT参考线
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Ground Truth')
    
    # 绘制CDF曲线
    data_to_plot = [full_compute_bleu, similarity_top1_bleu, reused_token_top1_bleu]
    
    # 收集所有均值用于统一显示
    means = []
    for scores, label in zip(data_to_plot, labels):
        # 计算CDF
        sorted_scores = np.sort(scores)
        p = np.arange(1, len(scores) + 1) / len(scores)
        
        # 绘制CDF曲线
        plt.plot(sorted_scores, p, label=label, color=colors[label], alpha=0.7)
        
        # 收集均值
        mean_value = np.mean(scores)
        means.append((label, mean_value))
    
    # 统一显示所有均值
    # mean_text = "Means:\n" + "\n".join([f"{label}: {mean:.3f}" for label, mean in means])
    # plt.text(0.5, 0.5, mean_text, transform=plt.gca().transAxes, 
    #          bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('BLEU Score')
    plt.ylabel('Cumulative Probability')
    # plt.title('BLEU Score Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # 将图例放在图表右上角
    plt.legend(loc='lower right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 打印详细统计信息
    print("\nBLEU分数统计:")
    print("-" * 50)
    print(f"全量计算:")
    print(f"  样本数量: {len(full_compute_bleu)}")
    print(f"  平均BLEU: {np.mean(full_compute_bleu):.4f} ± {np.std(full_compute_bleu):.4f}")
    print(f"  中位数: {np.median(full_compute_bleu):.4f}")
    print(f"  范围: [{np.min(full_compute_bleu):.4f}, {np.max(full_compute_bleu):.4f}]")
    
    print(f"\n相似度Top1:")
    print(f"  样本数量: {len(similarity_top1_bleu)}")
    print(f"  平均BLEU: {np.mean(similarity_top1_bleu):.4f} ± {np.std(similarity_top1_bleu):.4f}")
    print(f"  中位数: {np.median(similarity_top1_bleu):.4f}")
    print(f"  范围: [{np.min(similarity_top1_bleu):.4f}, {np.max(similarity_top1_bleu):.4f}]")
    
    print(f"\n重用Token Top1:")
    print(f"  样本数量: {len(reused_token_top1_bleu)}")
    print(f"  平均BLEU: {np.mean(reused_token_top1_bleu):.4f} ± {np.std(reused_token_top1_bleu):.4f}")
    print(f"  中位数: {np.median(reused_token_top1_bleu):.4f}")
    print(f"  范围: [{np.min(reused_token_top1_bleu):.4f}, {np.max(reused_token_top1_bleu):.4f}]")

def save_filtered_data(input_path: str, output_path: str, tag: str = "reused_top1_w31"):
    """保存过滤后的数据点到新的JSON文件"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    window_sizes = [6, 12, 24]
    filtered_pairs = []
    
    for item in data["similar_pairs"]:
        try:
            scores = []
            valid = True
            for window_size in window_sizes:
                if f"meteor_w{window_size}" in item[tag]:
                    score = item[tag][f"meteor_w{window_size}"]["meteor"]
                    if score == 0:  # 去掉分数为0的点
                        valid = False
                        break
                    scores.append(score)
                else:
                    valid = False
                    break
            if valid:
                # 检查所有分数是否相等
                # if not all(x == scores[0] for x in scores):
                filtered_pairs.append(item)
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    # 保存过滤后的数据
    data["similar_pairs"] = filtered_pairs
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"过滤后的数据已保存至: {output_path}")
    print(f"原始数据点数量: {len(data['similar_pairs'])}")
    print(f"过滤后数据点数量: {len(filtered_pairs)}")
    return output_path

def plot_meteor_by_window_size(input_path: str, save_path: str = "examples/pipeline/images/opus_meteor_by_window.png", show_boxplot: bool = False,tag="reused_top1_w31"):
    """绘制不同窗口大小下的METEOR分数的CDF曲线对比，并添加full compute的CDF曲线
    
    Args:
        input_path: 输入数据文件路径
        save_path: 保存图片的路径
        show_boxplot: 是否显示箱线图，默认为False
    """
    # 首先过滤数据
    filtered_path = input_path.replace(".json", "_filtered.json")
    filtered_path = save_filtered_data(input_path, filtered_path,tag)
    
    with open(filtered_path, "r") as f:
        data = json.load(f)
    
    # 收集不同窗口大小的METEOR分数
    window_sizes = [6, 12, 24]  # 根据实际使用的窗口大小调整
    meteor_scores_by_window = {window: [] for window in window_sizes}
    
    # 收集full compute的METEOR分数
    full_compute_scores = []
    
    for item in data["similar_pairs"]:
        try:
            # 收集full compute分数
            if "meteor" in item:
                full_compute_scores.append(item["meteor"])
                
            for window_size in window_sizes:
                if f"meteor_w{window_size}" in item[tag]:
                    score = item[tag][f"meteor_w{window_size}"]["meteor"]
                    meteor_scores_by_window[window_size].append(score)
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    # 定义统一的颜色方案
    colors = {
        6: 'blue',    # 蓝色
        12: 'green',  # 绿色
        24: 'red',    # 红色
        'full': 'black'  # 黑色表示full compute
    }
    
    # 根据是否显示箱线图创建不同的子图布局
    if show_boxplot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    
    # 绘制full compute的CDF曲线
    if full_compute_scores:
        sorted_scores = np.sort(full_compute_scores)
        p = np.arange(1, len(full_compute_scores) + 1) / len(full_compute_scores)
        
        # 在第一个子图上绘制CDF曲线
        ax1.plot(sorted_scores, p, 
                label='Full Compute', 
                color=colors['full'], 
                linestyle='--',
                alpha=0.7)
        
        # 如果显示箱线图，在第二个子图上绘制
        if show_boxplot:
            ax2.boxplot(full_compute_scores, 
                       positions=[0], 
                       widths=2,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors['full'], alpha=0.3),
                       medianprops=dict(color=colors['full'], linewidth=2),
                       whiskerprops=dict(color=colors['full']),
                       capprops=dict(color=colors['full']),
                       flierprops=dict(marker='o', 
                                     markerfacecolor=colors['full'], 
                                     markeredgecolor=colors['full'],
                                     alpha=0.5))
            
            # 在箱线图上添加统计信息
            mean_value = np.mean(full_compute_scores)
            std_value = np.std(full_compute_scores)
            ax2.text(0, np.max(full_compute_scores) + 0.02, 
                    f'μ={mean_value:.3f}\nσ={std_value:.3f}',
                    ha='center', va='bottom',
                    fontsize=8, color=colors['full'])
        
        # 计算并打印full compute的统计信息
        mean_value = np.mean(full_compute_scores)
        std_value = np.std(full_compute_scores)
        print("\nFull Compute统计信息:")
        print(f"  样本数量: {len(full_compute_scores)}")
        print(f"  平均METEOR: {mean_value:.4f} ± {std_value:.4f}")
        print(f"  中位数: {np.median(full_compute_scores):.4f}")
        print(f"  范围: [{np.min(full_compute_scores):.4f}, {np.max(full_compute_scores):.4f}]")
    
    # 绘制CDF曲线
    for window_size in window_sizes:
        scores = meteor_scores_by_window[window_size]
        if len(scores) == 0:
            print(f"窗口大小 {window_size} 没有数据")
            continue
            
        # 计算CDF
        sorted_scores = np.sort(scores)
        p = np.arange(1, len(scores) + 1) / len(scores)
        
        # 在第一个子图上绘制CDF曲线
        ax1.plot(sorted_scores, p, 
                label=f'Window Size {window_size}', 
                color=colors[window_size], 
                alpha=0.7)
        
        # 如果显示箱线图，在第二个子图上绘制
        if show_boxplot:
            ax2.boxplot(scores, 
                       positions=[window_size], 
                       widths=2,
                       patch_artist=True,
                       boxprops=dict(facecolor=colors[window_size], alpha=0.3),
                       medianprops=dict(color=colors[window_size], linewidth=2),
                       whiskerprops=dict(color=colors[window_size]),
                       capprops=dict(color=colors[window_size]),
                       flierprops=dict(marker='o', 
                                     markerfacecolor=colors[window_size], 
                                     markeredgecolor=colors[window_size],
                                     alpha=0.5))
            
            # 在箱线图上添加统计信息
            mean_value = np.mean(scores)
            std_value = np.std(scores)
            ax2.text(window_size, np.max(scores) + 0.02, 
                    f'μ={mean_value:.3f}\nσ={std_value:.3f}',
                    ha='center', va='bottom',
                    fontsize=8, color=colors[window_size])
        
        # 计算并打印统计信息
        mean_value = np.mean(scores)
        median_value = np.median(scores)
        std_value = np.std(scores)
        print(f"\n窗口大小 {window_size}:")
        print(f"  样本数量: {len(scores)}")
        print(f"  平均METEOR: {mean_value:.4f} ± {std_value:.4f}")
        print(f"  中位数: {median_value:.4f}")
        print(f"  范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # 设置第一个子图（CDF）的属性
    ax1.set_xlabel('METEOR Score')
    ax1.set_ylabel('Cumulative Probability')
    # ax1.set_title('METEOR Score CDF by Window Size')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # 如果显示箱线图，设置第二个子图的属性
    if show_boxplot:
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('METEOR Score')
        ax2.set_title('METEOR Score Distribution by Window Size')
        ax2.set_xticks([0] + window_sizes)
        ax2.set_xticklabels(['Full'] + [f'w={w}' for w in window_sizes])
        ax2.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"图表已保存至: {save_path}")

def split_data_by_windows_size(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)
    similar_docs = data["similar_pairs"]
    all_data = data["all_translations"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    save_data = []
    windows_size = [31]
    # random_keys = random.sample(list(similar_docs.keys()),1000)
    for key,doc_item in tqdm(similar_docs.items(),desc="Processing"):
        # if key not in random_keys:
        #     continue
        target_doc = all_data[str(doc_item["id"])]
        target_doc_tokens = tokenizer.encode(target_doc["zh"])
        if len(target_doc_tokens) > 4096:
            continue
        term_items = []
        for reused_item in doc_item["similar_items"]:
            reused_doc = all_data[str(reused_item["id"])]
            if reused_item["similarity"] > 0.9995:
                continue
            reused_doc_tokens = tokenizer.encode(reused_doc["zh"])
            reused_item["reused_token_num"] = {}
            for window_size in windows_size:
                diff_report = KVEditor.find_text_differences(target_doc_tokens,reused_doc_tokens,window_size=window_size)
                if len(diff_report["moves"]) ==0:
                    reused_item["reused_token_num"][window_size] = 0
                else:
                    reused_item["reused_token_num"][window_size] = sum([move["to_position"][1]-move["to_position"][0]+1 for move in diff_report["moves"]])
            if reused_item["reused_token_num"][31] > 0:
                term_items.append(reused_item)
        # doc_item["similar_items"] = term_items
        if len(term_items) == 0:
            continue
        else:
            simi_top1 = sorted(term_items,key=lambda x:x["similarity"],reverse=True)[0]
            # reused_top1_w3 = sorted(term_items,key=lambda x:x["reused_token_num"][3],reverse=True)[0]
            # reused_top1_w7 = sorted(term_items,key=lambda x:x["reused_token_num"][7],reverse=True)[0]
            # reused_top1_w15 = sorted(term_items,key=lambda x:x["reused_token_num"][15],reverse=True)[0]
            reused_top1_w31 = sorted(term_items,key=lambda x:x["reused_token_num"][31],reverse=True)[0]
            doc_item["simi_top1"] = simi_top1["id"]
            doc_item["reused_items"] = term_items
            save_data.append({
                "id": doc_item["id"],
                "simi_top1": simi_top1["id"],
                # "reused_top1_w3": reused_top1_w3,
                # "reused_top1_w7": reused_top1_w7,
                # "reused_top1_w15": reused_top1_w15,
                "reused_top1_w31": reused_top1_w31
            })
    data["similar_pairs"] = save_data   
    print(f"处理后样本数量: {len(data['similar_pairs'])}")
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)

def compute_full_compute_acc(input_path,output_path,model_name,batch_size=32):
    with open(input_path, "r") as f:
        data = json.load(f)
    similar_pairs = data["similar_pairs"]
    all_data = data["all_translations"]
    meteor = evaluate.load('meteor')
    pipeline = KVShareNewPipeline(model_name=model_name)
    save_data = []
    # similar_pairs = random.sample(similar_pairs,24)
    for i in tqdm(range(0, len(similar_pairs), batch_size), desc="Processing batches"):
        try:
            batch_items = similar_pairs[i:i + batch_size]
            # 准备所有prompt
            all_target_prompts = []
            batch_answers = []
            
            for item in batch_items:
                question = all_data[str(item["id"])]["zh"]
                answer = all_data[str(item["id"])]["en"]
                batch_answers.append(answer)
                all_target_prompts.append(template.format(text=question))
                     
            # 批量计算full compute
            full_compute_outputs = KVShareNewPipeline.batch_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=512),
                all_target_prompts
            )

            for idx,item in enumerate(batch_items):
                acc = meteor.compute(predictions=[full_compute_outputs[idx].outputs[0].text], references=[batch_answers[idx]])
                item["output"] = full_compute_outputs[idx].outputs[0].text
                item["meteor"] = acc["meteor"]
            
                save_data.append(item)
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            continue
    data["similar_pairs"] = save_data
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
def plot_similarity_vs_meteor(input_path: str, save_path: str = "examples/pipeline/images/opus_similarity_vs_meteor.png"):
    """绘制不同相似度区间内的METEOR分数均值和方差，并添加full compute的CDF曲线"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    window_sizes = [6, 12, 24]
    # 定义相似度区间
    similarity_bins = np.linspace(0, 1, 11)  # 将相似度分成10个区间
    bin_centers = (similarity_bins[:-1] + similarity_bins[1:]) / 2
    
    # 为每个窗口大小准备数据
    bin_data = {window: {
        'means': [],
        'stds': [],
        'counts': []
    } for window in window_sizes}
    
    # 收集full compute的METEOR分数
    full_compute_scores = []
    
    # 收集数据
    for item in data["similar_pairs"]:
        try:
            # 收集full compute分数
            if "meteor" in item:
                full_compute_scores.append(item["meteor"])
            
            similarity = item["reused_top1_w31"]["similarity"]
            for window_size in window_sizes:
                if f"meteor_w{window_size}" in item["reused_top1_w31"]:
                    score = item["reused_top1_w31"][f"meteor_w{window_size}"]["meteor"]
                    if score > 0:  # 只保留有效分数
                        # 找到对应的区间
                        bin_idx = np.digitize(similarity, similarity_bins) - 1
                        if bin_idx < len(bin_centers):  # 确保在有效区间内
                            if 'scores' not in bin_data[window_size]:
                                bin_data[window_size]['scores'] = [[] for _ in range(len(bin_centers))]
                            bin_data[window_size]['scores'][bin_idx].append(score)
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    # 计算每个区间的统计量
    for window_size in window_sizes:
        if 'scores' in bin_data[window_size]:
            for scores in bin_data[window_size]['scores']:
                if len(scores) > 0:
                    bin_data[window_size]['means'].append(np.mean(scores))
                    bin_data[window_size]['stds'].append(np.std(scores))
                    bin_data[window_size]['counts'].append(len(scores))
                else:
                    bin_data[window_size]['means'].append(np.nan)
                    bin_data[window_size]['stds'].append(np.nan)
                    bin_data[window_size]['counts'].append(0)
    
    # 定义颜色方案
    colors = {
        6: 'blue',    # 蓝色
        12: 'green',  # 绿色
        24: 'red',    # 红色
        'full': 'black'  # 黑色表示full compute
    }
    
    # 创建图表
    plt.figure(figsize=(8, 6))
    
    # 绘制full compute的CDF曲线
    if full_compute_scores:
        sorted_scores = np.sort(full_compute_scores)
        p = np.arange(1, len(full_compute_scores) + 1) / len(full_compute_scores)
        plt.plot(sorted_scores, p, 
                label='Full Compute', 
                color=colors['full'], 
                linestyle='--',
                alpha=0.7)
        
        # 打印full compute的统计信息
        mean_value = np.mean(full_compute_scores)
        std_value = np.std(full_compute_scores)
        print("\nFull Compute统计信息:")
        print(f"  样本数量: {len(full_compute_scores)}")
        print(f"  平均METEOR: {mean_value:.4f} ± {std_value:.4f}")
        print(f"  中位数: {np.median(full_compute_scores):.4f}")
        print(f"  范围: [{np.min(full_compute_scores):.4f}, {np.max(full_compute_scores):.4f}]")
    
    # 绘制误差棒图
    for window_size in window_sizes:
        if len(bin_data[window_size]['means']) > 0:
            # 过滤掉没有数据的区间
            valid_mask = ~np.isnan(bin_data[window_size]['means'])
            valid_centers = bin_centers[valid_mask]
            valid_means = np.array(bin_data[window_size]['means'])[valid_mask]
            valid_stds = np.array(bin_data[window_size]['stds'])[valid_mask]
            valid_counts = np.array(bin_data[window_size]['counts'])[valid_mask]
            
            # 绘制误差棒
            plt.errorbar(
                valid_centers,
                valid_means,
                yerr=valid_stds,
                label=f'Window Size {window_size}',
                color=colors[window_size],
                fmt='o-',
                capsize=5,
                alpha=0.7
            )
            
            # 在每个点上显示样本数量
            for x, y, count in zip(valid_centers, valid_means, valid_counts):
                plt.text(x, y, f'n={count}', 
                        ha='center', va='bottom',
                        fontsize=8, color=colors[window_size])
    
    plt.xlabel('Similarity')
    plt.ylabel('METEOR Score (Mean ± Std)')
    plt.title('METEOR Score by Similarity Range and Window Size')
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(loc='upper left')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"图表已保存至: {save_path}")
    
    # 打印详细统计信息
    print("\n详细统计信息:")
    for window_size in window_sizes:
        print(f"\n窗口大小 {window_size}:")
        for i, (center, mean, std, count) in enumerate(zip(
            bin_centers,
            bin_data[window_size]['means'],
            bin_data[window_size]['stds'],
            bin_data[window_size]['counts']
        )):
            if count > 0:
                print(f"相似度区间 {similarity_bins[i]:.2f}-{similarity_bins[i+1]:.2f}:")
                print(f"  样本数量: {count}")
                print(f"  平均分数: {mean:.4f} ± {std:.4f}")

def plot_zero_score_kv_error(input_path: str, save_path: str = "examples/pipeline/images/zero_score_kv_error.png", window_size: int = 20):
    """可视化分数为0的样本的KV误差分布
    
    Args:
        input_path: 输入数据文件路径
        save_path: 保存图片的路径
        window_size: 窗口大小
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 收集分数为0的样本的KV误差
    key_error_reused = []
    value_error_reused = []
    key_error_unreused = []
    value_error_unreused = []
    token_positions_reused = []
    token_positions_unreused = []
    layer_key_errors_reused = {}  # 存储每层的Key误差
    layer_value_errors_reused = {}  # 存储每层的Value误差
    layer_key_errors_unreused = {}  # 存储每层的Key误差
    layer_value_errors_unreused = {}  # 存储每层的Value误差
    
    for item in data["similar_pairs"]:
        try:
            if f"meteor_w{window_size}" in item["reused_top1_w31"]:
                score = item["reused_top1_w31"][f"meteor_w{window_size}"]["meteor"]
                if score == 0:
                    # 收集reused位置的误差
                    key_errors_reused = item["reused_top1_w31"][f"key_error_token_reused_w{window_size}"]
                    value_errors_reused = item["reused_top1_w31"][f"value_error_token_reused_w{window_size}"]
                    
                    # 记录token位置
                    positions = list(range(len(key_errors_reused)))
                    token_positions_reused.extend(positions)
                    key_error_reused.extend(key_errors_reused)
                    value_error_reused.extend(value_errors_reused)
                    
                    # 收集每层的误差
                    layer_key_errors_item = item["reused_top1_w31"][f"key_error_layer_reused_w{window_size}"]
                    layer_value_errors_item = item["reused_top1_w31"][f"value_error_layer_reused_w{window_size}"]
                    
                    for layer_idx, (key_err, value_err) in enumerate(zip(layer_key_errors_item, layer_value_errors_item)):
                        if layer_idx not in layer_key_errors_reused:
                            layer_key_errors_reused[layer_idx] = []
                            layer_value_errors_reused[layer_idx] = []
                        layer_key_errors_reused[layer_idx].append(key_err)
                        layer_value_errors_reused[layer_idx].append(value_err)
                    
                    # 收集unreused位置的误差
                    key_errors_unreused = item["reused_top1_w31"][f"key_error_token_unreused_w{window_size}"]
                    value_errors_unreused = item["reused_top1_w31"][f"value_error_token_unreused_w{window_size}"]
                    
                    # 记录token位置
                    positions = list(range(len(key_errors_unreused)))
                    token_positions_unreused.extend(positions)
                    key_error_unreused.extend(key_errors_unreused)
                    value_error_unreused.extend(value_errors_unreused)
                    
                    # 收集每层的误差
                    layer_key_errors_item = item["reused_top1_w31"][f"key_error_layer_unreused_w{window_size}"]
                    layer_value_errors_item = item["reused_top1_w31"][f"value_error_layer_unreused_w{window_size}"]
                    
                    for layer_idx, (key_err, value_err) in enumerate(zip(layer_key_errors_item, layer_value_errors_item)):
                        if layer_idx not in layer_key_errors_unreused:
                            layer_key_errors_unreused[layer_idx] = []
                            layer_value_errors_unreused[layer_idx] = []
                        layer_key_errors_unreused[layer_idx].append(key_err)
                        layer_value_errors_unreused[layer_idx].append(value_err)
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    # 创建图表
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 绘制reused token的Key误差随token位置的变化
    ax1 = fig.add_subplot(gs[0, 0])
    sns.scatterplot(x=token_positions_reused, y=key_error_reused, ax=ax1, alpha=0.5)
    sns.lineplot(x=token_positions_reused, y=key_error_reused, ax=ax1, color='red', label='移动平均')
    ax1.set_title(f'Reused Token Key Error by Position (Window Size {window_size})')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Error Value')
    ax1.grid(True, alpha=0.3)
    
    # 绘制reused token的Value误差随token位置的变化
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(x=token_positions_reused, y=value_error_reused, ax=ax2, alpha=0.5)
    sns.lineplot(x=token_positions_reused, y=value_error_reused, ax=ax2, color='red', label='移动平均')
    ax2.set_title(f'Reused Token Value Error by Position (Window Size {window_size})')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Error Value')
    ax2.grid(True, alpha=0.3)
    
    # 绘制unreused token的Key误差随token位置的变化
    ax3 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(x=token_positions_unreused, y=key_error_unreused, ax=ax3, alpha=0.5)
    sns.lineplot(x=token_positions_unreused, y=key_error_unreused, ax=ax3, color='red', label='移动平均')
    ax3.set_title(f'Unreused Token Key Error by Position (Window Size {window_size})')
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Error Value')
    ax3.grid(True, alpha=0.3)
    
    # 绘制unreused token的Value误差随token位置的变化
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(x=token_positions_unreused, y=value_error_unreused, ax=ax4, alpha=0.5)
    sns.lineplot(x=token_positions_unreused, y=value_error_unreused, ax=ax4, color='red', label='移动平均')
    ax4.set_title(f'Unreused Token Value Error by Position (Window Size {window_size})')
    ax4.set_xlabel('Token Position')
    ax4.set_ylabel('Error Value')
    ax4.grid(True, alpha=0.3)
    
    # 绘制reused token的层级误差
    ax5 = fig.add_subplot(gs[2, 0])
    layer_indices = sorted(layer_key_errors_reused.keys())
    layer_key_means_reused = [np.mean(layer_key_errors_reused[idx]) for idx in layer_indices]
    layer_value_means_reused = [np.mean(layer_value_errors_reused[idx]) for idx in layer_indices]
    
    ax5.errorbar(layer_indices, layer_key_means_reused, yerr=layer_value_means_reused, fmt='o-', capsize=5, label='Key Error')
    ax5.set_title('Reused Token Error by Layer')
    ax5.set_xlabel('Layer Index')
    ax5.set_ylabel('Error Value')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 绘制unreused token的层级误差
    ax6 = fig.add_subplot(gs[2, 1])
    layer_indices = sorted(layer_key_errors_unreused.keys())
    layer_key_means_unreused = [np.mean(layer_key_errors_unreused[idx]) for idx in layer_indices]
    layer_value_means_unreused = [np.mean(layer_value_errors_unreused[idx]) for idx in layer_indices]
    
    ax6.errorbar(layer_indices, layer_key_means_unreused, yerr=layer_value_means_unreused, fmt='o-', capsize=5, label='Key Error')
    ax6.set_title('Unreused Token Error by Layer')
    ax6.set_xlabel('Layer Index')
    ax6.set_ylabel('Error Value')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 计算每个位置的统计信息
    position_stats_reused = {}
    for pos, key_err, value_err in zip(token_positions_reused, key_error_reused, value_error_reused):
        if pos not in position_stats_reused:
            position_stats_reused[pos] = {'key_errors': [], 'value_errors': []}
        position_stats_reused[pos]['key_errors'].append(key_err)
        position_stats_reused[pos]['value_errors'].append(value_err)
    
    position_stats_unreused = {}
    for pos, key_err, value_err in zip(token_positions_unreused, key_error_unreused, value_error_unreused):
        if pos not in position_stats_unreused:
            position_stats_unreused[pos] = {'key_errors': [], 'value_errors': []}
        position_stats_unreused[pos]['key_errors'].append(key_err)
        position_stats_unreused[pos]['value_errors'].append(value_err)
    
    # 打印统计信息
    print("\nReused Token位置误差统计信息:")
    print("-" * 50)
    print("位置\tKey误差均值\tKey误差标准差\tValue误差均值\tValue误差标准差")
    print("-" * 80)
    
    for pos in sorted(position_stats_reused.keys()):
        key_mean = np.mean(position_stats_reused[pos]['key_errors'])
        key_std = np.std(position_stats_reused[pos]['key_errors'])
        value_mean = np.mean(position_stats_reused[pos]['value_errors'])
        value_std = np.std(position_stats_reused[pos]['value_errors'])
        print(f"{pos}\t{key_mean:.4f}\t{key_std:.4f}\t{value_mean:.4f}\t{value_std:.4f}")
    
    print("\nUnreused Token位置误差统计信息:")
    print("-" * 50)
    print("位置\tKey误差均值\tKey误差标准差\tValue误差均值\tValue误差标准差")
    print("-" * 80)
    
    for pos in sorted(position_stats_unreused.keys()):
        key_mean = np.mean(position_stats_unreused[pos]['key_errors'])
        key_std = np.std(position_stats_unreused[pos]['key_errors'])
        value_mean = np.mean(position_stats_unreused[pos]['value_errors'])
        value_std = np.std(position_stats_unreused[pos]['value_errors'])
        print(f"{pos}\t{key_mean:.4f}\t{key_std:.4f}\t{value_mean:.4f}\t{value_std:.4f}")
    
    print("\n层误差统计信息:")
    print("-" * 50)
    print("层\tReused Key均值\tReused Key标准差\tReused Value均值\tReused Value标准差\t" +
          "Unreused Key均值\tUnreused Key标准差\tUnreused Value均值\tUnreused Value标准差")
    print("-" * 120)
    
    for layer_idx in sorted(layer_key_errors_reused.keys()):
        key_mean_reused = np.mean(layer_key_errors_reused[layer_idx])
        key_std_reused = np.std(layer_key_errors_reused[layer_idx])
        value_mean_reused = np.mean(layer_value_errors_reused[layer_idx])
        value_std_reused = np.std(layer_value_errors_reused[layer_idx])
        
        key_mean_unreused = np.mean(layer_key_errors_unreused[layer_idx])
        key_std_unreused = np.std(layer_key_errors_unreused[layer_idx])
        value_mean_unreused = np.mean(layer_value_errors_unreused[layer_idx])
        value_std_unreused = np.std(layer_value_errors_unreused[layer_idx])
        
        print(f"{layer_idx}\t{key_mean_reused:.4f}\t{key_std_reused:.4f}\t{value_mean_reused:.4f}\t{value_std_reused:.4f}\t" +
              f"{key_mean_unreused:.4f}\t{key_std_unreused:.4f}\t{value_mean_unreused:.4f}\t{value_std_unreused:.4f}")
    
    print(f"\n图表已保存至: {save_path}")

def plot_kv_error_by_window_size(input_path: str, save_path: str = "examples/pipeline/images/kv_error_by_window.png"):
    """分析不同窗口大小下0分和非0分样本的unreused token的KV误差分布
    
    Args:
        input_path: 输入数据文件路径
        save_path: 保存图片的路径
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    window_sizes = [5,10,15,20,25]
    window_stats = {size: {
        'layer_key_errors_unreused_zero': {},
        'layer_value_errors_unreused_zero': {},
        'layer_key_errors_unreused_nonzero': {},
        'layer_value_errors_unreused_nonzero': {},
        'zero_count': 0,
        'nonzero_count': 0
    } for size in window_sizes}
    
    # 收集数据
    for item in data["similar_pairs"]:
        try:
            for window_size in window_sizes:
                if f"meteor_w{window_size}" in item["reused_top1_w31"]:
                    meteor_score = item["reused_top1_w31"][f"meteor_w{window_size}"]["meteor"]
                    is_zero = meteor_score == 0
                    
                    # 更新计数
                    if is_zero:
                        window_stats[window_size]['zero_count'] += 1
                    else:
                        window_stats[window_size]['nonzero_count'] += 1
                        
                    # 收集层级误差
                    layer_key_errors_unreused = item["reused_top1_w31"][f"key_error_layer_unreused_w{window_size}"]
                    layer_value_errors_unreused = item["reused_top1_w31"][f"value_error_layer_unreused_w{window_size}"]
                    
                    # 存储层级误差
                    target_dict_key = 'layer_key_errors_unreused_zero' if is_zero else 'layer_key_errors_unreused_nonzero'
                    target_dict_value = 'layer_value_errors_unreused_zero' if is_zero else 'layer_value_errors_unreused_nonzero'
                    
                    for layer_idx, (key_err, value_err) in enumerate(zip(
                        layer_key_errors_unreused, layer_value_errors_unreused)):
                        
                        for target_dict, err_value in [
                            (target_dict_key, key_err),
                            (target_dict_value, value_err)
                        ]:
                            if layer_idx not in window_stats[window_size][target_dict]:
                                window_stats[window_size][target_dict][layer_idx] = []
                            window_stats[window_size][target_dict][layer_idx].append(err_value)
                            
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    # 创建图表
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # 为每个窗口大小创建子图
    for i, window_size in enumerate(window_sizes):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # 获取层索引
        layer_indices = sorted(window_stats[window_size]['layer_key_errors_unreused_zero'].keys())
        
        # 计算每层的平均误差
        # 零分样本
        layer_key_means_zero = [np.mean(window_stats[window_size]['layer_key_errors_unreused_zero'][idx]) for idx in layer_indices]
        layer_value_means_zero = [np.mean(window_stats[window_size]['layer_value_errors_unreused_zero'][idx]) for idx in layer_indices]
        layer_key_std_zero = [np.std(window_stats[window_size]['layer_key_errors_unreused_zero'][idx]) for idx in layer_indices]
        layer_value_std_zero = [np.std(window_stats[window_size]['layer_value_errors_unreused_zero'][idx]) for idx in layer_indices]
        
        # 非零分样本
        layer_key_means_nonzero = [np.mean(window_stats[window_size]['layer_key_errors_unreused_nonzero'][idx]) for idx in layer_indices]
        layer_value_means_nonzero = [np.mean(window_stats[window_size]['layer_value_errors_unreused_nonzero'][idx]) for idx in layer_indices]
        layer_key_std_nonzero = [np.std(window_stats[window_size]['layer_key_errors_unreused_nonzero'][idx]) for idx in layer_indices]
        layer_value_std_nonzero = [np.std(window_stats[window_size]['layer_value_errors_unreused_nonzero'][idx]) for idx in layer_indices]
        
        # 绘制误差曲线和误差带
        # 零分样本
        ax.plot(layer_indices, layer_key_means_zero, 'o-', label='Key (Zero)', color='red')
        ax.fill_between(layer_indices, 
                       [m - s for m, s in zip(layer_key_means_zero, layer_key_std_zero)],
                       [m + s for m, s in zip(layer_key_means_zero, layer_key_std_zero)],
                       alpha=0.2, color='red')
        
        ax.plot(layer_indices, layer_value_means_zero, 's--', label='Value (Zero)', color='blue')
        ax.fill_between(layer_indices,
                       [m - s for m, s in zip(layer_value_means_zero, layer_value_std_zero)],
                       [m + s for m, s in zip(layer_value_means_zero, layer_value_std_zero)],
                       alpha=0.2, color='blue')
        
        # 非零分样本
        ax.plot(layer_indices, layer_key_means_nonzero, '^-', label='Key (Nonzero)', color='orange')
        ax.fill_between(layer_indices,
                       [m - s for m, s in zip(layer_key_means_nonzero, layer_key_std_nonzero)],
                       [m + s for m, s in zip(layer_key_means_nonzero, layer_key_std_nonzero)],
                       alpha=0.2, color='orange')
        
        ax.plot(layer_indices, layer_value_means_nonzero, 'v--', label='Value (Nonzero)', color='green')
        ax.fill_between(layer_indices,
                       [m - s for m, s in zip(layer_value_means_nonzero, layer_value_std_nonzero)],
                       [m + s for m, s in zip(layer_value_means_nonzero, layer_value_std_nonzero)],
                       alpha=0.2, color='green')
        
        # 设置子图属性
        ax.set_title(f'Window Size {window_size}\nZero: {window_stats[window_size]["zero_count"]}, Nonzero: {window_stats[window_size]["nonzero_count"]}')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Mean Error Value (with std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 打印统计信息
    print("\n每个窗口大小的unreused token层间误差统计信息:")
    print("-" * 50)
    print("窗口大小\t样本类型\t样本数\t层数\tKey均值\tKey标准差\tValue均值\tValue标准差")
    print("-" * 100)
    
    for window_size in window_sizes:
        layer_indices = sorted(window_stats[window_size]['layer_key_errors_unreused_zero'].keys())
        
        # 零分样本统计
        key_mean_zero = np.mean([np.mean(window_stats[window_size]['layer_key_errors_unreused_zero'][idx]) for idx in layer_indices])
        key_std_zero = np.mean([np.std(window_stats[window_size]['layer_key_errors_unreused_zero'][idx]) for idx in layer_indices])
        value_mean_zero = np.mean([np.mean(window_stats[window_size]['layer_value_errors_unreused_zero'][idx]) for idx in layer_indices])
        value_std_zero = np.mean([np.std(window_stats[window_size]['layer_value_errors_unreused_zero'][idx]) for idx in layer_indices])
        
        print(f"{window_size}\tZero\t{window_stats[window_size]['zero_count']}\t{len(layer_indices)}\t" +
              f"{key_mean_zero:.4f}\t{key_std_zero:.4f}\t{value_mean_zero:.4f}\t{value_std_zero:.4f}")
        
        # 非零分样本统计
        key_mean_nonzero = np.mean([np.mean(window_stats[window_size]['layer_key_errors_unreused_nonzero'][idx]) for idx in layer_indices])
        key_std_nonzero = np.mean([np.std(window_stats[window_size]['layer_key_errors_unreused_nonzero'][idx]) for idx in layer_indices])
        value_mean_nonzero = np.mean([np.mean(window_stats[window_size]['layer_value_errors_unreused_nonzero'][idx]) for idx in layer_indices])
        value_std_nonzero = np.mean([np.std(window_stats[window_size]['layer_value_errors_unreused_nonzero'][idx]) for idx in layer_indices])
        
        print(f"{window_size}\tNonzero\t{window_stats[window_size]['nonzero_count']}\t{len(layer_indices)}\t" +
              f"{key_mean_nonzero:.4f}\t{key_std_nonzero:.4f}\t{value_mean_nonzero:.4f}\t{value_std_nonzero:.4f}")
    
    print(f"\n图表已保存至: {save_path}")

def generate_reuse_output(text_a: str, text_b: str, text_c: str, template: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct", max_model_len=8192, max_generate_len=512, window_size=7):
    """Compare generation results when reusing KV Cache from text B and C for text A
    
    Args:
        text_a: Target text
        text_b: First source text
        text_c: Second source text
        model_name: Model name
        max_model_len: Maximum model length
        max_generate_len: Maximum generation length
        
    Returns:
        dict: Contains generation results and attention values
    """
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device, max_model_len)
    tokenizer = pipeline.model.get_tokenizer()
    
    # Prepare prompts
    prompt_a = template.format(text=text_a)
    prompt_b = template.format(text=text_b)
    prompt_c = template.format(text=text_c)
    
    # Check token length
    tokens_a = tokenizer.encode(prompt_a)
    tokens_b = tokenizer.encode(prompt_b)
    tokens_c = tokenizer.encode(prompt_c)
    
    if len(tokens_a) > max_model_len or len(tokens_b) > max_model_len or len(tokens_c) > max_model_len:
        raise ValueError("Text length exceeds maximum model length limit")
    
    results = {
        'full_compute': None,
        'reuse_b': None,
        'reuse_c': None,
        'source_outputs': {
            'b': None,
            'c': None
        },
        'attention_values': {
            'full_compute': None,
            'reuse_b': None,
            'reuse_c': None
        },
        'kv_errors': {
            'reuse_b': {
                'key_errors': [],  # 按层统计的Key误差
                'value_errors': [],  # 按层统计的Value误差
                'token_key_errors': [],  # 按token位置统计的Key误差
                'token_value_errors': []  # 按token位置统计的Value误差
            },
            'reuse_c': {
                'key_errors': [],
                'value_errors': [],
                'token_key_errors': [],
                'token_value_errors': []
            }
        }
    }
    
    # Get full computation KV cache as baseline
    full_kv_cache, full_outputs, _, full_attn = KVShareNewPipeline.get_kvcache_by_full_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_a]
    )
    results['attention_values']['full_compute'] = full_attn[0]
    results['full_compute'] = full_outputs[0].outputs[0].text
    
    # Get KV cache for B and C
    source_kv_cache_b, source_outputs_b, _, source_attn_b = KVShareNewPipeline.get_kvcache_by_full_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_b]
    )
    results['source_outputs']['b'] = source_outputs_b[0].outputs[0].text
    
    source_kv_cache_c, source_outputs_c, _, source_attn_c = KVShareNewPipeline.get_kvcache_by_full_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_c]
    )
    results['source_outputs']['c'] = source_outputs_c[0].outputs[0].text
    
    source_token_ids_b = [output.prompt_token_ids for output in source_outputs_b]
    source_token_ids_c = [output.prompt_token_ids for output in source_outputs_c]
    
    # Reuse KV Cache from B
    target_kv_cache_b, reused_indices_b, unreused_indices_b, selected_tokens_b, target_slices_b = KVEditor.batch_kvedit(
        [tokens_a],
        source_token_ids_b,
        source_kv_cache_b,
        window_size=window_size
    )
    
    # Reuse KV Cache from C
    target_kv_cache_c, reused_indices_c, unreused_indices_c, selected_tokens_c, target_slices_c = KVEditor.batch_kvedit(
        [tokens_a],
        source_token_ids_c,
        source_kv_cache_c,
        window_size=window_size
    )
    
    max_request_id = int(source_outputs_c[0].request_id)
    
    # Generate using B's KV Cache
    outputs_b, partial_kv_cache_b, batch_attn_b = KVShareNewPipeline.partial_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_a],
        target_kv_cache_b,
        reused_indices_b,
        unreused_indices_b,
        selected_tokens_b,
        target_slices_b,
        [max_request_id+1]
    )
    results['attention_values']['reuse_b'] = batch_attn_b[0]
    
    # 计算B的KV误差
    # 按层统计误差
    num_layers = full_kv_cache.shape[0]
    for layer_idx in range(num_layers):
        # Key误差
        key_error = torch.abs(full_kv_cache[layer_idx, 0] - partial_kv_cache_b[layer_idx, 0])
        results['kv_errors']['reuse_b']['key_errors'].append(torch.mean(key_error).item())
        
        # Value误差
        value_error = torch.abs(full_kv_cache[layer_idx, 1] - partial_kv_cache_b[layer_idx, 1])
        results['kv_errors']['reuse_b']['value_errors'].append(torch.mean(value_error).item())
    
    # 按token位置统计误差
    for token_idx in range(len(tokens_a)):
        # Key误差
        key_error = torch.abs(full_kv_cache[:, 0, token_idx] - partial_kv_cache_b[:, 0, token_idx])
        results['kv_errors']['reuse_b']['token_key_errors'].append(torch.mean(key_error).item())
        
        # Value误差
        value_error = torch.abs(full_kv_cache[:, 1, token_idx] - partial_kv_cache_b[:, 1, token_idx])
        results['kv_errors']['reuse_b']['token_value_errors'].append(torch.mean(value_error).item())
    
    max_request_id = int(outputs_b[0].request_id)
    
    # Generate using C's KV Cache
    outputs_c, partial_kv_cache_c, batch_attn_c = KVShareNewPipeline.partial_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_a],
        target_kv_cache_c,
        reused_indices_c,
        unreused_indices_c,
        selected_tokens_c,
        target_slices_c,
        [max_request_id+1]
    )
    results['attention_values']['reuse_c'] = batch_attn_c[0]
    
    # 计算C的KV误差
    # 按层统计误差
    for layer_idx in range(num_layers):
        # Key误差
        key_error = torch.abs(full_kv_cache[layer_idx, 0] - partial_kv_cache_c[layer_idx, 0])
        results['kv_errors']['reuse_c']['key_errors'].append(torch.mean(key_error).item())
        
        # Value误差
        value_error = torch.abs(full_kv_cache[layer_idx, 1] - partial_kv_cache_c[layer_idx, 1])
        results['kv_errors']['reuse_c']['value_errors'].append(torch.mean(value_error).item())
    
    # 按token位置统计误差
    for token_idx in range(len(tokens_a)):
        # Key误差
        key_error = torch.abs(full_kv_cache[:, 0, token_idx] - partial_kv_cache_c[:, 0, token_idx])
        results['kv_errors']['reuse_c']['token_key_errors'].append(torch.mean(key_error).item())
        
        # Value误差
        value_error = torch.abs(full_kv_cache[:, 1, token_idx] - partial_kv_cache_c[:, 1, token_idx])
        results['kv_errors']['reuse_c']['token_value_errors'].append(torch.mean(value_error).item())
    
    # Save results
    results['reuse_b'] = {
        'output': outputs_b[0].outputs[0].text,
        'reused_tokens': len(reused_indices_b[0]),
        'unreused_tokens': len(unreused_indices_b[0]),
        'reused_indices': reused_indices_b[0],
        'unreused_indices': unreused_indices_b[0]
    }
    
    results['reuse_c'] = {
        'output': outputs_c[0].outputs[0].text,
        'reused_tokens': len(reused_indices_c[0]),
        'unreused_tokens': len(unreused_indices_c[0]),
        'reused_indices': reused_indices_c[0],
        'unreused_indices': unreused_indices_c[0]
    }
    
    # Save token IDs for visualization
    results['token_ids'] = tokens_a
    
    return results

def print_reuse_results(results: dict):
    """Print comparison of reuse results
    
    Args:
        results: Return value from generate_reuse_output
    """
    print("\nFull Computation Result:")
    print("-" * 50)
    print(results['full_compute'])
    
    # print("\nSource Text B Output:")
    # print("-" * 50)
    # print(results['source_outputs']['b'])
    
    # print("\nSource Text C Output:")
    # print("-" * 50)
    # print(results['source_outputs']['c'])
    
    print("\nReusing Text B Result:")
    print("-" * 50)
    # print(f"Reused tokens: {results['reuse_b']['reused_tokens']}")
    # print(f"Unreused tokens: {results['reuse_b']['unreused_tokens']}")
    print(f"Generated output: {results['reuse_b']['output']}")
    
    print("\nReusing Text C Result:")
    print("-" * 50)
    # print(f"Reused tokens: {results['reuse_c']['reused_tokens']}")
    # print(f"Unreused tokens: {results['reuse_c']['unreused_tokens']}")
    print(f"Generated output: {results['reuse_c']['output']}")

def plot_kv_error_comparison(results: dict, save_path: str = "examples/pipeline/images/kv_error_comparison.png"):
    """Visualize KV error comparison when reusing KV Cache from B and C
    
    Args:
        results: Return value from generate_reuse_output
        save_path: Path to save the image
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get number of layers
    num_layers = len(results['kv_errors']['reuse_b']['key_errors'])
    layer_indices = range(num_layers)
    
    # Plot Key error comparison
    ax1.plot(layer_indices, results['kv_errors']['reuse_b']['key_errors'], 'o-', label='Reuse B', color='red')
    ax1.plot(layer_indices, results['kv_errors']['reuse_c']['key_errors'], 's--', label='Reuse C', color='blue')
    ax1.set_title('Key Error Comparison')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Value error comparison
    ax2.plot(layer_indices, results['kv_errors']['reuse_b']['value_errors'], 'o-', label='Reuse B', color='red')
    ax2.plot(layer_indices, results['kv_errors']['reuse_c']['value_errors'], 's--', label='Reuse C', color='blue')
    ax2.set_title('Value Error Comparison')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Average Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Print statistics
    print("\nKV Error Statistics:")
    print("-" * 50)
    print("Method\tKey Mean\tKey Std\tValue Mean\tValue Std")
    print("-" * 80)
    
    for method, label in [('reuse_b', 'Reuse B'), ('reuse_c', 'Reuse C')]:
        key_errors = results['kv_errors'][method]['key_errors']
        value_errors = results['kv_errors'][method]['value_errors']
        
        print(f"{label}\t{np.mean(key_errors):.4f}\t{np.std(key_errors):.4f}\t" +
              f"{np.mean(value_errors):.4f}\t{np.std(value_errors):.4f}")
    
    print(f"\nFigure saved to: {save_path}")

def plot_token_kv_errors(results: dict, tokenizer, save_path: str = "examples/pipeline/images/token_kv_errors.png"):
    """可视化每个token位置的Key和Value误差趋势
    
    Args:
        results: generate_reuse_output的返回结果
        tokenizer: 用于解码token的tokenizer
        save_path: 保存图片的路径
    """
    # 获取token对应的单词
    token_words = [tokenizer.decode([token]) for token in results['token_ids']]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 获取误差数据
    token_key_errors_b = results['kv_errors']['reuse_b']['token_key_errors']
    token_value_errors_b = results['kv_errors']['reuse_b']['token_value_errors']
    token_key_errors_c = results['kv_errors']['reuse_c']['token_key_errors']
    token_value_errors_c = results['kv_errors']['reuse_c']['token_value_errors']
    
    # 绘制Key误差趋势
    ax1.plot(token_key_errors_b, 'o-', label='Reuse B', color='blue')
    ax1.plot(token_key_errors_c, 's--', label='Reuse C', color='red')
    ax1.set_title('Key Error by Token Position')
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Average Key Error')
    ax1.set_xticks(range(len(token_words)))
    ax1.set_xticklabels(token_words, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制Value误差趋势
    ax2.plot(token_value_errors_b, 'o-', label='Reuse B', color='blue')
    ax2.plot(token_value_errors_c, 's--', label='Reuse C', color='red')
    ax2.set_title('Value Error by Token Position')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Average Value Error')
    ax2.set_xticks(range(len(token_words)))
    ax2.set_xticklabels(token_words, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Token位置KV误差趋势图已保存至: {save_path}")
    
    # 打印统计信息
    print("\nToken位置KV误差统计信息:")
    print("-" * 50)
    print("Token\tWord\tKey Error (B)\tValue Error (B)\tKey Error (C)\tValue Error (C)")
    print("-" * 100)
    
    for pos, word in enumerate(token_words):
        print(f"{pos}\t{word}\t{token_key_errors_b[pos]:.4f}\t{token_value_errors_b[pos]:.4f}\t{token_key_errors_c[pos]:.4f}\t{token_value_errors_c[pos]:.4f}")

def plot_cross_attention(results: dict, tokenizer, save_path: str = "examples/pipeline/images/cross_attention.png"):
    """可视化交叉注意力分数
    
    Args:
        results: generate_reuse_output的返回结果
        tokenizer: 用于解码token的tokenizer
        save_path: 保存图片的路径
    """
    # 获取token对应的单词
    token_words = [tokenizer.decode([token]) for token in results['token_ids']]
    
    # 获取unreused indices
    unreused_indices_b = results['reuse_b']['unreused_indices']
    unreused_indices_c = results['reuse_c']['unreused_indices']
    token_words_b = [token_words[i] for i in unreused_indices_b]
    token_words_c = [token_words[i] for i in unreused_indices_c]
    
    # 处理full attention
    full_attn = results['attention_values']['full_compute']
    if torch.is_tensor(full_attn):
        full_attn = full_attn.cpu().numpy()
    num_head, num_token, _ = full_attn.shape
    
    # 处理B attention
    b_attn = results['attention_values']['reuse_b']
    if torch.is_tensor(b_attn):
        b_attn = b_attn.cpu().numpy()
    
    # 处理C attention
    c_attn = results['attention_values']['reuse_c']
    if torch.is_tensor(c_attn):
        c_attn = c_attn.cpu().numpy()
    
    # 实现softmax
    def softmax(x, axis=-1):
        # 减去最大值以提高数值稳定性
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)
    
    # 应用softmax
    full_attn = softmax(full_attn, axis=-1)
    b_attn = softmax(b_attn, axis=-1)
    c_attn = softmax(c_attn, axis=-1)
    
    # 为每个head生成热力图
    for head_idx in range(num_head):
        # 创建2x2的子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 选择当前head的注意力值
        full_attn_b = full_attn[head_idx][unreused_indices_b, :]  # [num_unreused, num_token]
        full_attn_c = full_attn[head_idx][unreused_indices_c, :]  # [num_unreused, num_token]
        b_attn_current = b_attn[head_idx][:, :]  # [num_token, num_token]
        c_attn_current = c_attn[head_idx][:, :]  # [num_token, num_token]
        
        # 绘制full compute的注意力热力图 (B)
        sns.heatmap(full_attn_b, ax=ax1, cmap='viridis')
        ax1.set_title(f'Full Compute Cross Attention (Head {head_idx}) - Unreused Tokens B')
        ax1.set_xticks(range(len(token_words)))
        ax1.set_xticklabels(token_words, rotation=90, ha='right')
        ax1.set_yticks(range(len(token_words_b)))
        ax1.set_yticklabels(token_words_b, rotation=0, ha='right')
        ax1.set_xlabel('All Tokens')
        ax1.set_ylabel('Unreused Tokens (B)')
        
        # 绘制full compute的注意力热力图 (C)
        sns.heatmap(full_attn_c, ax=ax2, cmap='viridis')
        ax2.set_title(f'Full Compute Cross Attention (Head {head_idx}) - Unreused Tokens C')
        ax2.set_xticks(range(len(token_words)))
        ax2.set_xticklabels(token_words, rotation=90, ha='right')
        ax2.set_yticks(range(len(token_words_c)))
        ax2.set_yticklabels(token_words_c, rotation=0, ha='right')
        ax2.set_xlabel('All Tokens')
        ax2.set_ylabel('Unreused Tokens (C)')
        
        # 绘制B的注意力热力图
        sns.heatmap(b_attn_current, ax=ax3, cmap='viridis')
        ax3.set_title(f'Reuse B Cross Attention (Head {head_idx}) - Unreused Tokens')
        ax3.set_xticks(range(len(token_words)))
        ax3.set_xticklabels(token_words, rotation=90, ha='right')
        ax3.set_yticks(range(len(token_words_b)))
        ax3.set_yticklabels(token_words_b, rotation=0, ha='right')
        ax3.set_xlabel('All Tokens')
        ax3.set_ylabel('Unreused Tokens (B)')
        
        # 绘制C的注意力热力图
        sns.heatmap(c_attn_current, ax=ax4, cmap='viridis')
        ax4.set_title(f'Reuse C Cross Attention (Head {head_idx}) - Unreused Tokens')
        ax4.set_xticks(range(len(token_words)))
        ax4.set_xticklabels(token_words, rotation=90, ha='right')
        ax4.set_yticks(range(len(token_words_c)))
        ax4.set_yticklabels(token_words_c, rotation=0, ha='right')
        ax4.set_xlabel('All Tokens')
        ax4.set_ylabel('Unreused Tokens (C)')
        
        # 调整布局
        plt.tight_layout()
        
        # 为每个head保存单独的图片
        head_save_path = save_path.replace('.png', f'_head_{head_idx}.png')
        plt.savefig(head_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Head {head_idx}的交叉注意力热力图已保存至: {head_save_path}")
    
    print(f"所有head的交叉注意力热力图已保存完成")

def save_results_to_cache(results: dict, cache_path: str = "examples/pipeline/cache/results_cache.json"):
    """保存results结果到缓存文件
    
    Args:
        results: generate_reuse_output的返回结果
        cache_path: 缓存文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # 将numpy数组转换为列表
    cache_data = {
        'full_compute': results['full_compute'],
        'reuse_b': results['reuse_b'],
        'reuse_c': results['reuse_c'],
        'source_outputs': results['source_outputs'],
        'token_ids': results['token_ids'],
        'attention_values': {
            'full_compute': results['attention_values']['full_compute'].tolist(),
            'reuse_b': results['attention_values']['reuse_b'].tolist(),
            'reuse_c': results['attention_values']['reuse_c'].tolist()
        },
        'kv_errors': {
            'reuse_b': {
                'key_errors': results['kv_errors']['reuse_b']['key_errors'],
                'value_errors': results['kv_errors']['reuse_b']['value_errors'],
                'token_key_errors': results['kv_errors']['reuse_b']['token_key_errors'],
                'token_value_errors': results['kv_errors']['reuse_b']['token_value_errors']
            },
            'reuse_c': {
                'key_errors': results['kv_errors']['reuse_c']['key_errors'],
                'value_errors': results['kv_errors']['reuse_c']['value_errors'],
                'token_key_errors': results['kv_errors']['reuse_c']['token_key_errors'],
                'token_value_errors': results['kv_errors']['reuse_c']['token_value_errors']
            }
        }
    }
    
    # 保存到JSON文件
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=4)
    
    print(f"Results已保存至: {cache_path}")

def load_results_from_cache(cache_path: str = "examples/pipeline/cache/results_cache.json") -> dict:
    """从缓存文件加载results结果
    
    Args:
        cache_path: 缓存文件路径
        
    Returns:
        dict: 加载的results结果
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
    
    # 从JSON文件加载
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    
    # 将列表转换回numpy数组
    results = {
        'full_compute': cache_data['full_compute'],
        'reuse_b': cache_data['reuse_b'],
        'reuse_c': cache_data['reuse_c'],
        'source_outputs': cache_data['source_outputs'],
        'token_ids': cache_data['token_ids'],
        'attention_values': {
            'full_compute': np.array(cache_data['attention_values']['full_compute']),
            'reuse_b': np.array(cache_data['attention_values']['reuse_b']),
            'reuse_c': np.array(cache_data['attention_values']['reuse_c'])
        },
        'kv_errors': {
            'reuse_b': {
                'key_errors': np.array(cache_data['kv_errors']['reuse_b']['key_errors']),
                'value_errors': np.array(cache_data['kv_errors']['reuse_b']['value_errors']),
                'token_key_errors': np.array(cache_data['kv_errors']['reuse_b']['token_key_errors']),
                'token_value_errors': np.array(cache_data['kv_errors']['reuse_b']['token_value_errors'])
            },
            'reuse_c': {
                'key_errors': np.array(cache_data['kv_errors']['reuse_c']['key_errors']),
                'value_errors': np.array(cache_data['kv_errors']['reuse_c']['value_errors']),
                'token_key_errors': np.array(cache_data['kv_errors']['reuse_c']['token_key_errors']),
                'token_value_errors': np.array(cache_data['kv_errors']['reuse_c']['token_value_errors'])
            }
        }
    }
    
    print(f"Results已从 {cache_path} 加载")
    return results

# def plot_attention_matrix(results: dict, 
#                         tokenizer,
#                         save_path: str = "examples/pipeline/images/attention_matrix.png"):
#     """可视化预填充阶段token之间的注意力分数
    
#     Args:
#         results: generate_reuse_output的返回结果
#         tokenizer: 用于解码token的tokenizer
#         save_path: 保存图片的路径
#     """
#     # 获取token对应的单词
#     token_words = [tokenizer.decode([token]) for token in results['token_ids']]
    
#     # 获取未复用的位置
#     unreused_indices_b = results['reuse_b']['unreused_indices']
#     unreused_indices_c = results['reuse_c']['unreused_indices']
    
#     # 创建2x2的子图
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
#     # 定义切片范围
#     s, e = 24, -6
    
#     # 获取所有注意力矩阵
#     full_attn = results['attention_values']['full_compute'][:, :]
#     b_attn = results['attention_values']['reuse_b'][:, :]
#     c_attn = results['attention_values']['reuse_c'][:, :]
    
#     # 确保张量在CPU上
#     if torch.is_tensor(full_attn):
#         full_attn = full_attn.cpu().numpy()
#     if torch.is_tensor(b_attn):
#         b_attn = b_attn.cpu().numpy()
#     if torch.is_tensor(c_attn):
#         c_attn = c_attn.cpu().numpy()
    
#     # 准备标签
#     visible_tokens = token_words[s:e]
#     visible_tokens_b = [token_words[i] for i in unreused_indices_b]
#     visible_tokens_c = [token_words[i] for i in unreused_indices_c]
    
#     # 计算full compute attention的最大最小值作为归一化范围
#     full_attn_b = full_attn[[i for i in unreused_indices_b],:]
#     full_attn_c = full_attn[[i for i in unreused_indices_c],:]
    
#     # 使用full compute attention的最大最小值
#     global_min = min(full_attn.min(), full_attn.min())
#     global_max = max(full_attn.max(), full_attn.max())
    
#     # 左上：full attn在b indices的注意可视化
#     normalized_attn = (full_attn_b - global_min) / (global_max - global_min)
#     sns.heatmap(normalized_attn,
#                cmap='viridis',
#                vmin=0,
#                vmax=1,
#             #    xticklabels=visible_tokens,
#                yticklabels=visible_tokens_b,
#                cbar_kws={'label': 'Normalized Attention Weight'},
#                ax=ax1)
#     ax1.set_title(f'Full Compute Attention (B Unreused Tokens)\nRange: [{global_min:.4f}, {global_max:.4f}]')
#     ax1.set_xlabel('Prefill Token')
#     ax1.set_ylabel('Prefill Token (Unreused)')
#     plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
#     plt.setp(ax1.get_yticklabels(), rotation=0, ha='right')
    
#     # 右上：full attn的注意力可视化（与C的未复用token对齐）
#     normalized_attn = (full_attn_c - global_min) / (global_max - global_min)
#     sns.heatmap(normalized_attn,
#                cmap='viridis',
#                vmin=0,
#                vmax=1,
#             #    xticklabels=visible_tokens,
#                yticklabels=visible_tokens_c,
#                cbar_kws={'label': 'Normalized Attention Weight'},
#                ax=ax2)
#     ax2.set_title(f'Full Compute Attention (C Unreused Tokens)\nRange: [{global_min:.4f}, {global_max:.4f}]')
#     ax2.set_xlabel('Prefill Token')
#     ax2.set_ylabel('Prefill Token (Unreused)')
#     plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
#     plt.setp(ax2.get_yticklabels(), rotation=0, ha='right')
    
#     # 左下：B的注意力矩阵
#     normalized_attn = (b_attn - global_min) / (global_max - global_min)
#     sns.heatmap(normalized_attn,
#                cmap='viridis',
#                vmin=0,
#                vmax=1,
#             #    xticklabels=visible_tokens_b,
#                yticklabels=visible_tokens_b,
#                cbar_kws={'label': 'Normalized Attention Weight'},
#                ax=ax3)
#     ax3.set_title(f'Reuse B Attention\nRange: [{global_min:.4f}, {global_max:.4f}]')
#     ax3.set_xlabel('Prefill Token')
#     ax3.set_ylabel('Prefill Token (Unreused)')
#     plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
#     plt.setp(ax3.get_yticklabels(), rotation=0, ha='right')
    
#     # 右下：C的注意力矩阵
#     normalized_attn = (c_attn - global_min) / (global_max - global_min)
#     sns.heatmap(normalized_attn,
#                cmap='viridis',
#                vmin=0,
#                vmax=1,
#             #    xticklabels=visible_tokens_c,
#                yticklabels=visible_tokens_c,
#                cbar_kws={'label': 'Normalized Attention Weight'},
#                ax=ax4)
#     ax4.set_title(f'Reuse C Attention\nRange: [{global_min:.4f}, {global_max:.4f}]')
#     ax4.set_xlabel('Prefill Token')
#     ax4.set_ylabel('Prefill Token (Unreused)')
#     plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
#     plt.setp(ax4.get_yticklabels(), rotation=0, ha='right')
    
#     # 调整布局
#     plt.tight_layout()
    
#     # 保存图片
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
    
#     print(f"\nAttention error heatmap已保存至: {save_path}")

# 使用示例
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    
    # text_a = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27. In the sequence of numbers, what is the median number?"
    # text_b = "1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26. In the sequence of numbers, what is the median number?"
    # text_c = "In the sequence of numbers, what is the median number?"
    # text_a = "apple, banana, orange, pear, pineapple, mango, strawberry, kiwi, grape, watermelon. How many fruits are there in the sequence?"
    # text_b = "apple, watermelon.How many fruits are there in the sequence?"
    # text_c = "apple, banana, orange, grape, watermelon. How many fruits are there in the sequence?"

    # text_a = "Please help me analyze the personality traits of the character Jon Snow in A Song of Ice and Fire."
    # text_b = "Lin Daiyu in Dream of the Red Chamber. Please help me analyze the personality traits of the character."
    # text_c = "Daenerys Targaryen in A Song of Ice and Fire. Please help me analyze the personality traits of the character."
    
    # text_a = "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day? "
    # text_b = "He buys 5 large pizzas and 6 small pizzas. A large pizza has 9 slices and a small pizza has 6 slices. If he eats it all, how many pieces does he eat that day? "
    # text_c = "He buys 3 large pizzas and 4 small pizzas. A large pizza has 9 slices and a small pizza has 6 slices."
    
    text_a = "Buses arrive at a stop every 8 minutes starting at 8:00 AM, forming a time sequence: 8:00, 8:08, 10:00 AM. How many buses arrive between 8:30 AM and 9:30 AM, and what is the median arrival time in this interval?"
    text_b = "Buses arrive at a stop every 6 minutes starting at 8:00 AM, forming a time sequence: 8:00, 8:06, 10:00 AM. How many buses arrive between 8:40 AM and 9:20 AM, and what is the median arrival time in this interval?"
    text_c = "Buses arrive at a stop every 4 minutes starting at 8:00 AM, forming a time sequence: 8:00, 8:04, 10:00 AM. How many buses arrive between 8:30 AM and 9:30 AM, and what is the median arrival time in this interval?"
    
    
    template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
    model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

    # 检查缓存文件是否存在
    cache_path = "examples/pipeline/cache/results_cache.json"
    if os.path.exists(cache_path):
        # 如果缓存存在,直接加载
        results = load_results_from_cache(cache_path)
    else:
        # 如果缓存不存在,重新计算并保存
        results = generate_reuse_output(text_a, text_b, text_c, template, model_name,window_size=5)
        save_results_to_cache(results, cache_path)
    
    print_reuse_results(results)
    
    # 可视化交叉注意力
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    plot_cross_attention(results, tokenizer)
    plot_token_kv_errors(results, tokenizer)
    # plot_attention_matrix(results, tokenizer)
    # plot_attention_error(results, tokenizer, num_decode_tokens=4)
    # plot_kv_error_by_window_size("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250405_windows_output_qwen2.5-32b.json", "examples/pipeline/images/kv_error_by_window_size.png")
    