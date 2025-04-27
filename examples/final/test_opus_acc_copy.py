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
# font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
# font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
# for file in font_files:
#     font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'WenQuanYi Micro Hei'  # 设置字体为黑体
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

def generate_reuse_output(target: str, candidates: List[str], template: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct", max_model_len=8192, max_generate_len=512, window_size=7):
    """生成重用输出
    
    Args:
        target: 目标文本
        candidates: 候选文本列表
        template: 模板
        model_name: 模型名称
        max_model_len: 最大模型长度
        max_generate_len: 最大生成长度
        window_size: 窗口大小
    
    Returns:
        results: 包含生成结果和注意力分数的字典
    """
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device, max_model_len)
    tokenizer = pipeline.model.get_tokenizer()
    
    # Prepare prompts
    prompt_a = template.format(text=target)
    
    candidate_prompts = [template.format(text=candidate) for candidate in candidates]
    
    # Check token length
    tokens_a = tokenizer.encode(prompt_a)
    candidate_token_ids = [tokenizer.encode(prompt) for prompt in candidate_prompts]
    
    # if len(tokens_a) > max_model_len or any(len(token_ids) > max_model_len for token_ids in candidate_token_ids):
    #     raise ValueError("Text length exceeds maximum model length limit")
    
    results = {
        "input_token_ids": tokens_a,
        'full_compute_outputs': None,
        'partial_compute_outputs': None,
        "partial_compute_updated_indice": None,
        'source_outputs': {
           
        },
        'forward_attention': {
            'full_compute': None,
            'partial_compute': None,
        },
        'cross_attention': {
            'full_compute': None,
            'partial_compute': None,
        },
        'attention': {
            'full_compute': None,
            'partial_compute': None,
        },
        'kv_errors': {
            'partial_compute': {
                'key_errors': [],  # 按层统计的Key误差
                'value_errors': [],  # 按层统计的Value误差
                'token_key_errors': [],  # 按token位置统计的Key误差
                'token_value_errors': []  # 按token位置统计的Value误差
            }
        }
    }
    
    # Get full computation KV cache as baseline
    full_kv_cache, full_outputs, _, full_forward_attn, full_cross_attn, full_attn = KVShareNewPipeline.get_kvcache_by_full_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_a]
    )
    results['forward_attention']['full_compute'] = full_forward_attn[0]
    results['cross_attention']['full_compute'] = full_cross_attn[0]
    results['attention']['full_compute'] = full_attn[0]
    results['full_compute_outputs'] = full_outputs[0].outputs[0].text
    
    # candidates_token_ids = [tokenizer.encode(candidate) for candidate in candidates]
    candidates_kv_cache = []
    max_request_id = 0
    for idx,candidate_prompt in enumerate(candidate_prompts):
        # Get KV cache for B and C
        candidate_kv_cache, candidate_outputs, _, candidate_forward_attn, candidate_cross_attn, candidate_attn = KVShareNewPipeline.get_kvcache_by_full_compute(
            pipeline.model,
            SamplingParams(temperature=0, max_tokens=max_generate_len),
            [candidate_prompt]
        )
        max_request_id = int(candidate_outputs[-1].request_id)
        candidates_kv_cache.append(candidate_kv_cache)

    # target_kvcache,batch_reused_map_indices,batch_unreused_map_indices,batch_sample_selected_token_indices,batch_target_slice_list = KVEditor.kvedit_v2(
    #     tokens_a,
    #     candidate_token_ids,
    #     candidates_kv_cache,
    #     window_size=window_size
    # )
    target_kvcache,batch_reused_map_indices,batch_unreused_map_indices,batch_sample_selected_token_indices,batch_target_slice_list = KVEditor.batch_kvedit(
        [tokens_a],
        candidate_token_ids,
        candidates_kv_cache[0],
        window_size=window_size
    )
    
    
    
    for i in range(len(batch_reused_map_indices[0])):
        print("resue token: ",tokenizer.decode(tokens_a[batch_reused_map_indices[0][i]]))
    
    partial_outputs, partial_kv_cache, batch_forward_attn, batch_cross_attn ,batch_attn,partial_compute_updated_indice = KVShareNewPipeline.partial_compute(
        pipeline.model,
        SamplingParams(temperature=0, max_tokens=max_generate_len),
        [prompt_a],
        target_kvcache,
        batch_reused_map_indices,
        batch_unreused_map_indices,
        batch_sample_selected_token_indices,
        batch_target_slice_list,
        [max_request_id+1]
    )
    results['forward_attention']['partial_compute'] = batch_forward_attn[0]
    results['cross_attention']['partial_compute'] = batch_cross_attn[0]
    results['attention']['partial_compute'] = batch_attn[0]
    results['partial_compute_outputs'] = partial_outputs[0].outputs[0].text

    for i in range(len(batch_unreused_map_indices[0])):
        print("unreused token: ",tokenizer.decode(tokens_a[batch_unreused_map_indices[0][i]]))
    for i in range(len(partial_compute_updated_indice)):
        print("updated token: ",tokenizer.decode(tokens_a[partial_compute_updated_indice[i]]))

    # 按层统计误差
    num_layers = full_kv_cache.shape[0]
    for layer_idx in range(num_layers):
        # Key误差
        key_error = torch.abs(full_kv_cache[layer_idx, 0] - partial_kv_cache[layer_idx, 0])
        results['kv_errors']['partial_compute']['key_errors'].append(torch.mean(key_error).item())
        
        # Value误差
        value_error = torch.abs(full_kv_cache[layer_idx, 1] - partial_kv_cache[layer_idx, 1])
        results['kv_errors']['partial_compute']['value_errors'].append(torch.mean(value_error).item())
    
    # 按token位置统计误差
    for token_idx in range(len(tokens_a)):
        # Key误差
        key_error = torch.abs(full_kv_cache[:, 0, token_idx] - partial_kv_cache[:, 0, token_idx])
        results['kv_errors']['partial_compute']['token_key_errors'].append(torch.mean(key_error).item())
        
        # Value误差
        value_error = torch.abs(full_kv_cache[:, 1, token_idx] - partial_kv_cache[:, 1, token_idx])
        results['kv_errors']['partial_compute']['token_value_errors'].append(torch.mean(value_error).item())
    
    results['partial_compute_updated_indice'] = partial_compute_updated_indice.detach().cpu().tolist()
    
    return results

def print_reuse_results(results: dict):
    """Print comparison of reuse results
    
    Args:
        results: Return value from generate_reuse_output
    """
    print("\nFull Computation Result:")
    print("-" * 50)
    print(results['full_compute_outputs'])

    
    print("\nReusing Result:")
    print("-" * 50)
    # print(f"Reused tokens: {results['reuse_b']['reused_tokens']}")
    # print(f"Unreused tokens: {results['reuse_b']['unreused_tokens']}")
    print(f"Generated output: {results['partial_compute_outputs']}")
    


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

def plot_forward_attention(results: dict, tokenizer, save_path: str = "examples/pipeline/images/forward_attention.png", topk: int = 4):
    """可视化前向注意力分数
    
    Args:
        results: generate_reuse_output的返回结果
        tokenizer: 用于解码token的tokenizer
        save_path: 保存图片的路径
        topk: 显示的解码token数量
    """
    # 获取token对应的单词
    token_words = [tokenizer.decode([token]) for token in results['token_ids']]
    
    # 获取B和C的输出token
    a_output_tokens = tokenizer.encode(results['full_compute'])
    a_output_words = [tokenizer.decode([token]) for token in a_output_tokens]
    b_output_tokens = tokenizer.encode(results['reuse_b']['output'])
    c_output_tokens = tokenizer.encode(results['reuse_c']['output'])
    b_output_words = [tokenizer.decode([token]) for token in b_output_tokens]
    c_output_words = [tokenizer.decode([token]) for token in c_output_tokens]
    
    # 获取注意力值
    full_attn = results['forward_attention']['full_compute']
    b_attn = results['forward_attention']['reuse_b']
    c_attn = results['forward_attention']['reuse_c']
    
    # 确保张量在CPU上
    if torch.is_tensor(full_attn):
        full_attn = full_attn.cpu().numpy()
    if torch.is_tensor(b_attn):
        b_attn = b_attn.cpu().numpy()
    if torch.is_tensor(c_attn):
        c_attn = c_attn.cpu().numpy()
        
    s,e = 25,-5
    full_attn = full_attn[:, :, s:e]
    b_attn = b_attn[:, :, s:e]
    c_attn = c_attn[:, :, s:e]
    token_words = token_words[s:e]
    # 获取注意力头数和token数量
    num_heads, num_tokens, _ = full_attn.shape
    
    # 归一化函数
    def normalize_attention(attn):
        # 使用最大最小归一化
        attn = torch.tensor(attn)
        min_val = attn.min()
        max_val = attn.max()
        if max_val != min_val:
            attn = (attn - min_val) / (max_val - min_val)
        return attn.numpy()
    
    full_attn = normalize_attention(full_attn)
    b_attn = normalize_attention(b_attn)
    c_attn = normalize_attention(c_attn)
    
    topk_a = 30
    topk_b = 38
    topk_c = 30
    # 为每个head生成热力图
    for head_idx in range(num_heads):
        # 创建3个子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
        
        # 选择当前head的注意力值
        full_head_attn = full_attn[head_idx, :topk_a, :]  # [num_tokens, num_tokens]
        b_head_attn = b_attn[head_idx, :topk_b, :]  # [num_tokens, num_tokens]
        c_head_attn = c_attn[head_idx, :topk_c, :]  # [num_tokens, num_tokens]

        a_output_words = a_output_words[:topk_a]
        b_output_words = b_output_words[:topk_b]
        c_output_words = c_output_words[:topk_c]
        # 绘制full compute的注意力热力图
        sns.heatmap(full_head_attn, 
                   ax=ax1,
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=a_output_words,
                   cbar_kws={'label': 'Attention Weight'})
        ax1.set_title(f'Full Compute Forward Attention (Head {head_idx})')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0, ha='right')
        
        # 绘制B的注意力热力图
        sns.heatmap(b_head_attn, 
                   ax=ax2,
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=b_output_words,
                   cbar_kws={'label': 'Attention Weight'})
        ax2.set_title(f'B Forward Attention (Head {head_idx})')
        # 设置unreused token标签为红色
        for i, label in enumerate(ax2.get_xticklabels()):
            if i in results['reuse_b']['unreused_indices']:
                label.set_color('red')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0, ha='right')
        
        # 绘制C的注意力热力图
        sns.heatmap(c_head_attn, 
                   ax=ax3,
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=c_output_words,
                   cbar_kws={'label': 'Attention Weight'})
        ax3.set_title(f'C Forward Attention (Head {head_idx})')
        # 设置unreused token标签为红色
        for i, label in enumerate(ax3.get_xticklabels()):
            if i in results['reuse_c']['unreused_indices']:
                label.set_color('red')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax3.get_yticklabels(), rotation=0, ha='right')
        
        # 调整布局
        plt.tight_layout()
        
        # 为每个head保存单独的图片
        head_save_path = save_path.replace('.png', f'_head_{head_idx}.png')
        plt.savefig(head_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Head {head_idx}的前向注意力热力图已保存至: {head_save_path}")
    
    # 创建所有head求和后的热力图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
    
    # 计算所有head的注意力值之和
    full_attn_sum = np.sum(full_attn, axis=0)  # [num_tokens, num_tokens]
    b_attn_sum = np.sum(b_attn, axis=0)  # [num_tokens, num_tokens]
    c_attn_sum = np.sum(c_attn, axis=0)  # [num_tokens, num_tokens]
    
    # 绘制full compute的注意力热力图
    sns.heatmap(full_attn_sum[:topk_a, :], 
               ax=ax1,
               cmap='viridis',
               xticklabels=token_words,
               yticklabels=a_output_words[:topk],
               cbar_kws={'label': 'Attention Weight'})
    ax1.set_title('Full Compute Forward Attention (All Heads Sum)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0, ha='right')
    
    # 绘制B的注意力热力图
    sns.heatmap(b_attn_sum[:topk_b, :], 
               ax=ax2,
               cmap='viridis',
               xticklabels=token_words,
               yticklabels=b_output_words,
               cbar_kws={'label': 'Attention Weight'})
    ax2.set_title('B Forward Attention (All Heads Sum)')
    # 设置unreused token标签为红色
    for i, label in enumerate(ax2.get_xticklabels()):
        if i in results['reuse_b']['unreused_indices']:
            label.set_color('red')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0, ha='right')
    
    # 绘制C的注意力热力图
    sns.heatmap(c_attn_sum[:topk_c, :], 
               ax=ax3,
               cmap='viridis',
               xticklabels=token_words,
               yticklabels=c_output_words,
               cbar_kws={'label': 'Attention Weight'})
    ax3.set_title('C Forward Attention (All Heads Sum)')
    # 设置unreused token标签为红色
    for i, label in enumerate(ax3.get_xticklabels()):
        if i in results['reuse_c']['unreused_indices']:
            label.set_color('red')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax3.get_yticklabels(), rotation=0, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存所有head求和后的图片
    sum_save_path = save_path.replace('.png', '_all_heads_sum.png')
    plt.savefig(sum_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"所有head求和后的前向注意力热力图已保存至: {sum_save_path}")
    
    print(f"所有head的前向注意力热力图已保存完成")

def plot_cross_attention(results: dict, tokenizer, save_path: str = "examples/pipeline/images/cross_attention.png", topk: int = 4):
    """可视化交叉注意力分数
    
    Args:
        results: generate_reuse_output的返回结果
        tokenizer: 用于解码token的tokenizer
        save_path: 保存图片的路径
        topk: 显示的解码token数量
    """
    # 获取token对应的单词
    token_words = [tokenizer.decode([token]) for token in results['token_ids']]
    
    # 获取B和C的输出token
    a_output_tokens = tokenizer.encode(results['full_compute'])
    a_output_words = [tokenizer.decode([token]) for token in a_output_tokens]
    b_output_tokens = tokenizer.encode(results['reuse_b']['output'])
    c_output_tokens = tokenizer.encode(results['reuse_c']['output'])
    b_output_words = [tokenizer.decode([token]) for token in b_output_tokens]
    c_output_words = [tokenizer.decode([token]) for token in c_output_tokens]
    
    # 获取注意力值
    full_attn = results['cross_attention']['full_compute']
    b_attn = results['cross_attention']['reuse_b']
    c_attn = results['cross_attention']['reuse_c']
    
    # 确保张量在CPU上
    if torch.is_tensor(full_attn):
        full_attn = full_attn.cpu().numpy()
    if torch.is_tensor(b_attn):
        b_attn = b_attn.cpu().numpy()
    if torch.is_tensor(c_attn):
        c_attn = c_attn.cpu().numpy()
        
    s,e = 25,-5
    full_attn = full_attn[:, :, s:e]
    b_attn = b_attn[:, :, s:e]
    c_attn = c_attn[:, :, s:e]
    token_words = token_words[s:e]
    # 获取注意力头数和token数量
    num_heads, num_tokens, _ = full_attn.shape
    
    # 归一化函数
    def normalize_attention(attn):
        # 使用最大最小归一化
        attn = torch.tensor(attn)
        min_val = attn.min()
        max_val = attn.max()
        if max_val != min_val:
            attn = (attn - min_val) / (max_val - min_val)
        return attn.numpy()
    
    full_attn = normalize_attention(full_attn)
    b_attn = normalize_attention(b_attn)
    c_attn = normalize_attention(c_attn)
    
    topk_a = 30
    topk_b = 38
    topk_c = 30
    # 为每个head生成热力图
    for head_idx in range(num_heads):
        # 创建3个子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
        
        # 选择当前head的注意力值
        full_head_attn = full_attn[head_idx, :topk_a, :]  # [num_tokens, num_tokens]
        b_head_attn = b_attn[head_idx, :topk_b, :]  # [num_tokens, num_tokens]
        c_head_attn = c_attn[head_idx, :topk_c, :]  # [num_tokens, num_tokens]

        a_output_words = a_output_words[:topk_a]
        b_output_words = b_output_words[:topk_b]
        c_output_words = c_output_words[:topk_c]
        # 绘制full compute的注意力热力图
        sns.heatmap(full_head_attn, 
                   ax=ax1,
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=a_output_words,
                   cbar_kws={'label': 'Attention Weight'})
        ax1.set_title(f'Full Compute Cross Attention (Head {head_idx})')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax1.get_yticklabels(), rotation=0, ha='right')
        
        # 绘制B的注意力热力图
        sns.heatmap(b_head_attn, 
                   ax=ax2,
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=b_output_words,
                   cbar_kws={'label': 'Attention Weight'})
        ax2.set_title(f'B Cross Attention (Head {head_idx})')
        # 设置unreused token标签为红色
        for i, label in enumerate(ax2.get_xticklabels()):
            if i in results['reuse_b']['unreused_indices']:
                label.set_color('red')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0, ha='right')
        
        # 绘制C的注意力热力图
        sns.heatmap(c_head_attn, 
                   ax=ax3,
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=c_output_words,
                   cbar_kws={'label': 'Attention Weight'})
        ax3.set_title(f'C Cross Attention (Head {head_idx})')
        # 设置unreused token标签为红色
        for i, label in enumerate(ax3.get_xticklabels()):
            if i in results['reuse_c']['unreused_indices']:
                label.set_color('red')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax3.get_yticklabels(), rotation=0, ha='right')
        
        # 调整布局
        plt.tight_layout()
        
        # 为每个head保存单独的图片
        head_save_path = save_path.replace('.png', f'_head_{head_idx}.png')
        plt.savefig(head_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Head {head_idx}的交叉注意力热力图已保存至: {head_save_path}")
    
    # 创建所有head求和后的热力图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12))
    
    # 计算所有head的注意力值之和
    full_attn_sum = np.sum(full_attn, axis=0)  # [num_tokens, num_tokens]
    b_attn_sum = np.sum(b_attn, axis=0)  # [num_tokens, num_tokens]
    c_attn_sum = np.sum(c_attn, axis=0)  # [num_tokens, num_tokens]
    
    # 绘制full compute的注意力热力图
    sns.heatmap(full_attn_sum[:topk_a, :], 
               ax=ax1,
               cmap='viridis',
               xticklabels=token_words,
               yticklabels=a_output_words[:topk],
               cbar_kws={'label': 'Attention Weight'})
    ax1.set_title('Full Compute Cross Attention (All Heads Sum)')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0, ha='right')
    
    # 绘制B的注意力热力图
    sns.heatmap(b_attn_sum[:topk_b, :], 
               ax=ax2,
               cmap='viridis',
               xticklabels=token_words,
               yticklabels=b_output_words,
               cbar_kws={'label': 'Attention Weight'})
    ax2.set_title('B Cross Attention (All Heads Sum)')
    # 设置unreused token标签为红色
    for i, label in enumerate(ax2.get_xticklabels()):
        if i in results['reuse_b']['unreused_indices']:
            label.set_color('red')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax2.get_yticklabels(), rotation=0, ha='right')
    
    # 绘制C的注意力热力图
    sns.heatmap(c_attn_sum[:topk_c, :], 
               ax=ax3,
               cmap='viridis',
               xticklabels=token_words,
               yticklabels=c_output_words,
               cbar_kws={'label': 'Attention Weight'})
    ax3.set_title('C Cross Attention (All Heads Sum)')
    # 设置unreused token标签为红色
    for i, label in enumerate(ax3.get_xticklabels()):
        if i in results['reuse_c']['unreused_indices']:
            label.set_color('red')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax3.get_yticklabels(), rotation=0, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存所有head求和后的图片
    sum_save_path = save_path.replace('.png', '_all_heads_sum.png')
    plt.savefig(sum_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"所有head求和后的交叉注意力热力图已保存至: {sum_save_path}")
    
    print(f"所有head的交叉注意力热力图已保存完成")

def save_results_to_cache(results: dict, cache_path: str = "examples/pipeline/cache/results_cache.json"):
    """保存results结果到缓存文件
    
    Args:
        results: generate_reuse_output的返回结果
        cache_path: 缓存文件路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # 将numpy数组转换为列表，处理None值
    def convert_to_list(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.cpu().numpy().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    cache_data = {
        'input_token_ids': results['input_token_ids'],
        'full_compute_outputs': results['full_compute_outputs'],
        'partial_compute_outputs': results['partial_compute_outputs'],
        'source_outputs': results['source_outputs'],
        'partial_compute_updated_indice': results['partial_compute_updated_indice'],
        'forward_attention': {
            'full_compute': convert_to_list(results['forward_attention']['full_compute']),
            'partial_compute': convert_to_list(results['forward_attention']['partial_compute'])
        },
        'cross_attention': {
            'full_compute': convert_to_list(results['cross_attention']['full_compute']),
            'partial_compute': convert_to_list(results['cross_attention']['partial_compute'])
        },
        'attention': {
            'full_compute': convert_to_list(results['attention']['full_compute']),
            'partial_compute': convert_to_list(results['attention']['partial_compute'])
        },
        'kv_errors': {
            'partial_compute': {
                'key_errors': convert_to_list(results['kv_errors']['partial_compute']['key_errors']),
                'value_errors': convert_to_list(results['kv_errors']['partial_compute']['value_errors']),
                'token_key_errors': convert_to_list(results['kv_errors']['partial_compute']['token_key_errors']),
                'token_value_errors': convert_to_list(results['kv_errors']['partial_compute']['token_value_errors'])
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
        'input_token_ids': cache_data['input_token_ids'],
        'full_compute_outputs': cache_data['full_compute_outputs'],
        'partial_compute_outputs': cache_data['partial_compute_outputs'],
        'source_outputs': cache_data['source_outputs'],
        'partial_compute_updated_indice': cache_data['partial_compute_updated_indice'],
        'forward_attention': {
            'full_compute': np.array(cache_data['forward_attention']['full_compute']),
            'partial_compute': np.array(cache_data['forward_attention']['partial_compute'])
        },
        'cross_attention': {
            'full_compute': np.array(cache_data['cross_attention']['full_compute']),
            'partial_compute': np.array(cache_data['cross_attention']['partial_compute'])
        },
        'attention': {
            'full_compute': np.array(cache_data['attention']['full_compute']),
            'partial_compute': np.array(cache_data['attention']['partial_compute'])
        },
        'kv_errors': {
            'partial_compute': {
                'key_errors': np.array(cache_data['kv_errors']['partial_compute']['key_errors']),
                'value_errors': np.array(cache_data['kv_errors']['partial_compute']['value_errors']),
                'token_key_errors': np.array(cache_data['kv_errors']['partial_compute']['token_key_errors']),
                'token_value_errors': np.array(cache_data['kv_errors']['partial_compute']['token_value_errors'])
            }
        }
    }
    
    print(f"Results已从 {cache_path} 加载")
    return results

def plot_attention_combined(results: dict, tokenizer, save_path: str = "examples/pipeline/images/attention_combined.png", topk: int = 100):
    """可视化交叉注意力分数和相似度分析
    
    Args:
        results: generate_reuse_output的返回结果
        tokenizer: 用于解码token的tokenizer
        save_path: 保存图片的路径
        topk: 显示的解码token数量
    """
    # 获取token对应的单词
    token_words = [tokenizer.decode([token]) for token in results['input_token_ids']]
    
    # 获取输出token
    full_output_tokens = tokenizer.encode(results['full_compute_outputs'])
    full_output_words = [tokenizer.decode([token]) for token in full_output_tokens]
    partial_output_tokens = tokenizer.encode(results['partial_compute_outputs'])
    partial_output_words = [tokenizer.decode([token]) for token in partial_output_tokens]
    
    # 获取注意力值
    full_cross_attn = results['cross_attention']['full_compute']
    partial_cross_attn = results['cross_attention']['partial_compute']
    full_forward_attn = results['forward_attention']['full_compute']
    partial_forward_attn = results['forward_attention']['partial_compute']
    
    # 确保数组不为空
    if len(full_cross_attn) == 0 or len(partial_cross_attn) == 0:
        print("警告：注意力数组为空，无法生成热力图")
        return
        
    # 对注意力值进行softmax归一化
    def normalize_attention(attn):
        # 使用softmax归一化
        attn = torch.softmax(torch.tensor(attn), dim=-1)
        return attn.numpy()
    
    full_cross_attn = normalize_attention(full_cross_attn)
    partial_cross_attn = normalize_attention(partial_cross_attn)
    full_forward_attn = normalize_attention(full_forward_attn)
    partial_forward_attn = normalize_attention(partial_forward_attn)
        
    # 获取注意力头数量
    num_heads = full_cross_attn.shape[0]
    s,e = 25,-5
    partial_compute_updated_indice = results['partial_compute_updated_indice']
    
    # # 分析注意力相似度
    similarity_results = analyze_attention_similarity(
        torch.tensor(full_cross_attn[:, partial_compute_updated_indice, s:e]), 
        torch.tensor( partial_cross_attn[:, :, s:e])
    )
    s = -5
    # e = -1
    # 为每个head创建热力图
    for head_idx in range(num_heads):
        # 获取当前head的注意力值
        # full_head_cross = full_cross_attn[head_idx, partial_compute_updated_indice, s:e]
        # partial_head_cross = partial_cross_attn[head_idx, :, s:e]
        full_head_forward = full_forward_attn[head_idx, :, :]
        partial_head_forward = partial_forward_attn[head_idx, s:, :]
        
        # # 创建full compute交叉注意力子图
        # fig, ax = plt.subplots(figsize=(8, 4))
        # sns.heatmap(full_head_cross, 
        #            ax=ax,
        #            cmap='viridis',
        #            xticklabels=token_words[s:e],
        #            yticklabels=[token_words[i] for i in range(len(token_words)) if i in partial_compute_updated_indice],
        #            cbar_kws={'label': 'Attention Weight'})
        # ax.set_title('Full Compute Cross Attention')
        # plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
        # plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
        # plt.tight_layout()
        
        # # 保存full compute交叉注意力图片
        # full_cross_save_path = save_path.replace('.png', f'_full_cross_head_{head_idx}.png')
        # plt.savefig(full_cross_save_path, dpi=300, bbox_inches="tight")
        # plt.close()
        
        # # 创建partial compute交叉注意力子图
        # fig, ax = plt.subplots(figsize=(8, 4))
        # sns.heatmap(partial_head_cross, 
        #            ax=ax,   
        #            cmap='viridis',
        #            xticklabels=token_words[s:e],
        #            yticklabels=[token_words[i] for i in range(len(token_words)) if i in partial_compute_updated_indice],
        #            cbar_kws={'label': 'Attention Weight'})
        # ax.set_title('Partial Compute Cross Attention')
        # plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
        # plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
        # plt.tight_layout()
        
        # # 保存partial compute交叉注意力图片
        # partial_cross_save_path = save_path.replace('.png', f'_partial_cross_head_{head_idx}.png')
        # plt.savefig(partial_cross_save_path, dpi=300, bbox_inches="tight")
        # plt.close()
        
        # 创建full compute前向注意力子图
        fig, ax = plt.subplots(figsize=(12, 1))
        sns.heatmap(full_head_forward, 
                   ax=ax,
                   cmap='viridis',
                   cbar=False,
                   xticklabels=False,
                   yticklabels=False)
        plt.tight_layout()
        
        # 保存full compute前向注意力图片
        full_forward_save_path = save_path.replace('.png', f'_full_forward_head_{head_idx}.png')
        plt.savefig(full_forward_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # 创建partial compute前向注意力子图
        fig, ax = plt.subplots(figsize=(12, 1))
        sns.heatmap(partial_head_forward, 
                   ax=ax,
                   cmap='viridis',
                   cbar=False,
                   xticklabels=False,
                   yticklabels=False)
        plt.tight_layout()
        
        # 保存partial compute前向注意力图片
        partial_forward_save_path = save_path.replace('.png', f'_partial_forward_head_{head_idx}.png')
        plt.savefig(partial_forward_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        # print(f"Head {head_idx}的注意力分析图已保存至: {full_cross_save_path}, {partial_cross_save_path}, {full_forward_save_path}, {partial_forward_save_path}")
    
    # 创建所有head求和后的热力图
    # 计算所有head的注意力值之和
    # full_cross_sum = np.sum(full_cross_attn, axis=0)
    # partial_cross_sum = np.sum(partial_cross_attn, axis=0)
    # full_forward_sum = np.sum(full_forward_attn, axis=0)
    # partial_forward_sum = np.sum(partial_forward_attn, axis=0)
    
    # # 创建full compute交叉注意力求和后的热力图
    # fig, ax = plt.subplots(figsize=(8, 4))
    # sns.heatmap(full_cross_sum[:topk, s:e], 
    #            ax=ax,
    #            cmap='viridis',
    #            xticklabels=token_words[s:e],
    #            yticklabels=full_output_words[:topk],
    #            cbar_kws={'label': 'Attention Weight'})
    # ax.set_title('Full Compute Cross Attention (Sum of All Heads)')
    # plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    # plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    # plt.tight_layout()
    
    # # 保存full compute交叉注意力求和后的图片
    # full_cross_sum_save_path = save_path.replace('.png', '_full_cross_all_heads_sum.png')
    # plt.savefig(full_cross_sum_save_path, dpi=300, bbox_inches="tight")
    # plt.close()
    
    # # 创建partial compute交叉注意力求和后的热力图
    # fig, ax = plt.subplots(figsize=(8, 4))
    # sns.heatmap(partial_cross_sum[:topk, s:e], 
    #            ax=ax,
    #            cmap='viridis',
    #            xticklabels=token_words[s:e],
    #            yticklabels=partial_output_words[:topk],
    #            cbar_kws={'label': 'Attention Weight'})
    # ax.set_title('Partial Compute Cross Attention (Sum of All Heads)')
    # plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    # plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    # plt.tight_layout()
    
    # # 保存partial compute交叉注意力求和后的图片
    # partial_cross_sum_save_path = save_path.replace('.png', '_partial_cross_all_heads_sum.png')
    # plt.savefig(partial_cross_sum_save_path, dpi=300, bbox_inches="tight")
    # plt.close()
    
    # # 创建full compute前向注意力求和后的热力图
    # fig, ax = plt.subplots(figsize=(8, 4))
    # sns.heatmap(full_forward_sum[:topk, :topk], 
    #            ax=ax,
    #            cmap='viridis',
    #            xticklabels=token_words[:topk],
    #            yticklabels=token_words[:topk],
    #            cbar_kws={'label': 'Attention Weight'})
    # ax.set_title('Full Compute Forward Attention (Sum of All Heads)')
    # plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    # plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    # plt.tight_layout()
    
    # # 保存full compute前向注意力求和后的图片
    # full_forward_sum_save_path = save_path.replace('.png', '_full_forward_all_heads_sum.png')
    # plt.savefig(full_forward_sum_save_path, dpi=300, bbox_inches="tight")
    # plt.close()
    
    # # 创建partial compute前向注意力求和后的热力图
    # fig, ax = plt.subplots(figsize=(8, 4))
    # sns.heatmap(partial_forward_sum[:topk, :topk], 
    #            ax=ax,
    #            cmap='viridis',
    #            xticklabels=token_words[:topk],
    #            yticklabels=token_words[:topk],
    #            cbar_kws={'label': 'Attention Weight'})
    # ax.set_title('Partial Compute Forward Attention (Sum of All Heads)')
    # plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
    # plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    # plt.tight_layout()
    
    # # 保存partial compute前向注意力求和后的图片
    # partial_forward_sum_save_path = save_path.replace('.png', '_partial_forward_all_heads_sum.png')
    # plt.savefig(partial_forward_sum_save_path, dpi=300, bbox_inches="tight")
    # plt.close()
    
    # print(f"所有head求和后的注意力分析图已保存至: {full_cross_sum_save_path}, {partial_cross_sum_save_path}, {full_forward_sum_save_path}, {partial_forward_sum_save_path}")
    
    # print(similarity_results)
    # for head_idx,pos_info in enumerate(similarity_results["head_similarities"]):
    #     print(f"Head {head_idx} 的平均相似度: {pos_info}")
    
    # print(f"所有head的注意力分析图已保存完成")

def analyze_attention_similarity(full_attn: torch.Tensor, partial_attn: torch.Tensor, threshold: float = 0.7) -> dict:
    """分析两个注意力矩阵的相似度
    
    Args:
        full_attn: 完整计算的注意力矩阵 [num_heads, num_tokens, num_tokens]
        partial_attn: 部分计算的注意力矩阵 [num_heads, num_tokens, num_tokens]
        threshold: 相似度阈值，低于该值认为显著下降
        
    Returns:
        dict: 包含相似度分析结果的字典
    """
    # 确保输入是torch张量
    if not isinstance(full_attn, torch.Tensor):
        full_attn = torch.tensor(full_attn)
    if not isinstance(partial_attn, torch.Tensor):
        partial_attn = torch.tensor(partial_attn)
        
    # 获取注意力头数和token数量
    num_heads, num_tokens, _ = full_attn.shape
    
    # 初始化结果字典
    results = {
        "head_similarities": [],  # 每个头的平均相似度
        "low_similarity_heads": [],  # 相似度低的头索引
        "low_similarity_positions": [],  # 每个头中相似度低的位置
        "head_similarity_matrix": torch.zeros((num_heads, num_tokens, num_tokens))  # 每个位置的相似度矩阵
    }
    
    # 对每个注意力头计算相似度
    for head_idx in range(num_heads):
        # 获取当前头的注意力矩阵
        full_head = full_attn[head_idx].reshape(1, -1)
        partial_head = partial_attn[head_idx].reshape(1, -1)
        
        # 矩阵转换从1维向量
        similarity_matrix = torch.nn.functional.cosine_similarity(full_head, partial_head)
        
        # 保存相似度矩阵
        # results["head_similarity_matrix"][head_idx] = similarity_matrix
        
        # 计算平均相似度
        avg_similarity = similarity_matrix.item()
        results["head_similarities"].append(avg_similarity)
        
        # 找出相似度低的位置
        # low_sim_positions = torch.where(similarity_matrix < threshold)
        # if len(low_sim_positions[0]) > 0:
        #     results["low_similarity_positions"].append({
        #         "head_idx": head_idx,
        #         "positions": list(zip(low_sim_positions[0].tolist(), 
        #                             low_sim_positions[1].tolist()))
        #     })
    
    # 找出相似度低的头
    avg_similarities = torch.tensor(results["head_similarities"])
    low_sim_heads = torch.where(avg_similarities < threshold)[0].tolist()
    results["low_similarity_heads"] = low_sim_heads
    
    return results

def plot_attention_similarity(results: dict, tokenizer, save_path: str = "examples/pipeline/images/attention_similarity.png"):
    """可视化注意力相似度分析结果
    
    Args:
        results: analyze_attention_similarity的返回结果
        tokenizer: 用于解码token的tokenizer
        save_path: 保存图片的路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 获取token对应的单词
    token_words = [tokenizer.decode([token]) for token in results['token_ids']]
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 1. 绘制每个头的平均相似度柱状图
    plt.subplot(2, 1, 1)
    head_similarities = results["head_similarities"]
    plt.bar(range(len(head_similarities)), head_similarities)
    plt.axhline(y=0.7, color='r', linestyle='--', label='Threshold')
    plt.title('Average Similarity per Attention Head')
    plt.xlabel('Attention Head Index')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    
    # 2. 绘制相似度最低的头的注意力相似度热力图
    if results["low_similarity_heads"]:
        head_idx = results["low_similarity_heads"][0]  # 选择第一个低相似度头
        similarity_matrix = results["head_similarity_matrix"][head_idx]
        
        plt.subplot(2, 1, 2)
        sns.heatmap(similarity_matrix, 
                   cmap='viridis',
                   xticklabels=token_words,
                   yticklabels=token_words,
                   cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f'Attention Similarity Matrix (Head {head_idx})')
        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right')
        plt.setp(plt.gca().get_yticklabels(), rotation=0, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"注意力相似度分析图已保存至: {save_path}")
    
    # 打印低相似度位置信息
    print("\n低相似度位置信息:")
    for pos_info in results["low_similarity_positions"]:
        head_idx = pos_info["head_idx"]
        positions = pos_info["positions"]
        print(f"\nHead {head_idx} 的低相似度位置:")
        for i, j in positions[:10]:  # 只显示前10个位置
            print(f"位置 ({i}, {j}): {token_words[i]} -> {token_words[j]}")

# 使用示例
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    
    # text_a = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27. In the sequence of numbers, what is the median number?"
    # text_b = "1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26. In the sequence of numbers, what is the median number?"
    # text_c = "In the sequence of numbers, what is the median number?"
    # text_a = "How many fruits are there in the sequence? apple, banana, orange, pear, pineapple, mango, strawberry, kiwi, grape, watermelon."
    # text_b = "How many fruits are there in the sequence? apple, banana, orange."
    # text_c = "How many fruits are there in the sequence? apple, banana, orange, pear, mango, strawberry, kiwi, grape, watermelon."


    # text_a = "How many fruits are there in the sequence? banana, orange, pear, pineapple, mango, strawberry, grape, apple."
    # text_b = "How many fruits are there in the sequence? pear, pineapple, mango."
    # text_c = "How many fruits are there in the sequence? pear, pear, pear, pear, pear, pear, grape, apple."
    
    
    # text_a = "banana, orange, pear, pineapple, mango, strawberry, grape, apple. How many fruits are there in the sequence?"
    # text_b = "mango, strawberry, grape, apple, banana, orange, pear, pineapple."
    
    text_a = "\n banana, orange, pear, pineapple, mango, strawberry, grape, apple. Which position is the apple in the fruits sequence?"
    # text_b = "\n cherry, strawberry, grape, apple, cherry, orange, pear, pineapple."
    # text_b = "\n cherry, orange, pear, cherry, cherry, strawberry, grape, apple."
    text_b = "\n pineapple, mango, strawberry, grape, apple."
    # text_b = "orange, orange, orange, orange, orange, strawberry, grape, apple. Which number is the apple among the fruits in the sequential sequence?"
    # text_a = "banana, orange, pear, pineapple, mango, strawberry, grape, apple. What is the position of the apple in the sequence?"
    # text_b = "\n mango, mango, mango, mango, mango, strawberry, grape, apple."
    # text_c = "\n mango, orange, pear, pineapple, time, time, time, time."
    # text_d = "\n time, time, time, time, time, time, time, time. Which number is the apple among the fruits in the sequential sequence?"
    # text_c = "pineapple, orange, pear, pineapple, pear, pear, pear, pear."
    
    # text_c = "How many fruits are there in the sequence? apple, banana, orange, pear, mango, strawberry, kiwi, grape, watermelon."
    # text_a = "Please help me analyze the personality traits of the character Jon Snow in A Song of Ice and Fire."
    # text_b = "Lin Daiyu in Dream of the Red Chamber. Please help me analyze the personality traits of the character."
    # text_c = "Daenerys Targaryen in A Song of Ice and Fire. Please help me analyze the personality traits of the character."
    
    # text_a = "序列中有几个国家，阿根廷、亚美尼亚、澳大利亚、奥地利、比利时、巴西、保加利亚、布基那法索、喀麦隆、加拿大、智利、哥斯达黎加、科特迪瓦、克罗地亚、捷克共和国、丹麦、爱沙尼亚。"
    # text_b = "序列中有几个国家，阿根廷、亚美尼亚、澳大利亚、奥地利、比利时、布基那法索、喀麦隆、加拿大、智利、哥斯达黎加、科特迪瓦、克罗地亚、丹麦、爱沙尼亚。"
    # text_c = "序列中有几个国家，克罗地亚、捷克共和国、丹麦、爱沙尼亚。"
    template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
    model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

    # 检查缓存文件是否存在
    cache_path = "examples/pipeline/cache/results_cache.json"
    # if os.path.exists(cache_path):
    #     # 如果缓存存在,直接加载
    #     results = load_results_from_cache(cache_path)
    # else:
    #     # 如果缓存不存在,重新计算并保存
    results = generate_reuse_output(text_a, [text_b], template, model_name,window_size=4)
    save_results_to_cache(results, cache_path)
    
    print_reuse_results(results)
    
    # # 可视化交叉注意力
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    plot_attention_combined(results, tokenizer)
    # plot_cross_attention(results, tokenizer)
    # plot_forward_attention(results, tokenizer)
    # plot_token_kv_errors(results, tokenizer)
    # plot_attention_matrix(results, tokenizer)
    # plot_attention_error(results, tokenizer, num_decode_tokens=4)
    # plot_kv_error_by_window_size("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250405_windows_output_qwen2.5-32b.json", "examples/pipeline/images/kv_error_by_window_size.png")
    
    