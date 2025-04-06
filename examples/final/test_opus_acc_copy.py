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
    # similar_pairs = random.sample(similar_pairs, min(len(similar_pairs), 400))
    
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
            
            source_token_ids = [output.prompt_token_ids for output in source_outputs]
            max_request_id = max(int(req_output.request_id) for req_output in source_outputs)
            
            # 对不同window size进行处理
            for window_size in [6,12,24]:
                # 批量KV编辑
                target_kv_cache, reused_indices, unreused_indices, selected_tokens, target_slices = KVEditor.batch_kvedit(
                    [tokenizer.encode(prompt) for prompt in batch_target_prompts],
                    source_token_ids,
                    source_kv_cache,
                    window_size=window_size
                )
                
                target_req_ids = [max_request_id + 1 + idx for idx in range(len(batch_target_prompts))]
                
                # 批量partial计算
                target_outputs = KVShareNewPipeline.partial_compute(
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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_copy.json"
    
    
    
    
    # split_data_by_windows_size(input_path, "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json")
    # windows_sizes = [3,5,7,9,11,13,15]
    
    
    # # 创建进程池，进程数取窗口大小数量和CPU核心数的较小值
    # num_processes = min(len(windows_sizes), mp.cpu_count())
    # pool = mp.Pool(processes=num_processes)
    
    # # 使用partial固定input_path参数
    # process_func = partial(process_window_size, input_path)
    
    # # 并行处理所有窗口大小
    # pool.map(process_func, windows_sizes)
    
    # # 关闭进程池
    # pool.close()
    # pool.join()
    
    template = qwen_template
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    # template = llama3_template_text
    # model_name = "LLM-Research/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_llama3.1-8b.json"
    # generate_output_data(input_path,output_path,model_name=model_name,batch_size=32)
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_llama3.1-70b.json"
    # generate_output_data(input_path,output_path,model_name=model_name,batch_size=32)
    
    input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
    output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250405_windows_output_qwen2.5-32b.json"
    generate_output_data(input_path,output_path,model_name=model_name,batch_size=72,max_generate_len=512)
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_qwen2.5-32b.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_fc_output_qwen2.5-32b.json"
    # compute_full_compute_acc(input_path,output_path,model_name,batch_size=24)
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_llama3.1-8b.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_fc_output_llama3.1-8b.json"
    # compute_full_compute_acc(input_path,output_path,model_name,batch_size=24)
    
    # 绘制不同窗口大小下的METEOR分数的CDF曲线对比
    # plot_meteor_by_window_size(output_path, "examples/pipeline/images/opus_meteor_by_window_qwen2.5-32b.png")
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_7_output_qwen2.5-32b.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_15_output_qwen2.5-32b.json"
    # generate_output_data(input_path,output_path,batch_size=1,model_name=model_name,window_size=15)
    
    # output_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json"
    # generate_output_data(input_path,output_path,batch_size=2,model_name=model_name)
    # plot_bleu_comparison(output_path)
    
    
    # plot_bleu_comparison("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json","examples/pipeline/images/opus_bleu_comparison_llama3.1-8b.pdf")
    # plot_bleu_comparison("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1_output_qwen2.5-7b.json","examples/pipeline/images/opus_bleu_comparison_qwen32-7b.pdf")
    # outputs = [
    #     ("Llama3.1-8B","examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json"),
    #     ("Qwen2.5-32B","examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_qwen2.5-7b.json"),
    # ]
    # for model_name,output_path in outputs:
    #     generate_output_data(input_path,output_path,batch_size=2,model_name=model_name)
    #     plot_bleu_comparison(output_path)

    # 只保存过滤后的数据
    # save_filtered_data(
    #     "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_qwen2.5-32b.json",
    #     "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_qwen2.5-32b_filtered.json"
    # )

    # 或者直接使用plot_meteor_by_window_size，它会自动保存过滤后的数据
    # plot_meteor_by_window_size(
    #     "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_fc_output_qwen2.5-32b.json",
    #     "examples/pipeline/images/opus_meteor_by_window_qwen2.5-32b.pdf"
    # )
    # 或者直接使用plot_meteor_by_window_size，它会自动保存过滤后的数据
    # plot_meteor_by_window_size(
    #     "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_llama3.1-8b.json",
    #     "examples/pipeline/images/opus_meteor_by_window_llama3.1-8b.pdf",
    #     tag="reused_top1_w31"
    # )
    # 绘制相似度和METEOR分数的关系图
    # plot_meteor_by_window_size(
    #     "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_fc_output_qwen2.5-32b.json",
    #     "examples/pipeline/images/opus_similarity_vs_meteor.pdf"
    # )