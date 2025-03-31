import json
from kvshare_new_pipeline import KVShareNewPipeline
from nll_demo import calculate_nll
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


def generate_output_data(input_path: str, output_path: str):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name,device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    
    template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nTranslate the following text from Chinese to English:\n{text}\n<|im_end|>\n<|im_start|>assistant\n"""
    
    all_data = data["all_translations"]
    similar_pairs = data["similar_pairs"]
    similar_pairs = random.sample(similar_pairs,2000)
    save_data = []

    BLEU = evaluate.load('bleu')
    for item in tqdm(similar_pairs,total=len(similar_pairs)):
        # if key not in sample_keys:
        #     continue
        
        try:
            question = all_data[str(item["id"])]["zh"]
            answer = all_data[str(item["id"])]["en"]
            
            # 1. Full Compute
            sampling_params = SamplingParams(temperature=0, max_tokens=256)
            target_text = template.format(text=question)
            full_compute_output,target_token_ids,ttft_time = KVShareNewPipeline.full_compute(pipeline.model,sampling_params,target_text)
            
            item["output"] = full_compute_output
            item["bleu"] = BLEU.compute(predictions=[full_compute_output],references=[answer])
            profile_similar_top5_docs = []
            for index in range(0,len(item["cosine_similarity_top5"]),1):
                    # if str(item["cosine_similarity_top5"][index]["id"]) == str(key):
                    #     continue
                source_doc = all_data[str(item["cosine_similarity_top5"][index]["id"])]
              
                source_text = source_doc["zh"]
                source_text = template.format(text=source_text)
                target_text = template.format(text=question)
                
                sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
                source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,sampling_params_only_one,source_text)
                
                diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
                modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(
                    source_token_ids, target_token_ids, source_kvcache, diff_report)
                
                high_sim_output,_,_ = KVShareNewPipeline.partial_compute(
                    pipeline.model, sampling_params, target_text,
                    reused_map_indices, unused_map_indices, modified_kvcache)
                
                
                # source_doc["output"] = high_sim_output
                # is_correct_high_sim = is_correct(high_sim_output,answer)          
               
                profile_similar_top5_docs.append({
                    "id":item["cosine_similarity_top5"][index]["id"],
                    "output":high_sim_output,
                    "bleu": BLEU.compute(predictions=[high_sim_output],references=[answer]),
                    "cosine_similarity":item["cosine_similarity_top5"][index]["similarity"]
                })
            item["cosine_similarity_top5"] = profile_similar_top5_docs
            
            # 3. High Similarity Partial Compute
            profile_reused_token_num_top5_docs = []
            for index in range(0,len(item["reused_token_num_top5"]),1):
                # if str(item["reused_token_num_top5"][index]["id"]) == str(key):
                #     continue
                source_doc = all_data[str(item["reused_token_num_top5"][index]["id"])]
                source_text = source_doc["zh"]
                source_text = template.format(text=source_text)
                target_text = template.format(text=question)
                
                sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
                source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,sampling_params_only_one,source_text)
                
                diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
                modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(
                    source_token_ids, target_token_ids, source_kvcache, diff_report)

                high_reused_token_output,_,_ = KVShareNewPipeline.partial_compute(
                    pipeline.model, sampling_params, target_text,
                    reused_map_indices, unused_map_indices, modified_kvcache)
                
                # item["output"] = high_sim_output
                # is_correct_high_reused_token = is_correct(high_reused_token_output,answer)
                profile_reused_token_num_top5_docs.append({
                    "id":item["reused_token_num_top5"][index]["id"],
                    "output":high_reused_token_output,
                    "bleu": BLEU.compute(predictions=[high_reused_token_output],references=[answer]),
                    "reused_token_num":item["reused_token_num_top5"][index]["reused_token_num"]
                })
            item["reused_token_num_top5"] = profile_reused_token_num_top5_docs
            save_data.append(item)
        except Exception as e:
            print(e)
            continue
        json.dump(save_data,open(output_path,"w"),indent=4,ensure_ascii=False)
        
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
            # 全量计算的BLEU
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
        'Full Compute': '#1f77b4',      # 蓝色
        'Similarity Top1': '#2ca02c',    # 绿色
        'Reused Token Top1': '#ff7f0e'   # 橙色
    }
    labels = list(colors.keys())
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 箱线图比较
    data_to_plot = [full_compute_bleu, similarity_top1_bleu, reused_token_top1_bleu]
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # 设置箱线图颜色
    for patch, label in zip(bp['boxes'], labels):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.6)
    
    # 添加数据点
    for i, (data_points, label) in enumerate(zip(data_to_plot, labels), 1):
        ax1.scatter([i] * len(data_points), data_points, 
                   alpha=0.3, color=colors[label], s=20)
    
    ax1.set_ylabel('BLEU Score')
    ax1.set_title('BLEU Score Distribution Comparison')
    ax1.grid(True, alpha=0.3)
    
    # 2. 密度分布图
    for scores, label in zip(data_to_plot, labels):
        sns.kdeplot(data=scores, 
                   label=label, 
                   ax=ax2, 
                   color=colors[label],
                   fill=True, 
                   alpha=0.3)
    
    ax2.set_xlabel('BLEU Score')
    ax2.set_ylabel('Density')
    ax2.set_title('BLEU Score Density Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 添加统计信息
    stats_text = (
        f"Statistics:\n"
        f"Full Compute: {np.mean(full_compute_bleu):.4f} ± {np.std(full_compute_bleu):.4f} (N={len(full_compute_bleu)})\n"
        f"Similarity Top1: {np.mean(similarity_top1_bleu):.4f} ± {np.std(similarity_top1_bleu):.4f} (N={len(similarity_top1_bleu)})\n"
        f"Reused Token Top1: {np.mean(reused_token_top1_bleu):.4f} ± {np.std(reused_token_top1_bleu):.4f} (N={len(reused_token_top1_bleu)})"
    )
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
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

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1.json"
    output_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1_output.json"
    # generate_output_data(input_path,output_path)
    plot_bleu_comparison(output_path)