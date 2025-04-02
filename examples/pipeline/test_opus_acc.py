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
import matplotlib
from matplotlib import font_manager 
# download the font files and save in this fold
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_output_data(input_path: str, output_path: str,model_name = "Qwen/Qwen2.5-7B-Instruct"):
    
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
        'Full Compute': 'red',      # 红色
        'Similarity Top1': 'green',    # 绿色
        'Reused Token Top1': 'blue'   # 蓝色
    }
    labels = list(colors.keys())
    
    # 创建单个图
    plt.figure(figsize=(8, 6))
    
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
    mean_text = "Means:\n" + "\n".join([f"{label}: {mean:.3f}" for label, mean in means])
    plt.text(0.02, 0.87, mean_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('BLEU Score')
    plt.ylabel('Cumulative Probability')
    plt.title('BLEU Score Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # 将图例放在图表右上角
    plt.legend(loc='lower center')
    
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
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1.json"
    
    # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # output_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1_output_qwen2.5-32b.json"
    # generate_output_data(input_path,output_path)
    
    # outputs = [
    #     ("Llama3.1-8B",)
    # ]
    
    # plot_bleu_comparison(output_path)