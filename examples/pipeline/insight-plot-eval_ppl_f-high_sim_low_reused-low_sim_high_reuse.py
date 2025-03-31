import json
from kvshare_new_pipeline import KVShareNewPipeline
from nll_demo import calculate_nll,calculate_ppl
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import os
import random

def generate_output_data(input_path: str, output_path: str):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name,device)
    
    # sampling_params = SamplingParams(temperature=0, max_tokens=100)
    # nll_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",local_files_only=True)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    data = random.sample(json.load(open(input_path,"r")),100)
    for idx,item in tqdm(enumerate(data),total=len(data)):
        if item["source_token_len"] > 3000 or item["target_token_len"] > 3000:
            continue
        try:
            source_text = item["source_text"]
            target_text = item["target_text"]
            template="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            source_text = template.format(prompt=source_text)
            target_text = template.format(prompt=target_text)
            
            sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
            source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(pipeline.model,sampling_params_only_one,source_text)
            sampling_params = SamplingParams(temperature=0, max_tokens=512)
            target_gt_output,target_token_ids,ttft_time = KVShareNewPipeline.full_compute(pipeline.model,sampling_params,target_text)
            
            diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
            modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(source_token_ids,
                                                                                                            target_token_ids,
                                                                                                            source_kvcache,
                                                                                                            diff_report)
            sampling_params = SamplingParams(temperature=0, max_tokens=512)
            modified_output,modified_token_ids,ttft_time = KVShareNewPipeline.partial_compute(pipeline.model,
                                                                                            sampling_params,
                                                                                            target_text,
                                                                                            reused_map_indices,
                                                                                            unused_map_indices,
                                                                                            modified_kvcache)

            data[idx]["llm_output"] = {
                "target_output_full_compute":target_gt_output,
                "target_output_partial_compute":modified_output,      
            }
            
        except Exception as e:
            print(e)
            continue
    json.dump(data,open(output_path,"w"),indent=4,ensure_ascii=False)
    
    
def plot_ppl(data_path: str):
    """
    绘制cosine相似度与PPL的关系散点图
    Args:
        data_path: 数据文件路径
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # 加载数据
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # 提取数据
    cosine_similarities = []
    gt_ppls = []
    modified_ppls = []
    
    for item in data:
        # 确保所需的键存在且值不为None
        if all(key in item for key in ["gt_ppl", "modified_ppl", "cosine_similarity"]):
            try:
                # 将值转换为浮点数
                cos_sim = float(item["cosine_similarity"]["colbert"])
                gt_ppl = float(item["gt_ppl"])
                mod_ppl = float(item["modified_ppl"])
                
                # 添加到列表中
                cosine_similarities.append(cos_sim)
                gt_ppls.append(gt_ppl)
                modified_ppls.append(mod_ppl)
            except (ValueError, TypeError):
                continue
    
    
    # 将列表转换为numpy数组
    cosine_similarities = np.array(cosine_similarities)
    gt_ppls = np.array(gt_ppls)
    modified_ppls = np.array(modified_ppls)
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制散点图
    plt.scatter(cosine_similarities, gt_ppls, 
               alpha=0.5, label='Ground Truth PPL', 
               color='blue', s=50)
    plt.scatter(cosine_similarities, modified_ppls, 
               alpha=0.5, label='Modified PPL', 
               color='red', s=50)
    
    # 添加趋势线
    gt_z = np.polyfit(cosine_similarities, gt_ppls, 1)
    gt_p = np.poly1d(gt_z)
    plt.plot(cosine_similarities, gt_p(cosine_similarities), 
             "b--", alpha=0.8, label='GT PPL Trend')
    
    mod_z = np.polyfit(cosine_similarities, modified_ppls, 1)
    mod_p = np.poly1d(mod_z)
    plt.plot(cosine_similarities, mod_p(cosine_similarities), 
             "r--", alpha=0.8, label='Modified PPL Trend')
    
    # 计算相关系数
    gt_corr = np.corrcoef(cosine_similarities, gt_ppls)[0,1]
    mod_corr = np.corrcoef(cosine_similarities, modified_ppls)[0,1]
    
    # 添加相关系数文本
    plt.text(0.05, 0.95, 
             f'GT Correlation: {gt_corr:.3f}\nModified Correlation: {mod_corr:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='top')
    
    # 设置图表属性
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Perplexity (PPL)')
    plt.title('Relationship between Cosine Similarity and PPL')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig("examples/pipeline/images/insight2/cosine_similarity_ppl_relationship.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"数据点数量: {len(cosine_similarities)}")
    print(f"Ground Truth PPL:")
    print(f"  - 平均值: {np.mean(gt_ppls):.3f}")
    print(f"  - 标准差: {np.std(gt_ppls):.3f}")
    print(f"  - 与相似度相关系数: {gt_corr:.3f}")
    print(f"\nModified PPL:")
    print(f"  - 平均值: {np.mean(modified_ppls):.3f}")
    print(f"  - 标准差: {np.std(modified_ppls):.3f}")
    print(f"  - 与相似度相关系数: {mod_corr:.3f}")


def compute_ppl(data_path: str):
    """
    计算PPL
    Args:
        data_path: 数据文件路径
    """
    device = "cuda:0"
    nll_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct").to(device).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",local_files_only=True)
    
    with open(data_path, "r") as f:
        data = json.load(f)
    # data = random.sample(json.load(open(input_path,"r")),100)
    for idx,item in tqdm(enumerate(data),total=len(data)):
        # if item["source_token_len"] > 3000 or item["target_token_len"] > 3000:
        #     continue
        try:
            target_output_full_compute = item["llm_output"]["target_output_full_compute"]
            target_output_partial_compute = item["llm_output"]["target_output_partial_compute"]
            # sampling_params = SamplingParams(temperature=0, max_tokens=512)
            
            ppl_full_compute = calculate_ppl(target_output_full_compute,nll_model,tokenizer)
            ppl_partial_compute = calculate_ppl(target_output_partial_compute,nll_model,tokenizer)
            
            item["llm_output"]["full_compute_ppl"] = ppl_full_compute
            item["llm_output"]["partial_compute_ppl"] = ppl_partial_compute
            
        except Exception as e:
            print(e)
            continue
    json.dump(data,open(data_path,"w"),indent=4,ensure_ascii=False)
        
if __name__ == "__main__":  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    input_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity_profiled.json"
    output_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity_profiled_ppl_qwen2.5-7B.json"
    # sampled_data = generate_output_data(input_path, output_path)
    compute_ppl(output_path)
    # plot_ppl(output_path)

