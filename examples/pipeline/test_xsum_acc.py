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
# os.environ["VLLM_USE_MODELSCOPE"] = "true"

def generate_output_data(input_path: str, output_path: str):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name,device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    data = random.sample(data,min(len(data),100))
    rouge = load("rouge")
    
    template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nYou need to predict a very short sentence based on the input news text to describe what the article is about. Please keep the output in English and do not output other irrelevant information. Document:{prompt}.<|im_end|>\n<|im_start|>assistant\n """
    
    avg_full_compute_rouge = []
    for idx,item in tqdm(enumerate(data),total=len(data)):
        try:
            document = item["document"]
            summary = item["summary"]
            
            # 1. Full Compute
            sampling_params = SamplingParams(temperature=0, max_tokens=128)
            target_text = template.format(prompt=document)
            full_compute_output,target_token_ids,ttft_time = KVShareNewPipeline.full_compute(pipeline.model,sampling_params,target_text)
            
            rouge_scores = rouge.compute(
                predictions=[full_compute_output],
                references=[summary],
                rouge_types=["rougeL"]
            )
            item["output"] = full_compute_output
            item["rougeL"] = rouge_scores["rougeL"]
            # avg_full_compute_rouge.append(rouge_scores["rougeL"])
            
            # 2. High Similarity Partial Compute
            profile_similar_docs = []
            for index in range(0,len(item["similar_docs"]),1):
            
                source_doc = item["similar_docs"][index]
                
                if source_doc["similarity"] < 0.99999:
                    continue
                
                if source_doc["reused_token_num"] < 10:
                    continue
                
                source_text = source_doc["document"]
                source_text = template.format(prompt=source_text)
                target_text = template.format(prompt=document)
                
                sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
                source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,sampling_params_only_one,source_text)
                
                diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
                modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(
                    source_token_ids, target_token_ids, source_kvcache, diff_report)
                
                high_sim_output,_,_ = KVShareNewPipeline.partial_compute(
                    pipeline.model, sampling_params, target_text,
                    reused_map_indices, unused_map_indices, modified_kvcache)
                
                rouge_scores = rouge.compute(
                    predictions=[high_sim_output],
                    references=[summary],
                    rouge_types=["rougeL"]
                )
                source_doc["output"] = high_sim_output
                source_doc["rougeL"] = rouge_scores["rougeL"]
                profile_similar_docs.append(source_doc)
            item["similar_docs"] = profile_similar_docs
            save_data.append(item)
            
        except Exception as e:
            print(e)
            continue
            
    # print(f"Average RougeL scores:")
    # print(f"Full Compute: {np.mean(avg_full_compute_rouge):.4f}")
    # print(f"High Similarity: {np.mean(avg_high_sim_rouge):.4f}")
    # print(f"Max Reused: {np.mean(avg_max_reused_rouge):.4f}")
    
    json.dump(save_data,open(output_path,"w"),indent=4,ensure_ascii=False)


def plot_acc_with_high_sim(output_path: str, image_save_path: str = "examples/pipeline/images/rouge_scores_comparison.png"):
    # 加载数据
    data = json.load(open(output_path, "r"))
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 定义top-k列表和对应的颜色
    top_k_list = [1, 5, 10]
    colors = ['red', 'blue', 'green']
    
    # 收集所有数据点，按top-k分组
    grouped_data = {k: [] for k in top_k_list}
    
    for item in data:
        full_compute_rouge = item["rougeL"]
        if full_compute_rouge < 0.1:
            continue
            
        # 对similar_docs按相似度排序
        sorted_docs = sorted(item["similar_docs"], 
                           key=lambda x: float(x.get("similarity", 0)), 
                           reverse=True)
        
        # 为每个top-k收集数据
        for k in top_k_list:
            top_k_docs = sorted_docs[:k]
            for doc in top_k_docs:
                try:
                    if (all(key in doc for key in ["similarity", "resued_token_num", "rougeL"]) and 
                        doc["rougeL"] >= 0.1 and
                        10 <= float(doc["resued_token_num"]) <= 300 and
                        float(doc["similarity"]) >= 0.4):
                        grouped_data[k].append({
                            "similarity": float(doc["similarity"]),
                            "reused_tokens": float(doc["resued_token_num"]),
                            "partial_rouge": float(doc["rougeL"]),
                            "full_rouge": float(full_compute_rouge),
                            "rouge_diff": float(doc["rougeL"]) - float(full_compute_rouge)
                        })
                except (KeyError, ValueError, TypeError) as e:
                    continue
    
    # 绘制趋势分布
    for idx, (ax, x_key, x_label) in enumerate([
        (ax1, "similarity", "Similarity Score"),
        (ax2, "reused_tokens", "Reused Token Count")
    ]):
        try:
            # 为每个top-k绘制趋势线
            for k_idx, k in enumerate(top_k_list):
                data_points = grouped_data[k]
                if not data_points:
                    continue
                    
                x = np.array([d[x_key] for d in data_points])
                y = np.array([d["rouge_diff"] for d in data_points])
                
                # 计算趋势统计
                bin_edges = np.linspace(min(x), max(x), 10)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                means = []
                stds = []
                
                for i in range(len(bin_edges)-1):
                    mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
                    bin_data = y[mask]
                    
                    if len(bin_data) > 0:
                        means.append(np.mean(bin_data))
                        stds.append(np.std(bin_data))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                means = np.array(means)
                stds = np.array(stds)
                valid_mask = ~np.isnan(means)
                
                # 绘制均值线
                ax.plot(bin_centers[valid_mask], means[valid_mask],
                       color=colors[k_idx],
                       linewidth=2,
                       label=f'Top-{k}')
                
                # 绘制标准差范围
                ax.fill_between(bin_centers[valid_mask],
                              means[valid_mask] - stds[valid_mask],
                              means[valid_mask] + stds[valid_mask],
                              color=colors[k_idx],
                              alpha=0.1)
                
                # 添加相关系数
                correlation = np.corrcoef(x, y)[0, 1]
                ax.text(0.02, 0.98 - k_idx*0.06, 
                       f'Top-{k} (N={len(x)}, Corr={correlation:.3f})',
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8),
                       verticalalignment='top',
                       color=colors[k_idx])
            
            # 设置图表属性
            ax.set_xlabel(x_label)
            ax.set_ylabel('RougeL Difference\n(Partial - Full)')
            ax.set_title(f'{x_label} vs RougeL Difference')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
            
        except Exception as e:
            print(f"绘制 {x_label} 图表时出错: {e}")
            continue
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(image_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    

def analyze_top_performance(output_path: str):
    """分析并展示不同策略的性能指标"""
    # 加载数据
    data = json.load(open(output_path, "r"))
    
    # 定义要分析的top-k值
    top_k_values = [1, 5, 10, 15]
    
    # 收集性能数据
    performance_data = {
        "full_compute": []
    }
    # 为每个top-k创建数据收集列表
    for k in top_k_values:
        performance_data[f"similarity_top{k}"] = []
        performance_data[f"reused_tokens_top{k}"] = []
    
    for item in data:
        full_compute_rouge = item["rougeL"]
        if full_compute_rouge < 0.1:
            continue
            
        performance_data["full_compute"].append(full_compute_rouge)
        
        # 获取similar_docs并过滤无效数据
        valid_docs = []
        for doc in item["similar_docs"]:
            try:
                if (all(key in doc for key in ["similarity", "resued_token_num", "rougeL"]) and 
                    doc["rougeL"] >= 0.1):
                    valid_docs.append({
                        "similarity": float(doc["similarity"]),
                        "reused_tokens": float(doc["resued_token_num"]),
                        "rougeL": float(doc["rougeL"])
                    })
            except (KeyError, ValueError, TypeError):
                continue
        
        if not valid_docs:
            continue
            
        # 对文档按相似度和重用token数排序
        similarity_sorted = sorted(valid_docs, key=lambda x: x["similarity"], reverse=True)
        tokens_sorted = sorted(valid_docs, key=lambda x: x["reused_tokens"], reverse=True)
        
        # 为每个top-k收集数据
        for k in top_k_values:
            if len(similarity_sorted) >= k:
                top_k_sim = similarity_sorted[:k]
                performance_data[f"similarity_top{k}"].append(np.mean([d["rougeL"] for d in top_k_sim]))
            
            if len(tokens_sorted) >= k:
                top_k_tokens = tokens_sorted[:k]
                performance_data[f"reused_tokens_top{k}"].append(np.mean([d["rougeL"] for d in top_k_tokens]))
    
    # 计算统计指标
    stats = {}
    for method, scores in performance_data.items():
        if scores:
            stats[method] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "count": len(scores),
                "median": np.median(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
    
    # 打印结果
    print("\n性能分析结果:")
    print("-" * 50)
    for method, metrics in stats.items():
        print(f"\n{method}:")
        print(f"  样本数量: {metrics['count']}")
        print(f"  平均RougeL: {metrics['mean']:.4f} ± {metrics['std']:.4f}")
        print(f"  中位数: {metrics['median']:.4f}")
        print(f"  范围: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
    
    # 创建子图布局
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(gs[0, 0]),  # 箱线图
        fig.add_subplot(gs[0, 1]),  # top-k相似度性能
        fig.add_subplot(gs[1, 0]),  # top-k重用token性能
        fig.add_subplot(gs[1, 1]),  # 综合对比
    ]
    
    # 1. 箱线图比较
    data_to_plot = [performance_data["full_compute"]]
    labels = ['Full Compute']
    for k in top_k_values:
        data_to_plot.extend([
            performance_data[f"similarity_top{k}"],
            performance_data[f"reused_tokens_top{k}"]
        ])
        labels.extend([f'Sim Top{k}', f'Token Top{k}'])
    
    bp = axes[0].boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_xticklabels(labels, rotation=45)
    axes[0].set_ylabel('RougeL Score')
    axes[0].set_title('Performance Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Top-k相似度性能趋势
    sim_means = [stats[f"similarity_top{k}"]["mean"] for k in top_k_values]
    sim_stds = [stats[f"similarity_top{k}"]["std"] for k in top_k_values]
    axes[1].errorbar(top_k_values, sim_means, yerr=sim_stds, 
                    marker='o', label='Similarity Strategy')
    axes[1].axhline(y=stats["full_compute"]["mean"], color='r', 
                    linestyle='--', label='Full Compute Mean')
    axes[1].fill_between([min(top_k_values), max(top_k_values)],
                        stats["full_compute"]["mean"] - stats["full_compute"]["std"],
                        stats["full_compute"]["mean"] + stats["full_compute"]["std"],
                        color='r', alpha=0.1)
    axes[1].set_xlabel('Top-k')
    axes[1].set_ylabel('Mean RougeL Score')
    axes[1].set_title('Similarity Strategy Performance vs k')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Top-k重用token性能趋势
    token_means = [stats[f"reused_tokens_top{k}"]["mean"] for k in top_k_values]
    token_stds = [stats[f"reused_tokens_top{k}"]["std"] for k in top_k_values]
    axes[2].errorbar(top_k_values, token_means, yerr=token_stds, 
                    marker='o', label='Token Reuse Strategy')
    axes[2].axhline(y=stats["full_compute"]["mean"], color='r', 
                    linestyle='--', label='Full Compute Mean')
    axes[2].fill_between([min(top_k_values), max(top_k_values)],
                        stats["full_compute"]["mean"] - stats["full_compute"]["std"],
                        stats["full_compute"]["mean"] + stats["full_compute"]["std"],
                        color='r', alpha=0.1)
    axes[2].set_xlabel('Top-k')
    axes[2].set_ylabel('Mean RougeL Score')
    axes[2].set_title('Token Reuse Strategy Performance vs k')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # 4. 综合对比
    x = np.arange(len(top_k_values))
    width = 0.35
    axes[3].bar(x - width/2, sim_means, width, label='Similarity Strategy',
                yerr=sim_stds, capsize=5)
    axes[3].bar(x + width/2, token_means, width, label='Token Reuse Strategy',
                yerr=token_stds, capsize=5)
    axes[3].axhline(y=stats["full_compute"]["mean"], color='r',
                    linestyle='--', label='Full Compute Mean')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([f'Top-{k}' for k in top_k_values])
    axes[3].set_ylabel('Mean RougeL Score')
    axes[3].set_title('Strategy Comparison')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig('examples/pipeline/images/performance_comparison_detailed.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    input_path = "examples/dataset/data/xsum/all-MiniLM-L6-v2_train_similar_docs_topk50_test2.json"
    output_path = "examples/dataset/data/xsum/all-MiniLM-L6-v2_train_similar_docs_topk50_test2_all_methods_output.json"
    
    # 分析性能
    stats = analyze_top_performance(output_path)
