import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evaluate import load
import os
from tqdm import tqdm
import matplotlib
from matplotlib import font_manager 

# 设置字体
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
font_files = font_manager.findSystemFonts(fontpaths=font_path)
for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_bertscore_with_sim_top1_and_reused_top1(data_path,image_path):
    data = json.load(open(data_path,"r"))
    bertscore = load("bertscore")
    
    sim_scores = []
    reused_scores = []
    
    for item in tqdm(data,desc="plot_bertscore_with_sim_top1_and_reused_top1"):
        predictions = [item["target_text"]["full_output"], item["high_token_reused_top5"][0]["partial_compute_output"]]
        references = [item["target_text"]["chatgpt_output"]]
        results = bertscore.compute(predictions=predictions, references=references,lang="en")
        if results["f1"][0] > 0:
            sim_scores.append(results["f1"][0])
        if results["f1"][1] > 0:
            reused_scores.append(results["f1"][1])
    
    # 创建图表
    plt.figure(figsize=(8, 6))
    
    # 设置颜色方案
    colors = {
        'Full Compute': 'red',      # 红色
        'Similarity Top1': 'green',    # 绿色
        'Reused Token Top1': 'blue'   # 蓝色
    }
    
    # 添加ChatGPT参考线
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Ground Truth')
    
    # 计算并绘制CDF曲线
    def plot_cdf(data, color, label):
        sorted_data = np.sort(data)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, p, color=color, label=label, alpha=0.7)
        return np.mean(data)
    
    # 绘制CDF曲线
    sim_mean = plot_cdf(sim_scores, colors['Similarity Top1'], 'Similarity Top1')
    reused_mean = plot_cdf(reused_scores, colors['Reused Token Top1'], 'Reused Token Top1')
    
    # 添加均值标注
    mean_text = "Means:\n" + f"Similarity Top1: {sim_mean:.3f}\nReused Token Top1: {reused_mean:.3f}"
    plt.text(0.02, 0.9, mean_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 设置坐标轴标签和标题
    plt.xlabel('BERT Score F1')
    plt.ylabel('Cumulative Probability')
    plt.title('BERT Score Cumulative Distribution')
    
    # 设置网格
    plt.grid(True, alpha=0.3)
    
    # 设置图例位置
    plt.legend(loc='lower center')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 打印详细统计信息
    print("\nBERT分数统计:")
    print("-" * 50)
    print(f"相似度Top1:")
    print(f"  样本数量: {len(sim_scores)}")
    print(f"  平均BERT: {np.mean(sim_scores):.4f} ± {np.std(sim_scores):.4f}")
    print(f"  中位数: {np.median(sim_scores):.4f}")
    print(f"  范围: [{np.min(sim_scores):.4f}, {np.max(sim_scores):.4f}]")
    
    print(f"\n重用Token Top1:")
    print(f"  样本数量: {len(reused_scores)}")
    print(f"  平均BERT: {np.mean(reused_scores):.4f} ± {np.std(reused_scores):.4f}")
    print(f"  中位数: {np.median(reused_scores):.4f}")
    print(f"  范围: [{np.min(reused_scores):.4f}, {np.max(reused_scores):.4f}]")

if __name__ == "__main__":
    os.environ["CUDA_VISIBALE_DEVICES"]="1"
    data_path = "examples/final/data/instruction_wildv2_similar_250331_answer_by_Qwen2.5-7B-Instruct_partial_output.json"
    image_path = "examples/pipeline/images/instruction_wildv2_similar_250331_bertscore_distribution.png"
    plot_bertscore_with_sim_top1_and_reused_top1(data_path,image_path)