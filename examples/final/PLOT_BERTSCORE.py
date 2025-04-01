import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from evaluate import load
import os
from tqdm import tqdm

def plot_bertscore_with_sim_top1_and_reused_top1(data_path,image_path):
    data = json.load(open(data_path,"r"))
    bertscore = load("bertscore")
    
    sim_scores = []
    reused_scores = []
    
    for item in tqdm(data,desc="plot_bertscore_with_sim_top1_and_reused_top1"):
        predictions = [item["high_similarity_top5"][0]["partial_compute_output"], item["high_token_reused_top5"][0]["partial_compute_output"]]
        references = [item["full_compute_output"],item["full_compute_output"]]
        results = bertscore.compute(predictions=predictions, references=references,lang="en")
        if results["f1"][0] > 0:
            sim_scores.append(results["f1"][0])
        if results["f1"][1] > 0:
            reused_scores.append(results["f1"][1])
    
    # 创建图表
    plt.figure(figsize=(6, 4))  # 调整图形比例以适应交换后的轴
    
    # 设置颜色
    colors = ['#2ecc71', '#3498db']  # 绿色和蓝色
    
    # 计算生存函数
    def plot_survival(data, color, label):
        sorted_data = np.sort(data)[::-1]  
        survival_rate = np.linspace(0, 100, len(sorted_data))
        plt.plot(survival_rate, sorted_data, color=color, label=label)
        return survival_rate, sorted_data

    # 首先绘制理想的full compute曲线（作为背景）
    full_compute_x = np.array([0, 100])
    full_compute_y = np.array([1.0, 1.0])
    plt.plot(full_compute_x, full_compute_y, color='#cccccc', label='Full Compute', 
            linestyle='-', linewidth=2, alpha=0.5)

    # 绘制其他分布图
    sim_x, sim_y = plot_survival(sim_scores, colors[0], 'Similarity Top1')
    reused_x, reused_y = plot_survival(reused_scores, colors[1], 'Reused Top1')
    
    # 计算并显示均值
    sim_mean = np.mean(sim_scores)
    reused_mean = np.mean(reused_scores)
    
    # 添加均值水平线
    plt.axhline(y=sim_mean, color=colors[0], linestyle='--', alpha=0.8)
    plt.axhline(y=reused_mean, color=colors[1], linestyle='--', alpha=0.8)
    
    # 设置x轴范围（百分比）
    plt.xlim(0, 100)
    
    # 设置y轴范围（分数），将上限调整为1.05，以显示参考线
    plt.ylim(0.5, 1.05)
    
    # 添加均值标注
    plt.text(95, sim_mean, f'μ={sim_mean:.3f}', 
            color=colors[0], horizontalalignment='right', verticalalignment='bottom')
    plt.text(85, reused_mean, f'μ={reused_mean:.3f}', 
            color=colors[1], horizontalalignment='right', verticalalignment='bottom')
    
    # 添加参考线在1.0处
    # plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # 设置标题和标签
    plt.title('Distribution of BERTScore F1 Scores')
    plt.xlabel('Percentage of Samples Above Score')
    plt.ylabel('F1 Score')
    
    # 调整图例位置（确保能显示所有三条曲线）
    plt.legend(loc='lower left')
    
    # 保存图片
    plt.savefig(image_path)
    plt.close()

if __name__ == "__main__":
    os.environ["CUDA_VISIBALE_DEVICES"]="1"
    data_path = "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331_output.json"
    image_path = "examples/pipeline/images/instruction_wildv2_similar_250331_bertscore_distribution.png"
    plot_bertscore_with_sim_top1_and_reused_top1(data_path,image_path)