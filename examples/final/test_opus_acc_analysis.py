import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from evaluate import load
import seaborn as sns
from scipy.stats import zscore
import math
import re
import evaluate
import matplotlib
import traceback

# 设置中文字体
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
font_files = font_manager.findSystemFonts(fontpaths=font_path)
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

def analyze_accuracy(input_path: str, output_path: str):
    """分析不同方法的精度表现
    
    Args:
        input_path: 输入数据文件路径
        output_path: 输出分析结果文件路径
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 加载评估指标
    meteor_metric = evaluate.load('meteor')
    bleu_metric = evaluate.load('bleu')
    
    # 初始化结果字典
    results = {
        "full_compute": {
            "meteor_scores": [],
            "bleu_scores": [],
            "correct_count": 0,
            "total_count": 0
        },
        "similarity_top1": {
            "meteor_scores": [],
            "bleu_scores": [],
            "correct_count": 0,
            "total_count": 0
        },
        "reused_token_top1": {
            "meteor_scores": [],
            "bleu_scores": [],
            "correct_count": 0,
            "total_count": 0
        }
    }
    
    # 分析每个样本
    for item in data["similar_pairs"]:
        try:
            # 获取参考答案
            reference = item["answer_a"]
            
            # 分析全量计算结果
            if "output" in item:
                full_output = item["output"]
                meteor_score = meteor_metric.compute(predictions=[full_output], references=[reference])["meteor"]
                bleu_score = bleu_metric.compute(predictions=[full_output], references=[reference])["bleu"]
                results["full_compute"]["meteor_scores"].append(meteor_score)
                results["full_compute"]["bleu_scores"].append(bleu_score)
                results["full_compute"]["total_count"] += 1
                if full_output.strip() == reference.strip():
                    results["full_compute"]["correct_count"] += 1
            
            # 分析相似度top1结果
            if "reused_top1_w31" in item:
                simi_output = item["reused_top1_w31"].get("output_w31", "")
                if simi_output:
                    meteor_score = meteor_metric.compute(predictions=[simi_output], references=[reference])["meteor"]
                    bleu_score = bleu_metric.compute(predictions=[simi_output], references=[reference])["bleu"]
                    results["similarity_top1"]["meteor_scores"].append(meteor_score)
                    results["similarity_top1"]["bleu_scores"].append(bleu_score)
                    results["similarity_top1"]["total_count"] += 1
                    if simi_output.strip() == reference.strip():
                        results["similarity_top1"]["correct_count"] += 1
            
            # 分析重用token top1结果
            if "reused_top1_w31" in item:
                reuse_output = item["reused_top1_w31"].get("output_w31", "")
                if reuse_output:
                    meteor_score = meteor_metric.compute(predictions=[reuse_output], references=[reference])["meteor"]
                    bleu_score = bleu_metric.compute(predictions=[reuse_output], references=[reference])["bleu"]
                    results["reused_token_top1"]["meteor_scores"].append(meteor_score)
                    results["reused_token_top1"]["bleu_scores"].append(bleu_score)
                    results["reused_token_top1"]["total_count"] += 1
                    if reuse_output.strip() == reference.strip():
                        results["reused_token_top1"]["correct_count"] += 1
                        
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    # 计算统计信息
    for method in results:
        if results[method]["total_count"] > 0:
            results[method]["accuracy"] = results[method]["correct_count"] / results[method]["total_count"]
            results[method]["avg_meteor"] = np.mean(results[method]["meteor_scores"])
            results[method]["std_meteor"] = np.std(results[method]["meteor_scores"])
            results[method]["avg_bleu"] = np.mean(results[method]["bleu_scores"])
            results[method]["std_bleu"] = np.std(results[method]["bleu_scores"])
    
    # 保存分析结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return results

def plot_accuracy_comparison(results: dict, save_path: str = "examples/pipeline/images/accuracy_comparison.png"):
    """绘制不同方法的精度对比图
    
    Args:
        results: 分析结果字典
        save_path: 保存图片的路径
    """
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准备数据
    methods = list(results.keys())
    accuracies = [results[method]["accuracy"] for method in methods]
    meteor_scores = [results[method]["avg_meteor"] for method in methods]
    meteor_stds = [results[method]["std_meteor"] for method in methods]
    
    # 绘制准确率柱状图
    bars1 = ax1.bar(methods, accuracies, color=['red', 'green', 'blue'], alpha=0.7)
    ax1.set_title('准确率对比')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 绘制METEOR分数柱状图
    bars2 = ax2.bar(methods, meteor_scores, yerr=meteor_stds, 
                   color=['red', 'green', 'blue'], alpha=0.7,
                   capsize=5)
    ax2.set_title('METEOR分数对比')
    ax2.set_ylabel('METEOR分数')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, std in zip(bars2, meteor_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}±{std:.3f}',
                ha='center', va='bottom')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"图表已保存至: {save_path}")

def print_accuracy_summary(results: dict):
    """打印精度分析摘要
    
    Args:
        results: 分析结果字典
    """
    print("\n精度分析摘要:")
    print("-" * 50)
    for method in results:
        if results[method]["total_count"] > 0:
            print(f"\n{method}:")
            print(f"  样本数量: {results[method]['total_count']}")
            print(f"  准确率: {results[method]['accuracy']:.4f}")
            print(f"  平均METEOR: {results[method]['avg_meteor']:.4f} ± {results[method]['std_meteor']:.4f}")
            print(f"  平均BLEU: {results[method]['avg_bleu']:.4f} ± {results[method]['std_bleu']:.4f}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250405_windows_output_qwen2.5-32b.json"
    output_path = "examples/dataset/data/opus/opus_accuracy_analysis.json"
    
    # 运行分析
    results = analyze_accuracy(input_path, output_path)
    
    # 绘制对比图
    plot_accuracy_comparison(results)
    
    # 打印摘要
    print_accuracy_summary(results) 