import json
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import os

# 设置字体
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
font_files = font_manager.findSystemFonts(fontpaths=font_path)
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_bert_scores(data):
    """计算BERT分数"""
    bert = evaluate.load("bertscore")
    full_scores = []
    sim_scores = []
    reused_scores = []
    
    for item in data:
        gpt_output = item["target_text"]["chatgpt_output"]
        full_output = item["target_text"]["full_output"]
        sim_top1_partial_output = item["sim_top1"]["partial_output"]
        reused_top1_partial_output = item["reused_top1"]["partial_output"]
        
        output_score = bert.compute(predictions=[full_output, sim_top1_partial_output, reused_top1_partial_output],
                                  references=[gpt_output, gpt_output, gpt_output],lang="en")
        
        # 收集分数
        full_scores.append(output_score['f1'][0])
        sim_scores.append(output_score['f1'][1])
        reused_scores.append(output_score['f1'][2])
    
    return full_scores, sim_scores, reused_scores

def plot_bert_cdf(ax, full_scores, sim_scores, reused_scores):
    """在给定的axes上绘制BERT CDF图"""
    # 定义统一的颜色方案
    colors = {
        'Full Compute': 'red',      # 红色
        'Sim Top1': 'green',    # 绿色
        'Reused Top1': 'blue'   # 蓝色
    }
    labels = list(colors.keys())
    
    # 添加Ground Truth参考线
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Ground Truth')
    
    # 计算CDF并绘制
    data_to_plot = [full_scores, sim_scores, reused_scores]
    means = []
    
    for scores, label in zip(data_to_plot, labels):
        # 计算CDF
        sorted_scores = np.sort(scores)
        p = np.arange(1, len(scores) + 1) / len(scores)
        
        # 绘制CDF曲线
        ax.plot(sorted_scores, p, label=label, color=colors[label], alpha=0.7)
        
        # 收集均值
        mean_value = np.mean(scores)
        means.append((label, mean_value))
    
    # 统一显示所有均值
    mean_text = "Means:\n" + "\n".join([f"{label}: {mean:.3f}" for label, mean in means])
    ax.text(0.02, 0.78, mean_text, transform=ax.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlabel('BERT Score')
    ax.set_ylabel('Cumulative Probability')
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    return means

def print_statistics(full_scores, sim_scores, reused_scores, title):
    """打印统计信息"""
    print(f"\n{title} BERT分数统计:")
    print("-" * 50)
    print(f"Full Output:")
    print(f"  样本数量: {len(full_scores)}")
    print(f"  平均BERT: {np.mean(full_scores):.4f} ± {np.std(full_scores):.4f}")
    print(f"  中位数: {np.median(full_scores):.4f}")
    print(f"  范围: [{np.min(full_scores):.4f}, {np.max(full_scores):.4f}]")
    
    print(f"\nSim Top1:")
    print(f"  样本数量: {len(sim_scores)}")
    print(f"  平均BERT: {np.mean(sim_scores):.4f} ± {np.std(sim_scores):.4f}")
    print(f"  中位数: {np.median(sim_scores):.4f}")
    print(f"  范围: [{np.min(sim_scores):.4f}, {np.max(sim_scores):.4f}]")
    
    print(f"\nReused Top1:")
    print(f"  样本数量: {len(reused_scores)}")
    print(f"  平均BERT: {np.mean(reused_scores):.4f} ± {np.std(reused_scores):.4f}")
    print(f"  中位数: {np.median(reused_scores):.4f}")
    print(f"  范围: [{np.min(reused_scores):.4f}, {np.max(reused_scores):.4f}]")

# def test_bert(data_paths, save_path):
#     """处理多个数据文件并绘制子图
#     Args:
#         data_paths: 列表，每个元素是一个元组 (data_path, title)
#         save_path: 保存图片的路径
#     """
#     n_data = len(data_paths)
#     n_cols = min(3, n_data)  # 最多3列
#     n_rows = (n_data + n_cols - 1) // n_cols  # 向上取整
    
#     # 创建子图
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
#     if n_rows == 1 and n_cols == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1:
#         axes = axes.reshape(1, -1)
    
#     # 处理每个数据文件
#     for idx, (data_path, title) in enumerate(data_paths):
#         row = idx // n_cols
#         col = idx % n_cols
#         ax = axes[row, col]
        
#         # 读取数据
#         data = json.load(open(data_path, "r"))
        
#         # 计算BERT分数
#         full_scores, sim_scores, reused_scores = calculate_bert_scores(data)
        
#         # 绘制CDF图
#         means = plot_bert_cdf(ax, full_scores, sim_scores, reused_scores, title)
        
#         # 打印统计信息
#         print_statistics(full_scores, sim_scores, reused_scores, title)
    
#     # 如果子图数量不足，隐藏多余的子图
#     for idx in range(n_data, n_rows * n_cols):
#         row = idx // n_cols
#         col = idx % n_cols
#         axes[row, col].set_visible(False)
    
#     # 调整布局并保存
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
def test_bert(data_path, save_path):
    """处理单个数据文件并绘制BERT分数CDF图
    Args:
        data_path: 数据文件路径
        title: 图表标题
        save_path: 保存图片的路径
    """
    # 创建单个图
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 读取数据
    data = json.load(open(data_path, "r"))
    
    # 计算BERT分数
    full_scores, sim_scores, reused_scores = calculate_bert_scores(data)
    
    # 绘制CDF图
    means = plot_bert_cdf(ax, full_scores, sim_scores, reused_scores)
    
    # 打印统计信息
    # print_statistics(full_scores, sim_scores, reused_scores, title)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    os.environ["VLLM_USE_MODELSCOPE"]="true"
    # 示例：处理多个数据文件，每个元素包含数据路径和标题
    # data_paths = [
    #     # ("examples/final/data/instruction_wildv2_similar_250331_answer_by_Qwen2.5-7B-Instruct_partial_output.json", 
    #     #  "Qwen2.5-7B"),
    #      ("examples/final/data/instruction_wildv2_similar_250331_answer_by_Meta-Llama-3.1-8B-Instruct_partial_output.json",
    #      "Llama3.1-8B"),
    #     ("examples/final/data/instruction_wildv2_similar_250331_answer_by_Qwen2.5-32B-Instruct-GPTQ-Int4_partial_output.json",
    #      "Qwen2.5-32B"),
       
    # ]
    # save_path = "examples/pipeline/images/bert_scores_comparison.png"
    # test_bert(data_paths, save_path) 
    # 处理第一个模型
    data_path = "examples/final/data/instruction_wildv2_similar_250331_answer_by_Meta-Llama-3.1-8B-Instruct_partial_output.json"
    # title = "Llama3.1-8B"
    save_path = "examples/pipeline/images/bert_scores_llama3.1_8b.png"
    test_bert(data_path, save_path)

    # 处理第二个模型
    data_path = "examples/final/data/instruction_wildv2_similar_250331_answer_by_Qwen2.5-32B-Instruct-GPTQ-Int4_partial_output.json"
    # title = "Qwen2.5-32B"
    save_path = "examples/pipeline/images/bert_scores_qwen2.5_32b.png"
    test_bert(data_path, save_path)