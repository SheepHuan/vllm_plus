import json
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rouge_by_window_size(input_path: str, save_path: str = "examples/pipeline/images/xsum_rouge_by_window.png", show_boxplot: bool = False):
    """绘制不同窗口大小下的ROUGE分数的CDF曲线对比，并添加full compute的CDF曲线
    
    Args:
        input_path: 输入数据文件路径
        save_path: 保存图片的路径
        show_boxplot: 是否显示箱线图，默认为False
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 收集不同窗口大小的ROUGE分数
    window_sizes = [6, 12, 24]  # 根据实际使用的窗口大小调整
    rouge_scores_by_window = {window: [] for window in window_sizes}
    
    # 收集full compute的ROUGE分数
    full_compute_scores = []
    
    # 用于存储每个样本在不同窗口大小下的分数
    sample_scores = []
    
    for item in data["similar_docs"]:
        try:
            # 收集full compute分数
            if "rouge" in item:
                full_compute_scores.append(item["rouge"]["rougeL"])
            
            # 收集当前样本在不同窗口大小下的分数
            current_sample_scores = []
            valid = True
            
            for window_size in window_sizes:
                if f"rouge_w{window_size}" in item["reused_top1_w25"]:
                    score = item["reused_top1_w25"][f"rouge_w{window_size}"]["rougeL"]
                    current_sample_scores.append(score)
                else:
                    valid = False
                    break
            
            # 只有当所有窗口大小都有分数时才处理
            if valid and len(current_sample_scores) == len(window_sizes):
                # 检查所有分数是否相同
                if not all(x == current_sample_scores[0] for x in current_sample_scores):
                    sample_scores.append(current_sample_scores)
                    for window_size, score in zip(window_sizes, current_sample_scores):
                        rouge_scores_by_window[window_size].append(score)
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    print(f"过滤后剩余样本数量: {len(sample_scores)}")
    
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
        print(f"  平均ROUGE-1: {mean_value:.4f} ± {std_value:.4f}")
        print(f"  中位数: {np.median(full_compute_scores):.4f}")
        print(f"  范围: [{np.min(full_compute_scores):.4f}, {np.max(full_compute_scores):.4f}]")
    
    # 绘制CDF曲线
    for window_size in window_sizes:
        scores = rouge_scores_by_window[window_size]
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
        print(f"  平均ROUGE-1: {mean_value:.4f} ± {std_value:.4f}")
        print(f"  中位数: {median_value:.4f}")
        print(f"  范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # 设置第一个子图（CDF）的属性
    ax1.set_xlabel('ROUGE-1 Score')
    ax1.set_ylabel('Cumulative Probability')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # 如果显示箱线图，设置第二个子图的属性
    if show_boxplot:
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('ROUGE-1 Score')
        ax2.set_title('ROUGE-1 Score Distribution by Window Size')
        ax2.set_xticks([0] + window_sizes)
        ax2.set_xticklabels(['Full'] + [f'w={w}' for w in window_sizes])
        ax2.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"图表已保存至: {save_path}")

if __name__ == "__main__":
    input_path = "examples/dataset/data/xsum/xsum_dataset_similar_docs_top50_250403_windows_outputs-qwen2.5-32b.json"
    save_path = "examples/pipeline/images/xsum_rouge_by_window.png"
    plot_rouge_by_window_size(input_path, save_path)