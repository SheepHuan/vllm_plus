import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager 

# 设置字体
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
font_files = font_manager.findSystemFonts(fontpaths=font_path)
for file in font_files:
    font_manager.fontManager.addfont(file)

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_acc_cdf(input_path: str, save_path: str = "examples/pipeline/images/acc_cdf_comparison.png"):
    """绘制不同窗口大小下的准确率CDF曲线
    Args:
        input_path: 输入数据文件路径
        save_path: 保存图片的路径
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 定义窗口大小和对应的颜色
    window_sizes = [6, 12, 124]
    colors = {
        6: 'blue',    # 蓝色
        12: 'green',  # 绿色
        124: 'red'    # 红色
    }
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 收集每个窗口大小的准确率数据
    acc_data = {w: [] for w in window_sizes}
    
    for item in data:
        try:
            for window_size in window_sizes:
                if f"acc_w{window_size}" in item:
                    acc_data[window_size].append(float(item[f"acc_w{window_size}"]))
        except Exception as e:
            continue
    
    # 绘制CDF曲线
    for window_size in window_sizes:
        scores = acc_data[window_size]
        if len(scores) == 0:
            print(f"窗口大小 {window_size} 没有数据")
            continue
            
        # 计算CDF
        sorted_scores = np.sort(scores)
        p = np.arange(1, len(scores) + 1) / len(scores)
        
        # 绘制CDF曲线
        plt.plot(sorted_scores, p, 
                label=f'Window Size {window_size}', 
                color=colors[window_size], 
                alpha=0.7)
        
        # 计算并打印统计信息
        mean_value = np.mean(scores)
        median_value = np.median(scores)
        print(f"窗口大小 {window_size}:")
        print(f"  样本数量: {len(scores)}")
        print(f"  平均准确率: {mean_value:.4f} ± {np.std(scores):.4f}")
        print(f"  中位数: {median_value:.4f}")
        print(f"  范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # 添加理想参考线
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal (Accuracy=1.0)')
    
    # 设置图表属性
    plt.xlabel('Accuracy')
    plt.ylabel('Cumulative Probability')
    plt.title('Accuracy CDF by Window Size')
    plt.grid(True, alpha=0.3)
    
    # 将图例放在图表右上角
    plt.legend(loc='lower right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"图表已保存至: {save_path}")

if __name__ == "__main__":
    # 示例使用
    input_path = "examples/pipeline/data/acc_data.json"  # 请替换为实际的数据文件路径
    save_path = "examples/pipeline/images/acc_cdf_comparison.png"
    plot_acc_cdf(input_path, save_path) 