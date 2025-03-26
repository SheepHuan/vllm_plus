import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
import seaborn as sns

def setup_style(font_path="/root/code/vllm_plus/examples/dataset/data/fonts"):
    """设置绘图样式
    
    Args:
        font_path: 字体文件路径
    """
    # 加载字体
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    for file in font_files:
        font_manager.fontManager.addfont(file)
    
    # 设置绘图风格
    # plt.style.use('seaborn-v0_8')
    
    # 自定义样式设置
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.unicode_minus': False,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 14,
        'figure.dpi': 100,
        'axes.grid': False,
        'axes.spines.top': True,     # 显示上方轴线
        'axes.spines.right': True,   # 显示右侧轴线
    })

def plot_speed_comparison(
    token_lengths,
    embed_times,
    prefill_times,
    embed_models,
    prefill_models,
    save_path="examples/pipeline/images/speed.png",
    figsize=(10, 7)
):
    """绘制速度对比图
    
    Args:
        token_lengths: 输入token长度列表
        embed_times: 嵌入时间数据字典，key为模型名称，value为时间列表
        prefill_times: prefill时间数据字典
        embed_models: 嵌入模型名称列表
        prefill_models: prefill模型名称列表
        save_path: 图片保存路径
        figsize: 图像大小
    """
    # 创建图表
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    # 颜色设置
    embed_color = '#1f77b4'
    prefill_color = '#FF1493'
    markers = ['o', 's', '^', 'x', 'D', '*']
    
    lines = []
    
    # 绘制Embed时间
    for idx, model in enumerate(embed_models):
        line = ax1.plot(
            token_lengths, 
            embed_times[model],
            color=embed_color,
            linestyle='-',
            linewidth=2,
            marker=markers[idx],
            markersize=6,
            label=f"{model} Embed"
        )
        lines.extend(line)
    
    # 绘制Prefill时间
    for idx, model in enumerate(prefill_models):
        line = ax2.plot(
            token_lengths,
            prefill_times[model],
            color=prefill_color,
            linestyle='--',
            linewidth=2,
            marker=markers[idx+len(embed_models)],
            markersize=6,
            label=f"{model} Prefill"
        )
        lines.extend(line)
    
    # 设置坐标轴
    ax1.set_xlabel('Input Token Number')
    ax1.set_ylabel('Embed Time (ms)', color=embed_color)
    ax2.set_ylabel('Prefill Time (ms)', color=prefill_color)
    
    # 计算两个轴的数据范围
    embed_max = max(max(times) for times in embed_times.values())
    prefill_max = max(max(times) for times in prefill_times.values())
    embed_min = min(min(times) for times in embed_times.values())
    prefill_min = min(min(times) for times in prefill_times.values())
    
    # 设置对数刻度的范围
    y_min = min(embed_min, prefill_min)
    y_max = max(embed_max, prefill_max)
    
    # 创建对数刻度
    yticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    yticks = [y for y in yticks if y_min <= y <= y_max * 1.1]
    
    # 设置两个轴使用相同的对数刻度
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    
    # 设置相同的范围和刻度
    ax1.set_ylim(yticks[0], yticks[-1])
    ax2.set_ylim(yticks[0], yticks[-1])
    ax1.set_yticks(yticks)
    ax2.set_yticks(yticks)
    
    # 设置刻度标签格式
    ax1.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax2.yaxis.set_major_formatter(plt.ScalarFormatter())
    
    # 移除刻度标签的科学计数法
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.get_major_formatter().set_scientific(False)
    
    # 设置轴线颜色
    ax1.spines['left'].set_color(embed_color)
    ax2.spines['right'].set_color(prefill_color)
    
    # 添加图例
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines, 
        labels, 
        loc='upper left',
        framealpha=0.9,
        bbox_to_anchor=(0, 1.22),
        ncol=2
    )
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def draw():
    # 设置样式
    setup_style()
    
    # 数据定义
    token_lengths = [20, 120, 180, 400, 800, 1200, 1400, 1600]
    
    embed_times = {
        "Linq-Embed-Mistral-7B": [53.68, 85.77, 125.23, 262.12, 492.30, 686.43, 805.48, 946.00],
        "gte-Qwen2-1.5B": [27.26, 27.62, 29.98, 55.62, 98.34, 160.00, 160.00, 193.25],
        "bge-m3-0.5B": [15.27, 14.90, 15.42, 21.92, 42.70, 66.36, 73.97, 85.58],
        "all-MiniLM-L6-v2": [4.55, 4.71, 4.91, 5.28, 6.17, 6.81, 7.20, 7.73]
    }
    
    prefill_times = {
        "Qwen2.5-1.5B": [19.47, 19.34, 19.50, 19.72, 20.23, 21.80, 25.06, 28.88],
        "LLama3.1-8B": [25.07, 27.57, 29.54, 41.40, 70.11, 99.93, 110.11, 132.13]
    }
    
    # 模型名称列表
    embed_models = list(embed_times.keys())
    prefill_models = list(prefill_times.keys())
    
    # 绘制图表
    plot_speed_comparison(
        token_lengths,
        embed_times,
        prefill_times,
        embed_models,
        prefill_models,
        save_path="examples/pipeline/images/embed_speed_and_prefill_speed.png",
    )

if __name__ == "__main__":
    draw()