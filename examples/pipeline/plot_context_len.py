import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager 
# download the font files and save in this fold
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

data = [
    ("2023/07/18", "Llama2", 4, 'o'),
    ("2024/04/18", "Llama3", 8, 'o'),
    ("2024/12/06", "Llama3.3", 128, 'o'),
    ("2023/10/01", "Qwen", 8, 'D'),
    ("2024/06/07", "Qwen2", 128, 'D'),
    ("2024/12/19", "Qwen2.5", 1024, 'D'),
    ("2025/03/06", "QWQ", 128, 'D'),
    ("2025/01/20", "DeepSeek-V3", 128, 'x'),
    ("2024/05/01", "DeepSeek-V2", 128, 'x'),
    ("2023/03/14", "GPT-4", 128, "*"),
    ("2025/02/27", "GPT-4.5", 256, "*"),
    ("2025/03/12", "Gemma3", 128, '8'),
    ("2024/06/27", "Gemma2", 8, '8'),
    ("2024/06/05", "GLM4", 1024, '^'),
    ("2023/10/27", "ChatGLM3", 128, '^'),
    ("2025/02/24", "Claude3.7", 200, '.'),
    ("2023/03/14", "Claude", 9, '.'),
    ("2023/07/11", "Claude2", 100, '.'),
    ("2024/12/17", "Falcon 3", 32, '1'),
    ("2024/05/24", "Falcon 2", 8, '1'),
    ("2023/12/09", "Mixtral 8x7B", 32, 's'),
    ("2024/04/17", "Mixtral 8x22B", 64, 's'),
    ("2025/03/17", "Mistral Small 3.1", 128, 's'),

]


from datetime import datetime, timedelta

def draw():
    # 数据解析和转换
    dates = [datetime.strptime(item[0], "%Y/%m/%d") for item in data]
    models = [item[1] for item in data]
    context_len = [item[2] for item in data]

    # 创建画布
    plt.figure(figsize=(9, 8))
    
    # 使用深色系颜色方案，确保颜色足够区分
    colors = ['#1f77b4',  # 深蓝
             '#d62728',   # 深红
             '#2ca02c',   # 深绿
             '#9467bd',   # 深紫
             '#8c564b',   # 褐色
             '#e377c2',   # 深粉
             '#7f7f7f',   # 深灰
             '#bcbd22',   # 橄榄绿
             '#17becf',   # 青色
             '#ff7f0e',   # 橙色
             '#000000',
             '#ec6042',
             '#ecb542',
             '#e0ec42',
             '#87ec42',
             '#ec4287',
             '#42ec87',
             '#ec8742',
             '#4287ec',
             '#8742ec',
             '#ec4242',
             '#4242ec',
             '#ec4287',
             
             
             ]   # 黑色
    
    # 绘制散点图并创建图例，使用相同的样式
    for i, (date, model, cl,marker) in enumerate(data):
        plt.scatter(dates[i], cl, 
                   color=colors[i],     # 只改变颜色
                   alpha=0.9, 
                   edgecolors='white', 
                   s=100,
                   marker=marker,
                   label=model)

    # 添加所有水平虚线，包括8K
    plt.axhline(y=8, color='gray', linestyle='--', alpha=0.5, zorder=0)
    plt.axhline(y=32, color='gray', linestyle='--', alpha=0.5, zorder=0)
    plt.axhline(y=128, color='gray', linestyle='--', alpha=0.5, zorder=0)
    plt.axhline(y=1024, color='gray', linestyle='--', alpha=0.5, zorder=0)
    
    # 添加所有虚线标注，包括8K
    plt.text(dates[0], 8, '8K', 
             verticalalignment='bottom', 
             horizontalalignment='right',
             color='gray')
    plt.text(dates[0], 32, '32K', 
             verticalalignment='bottom', 
             horizontalalignment='right',
             color='gray')
    plt.text(dates[0], 128, '128K', 
             verticalalignment='bottom', 
             horizontalalignment='right',
             color='gray')
    plt.text(dates[0], 1024, '1024K', 
             verticalalignment='bottom', 
             horizontalalignment='right',
             color='gray')

    # 确保散点图在虚线上方
    plt.gca().set_axisbelow(True)

    # 设置对数坐标轴
    plt.yscale('log')
    plt.ylabel("Context Length (# K)", fontsize=12)
    
    # 配置时间坐标轴
    ax = plt.gca()
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
    
    plt.xlim(min(dates) - timedelta(days=30), 
             max(dates) + timedelta(days=30))
    
    plt.xlabel("Release Date", fontsize=14)
    plt.xticks(rotation=35, ha='right')

    # 添加标题和图例
    plt.legend(loc='upper center',
              bbox_to_anchor=(0.5, 1.15),
              fontsize=10,
              framealpha=0.9,
              edgecolor='none',
              ncol=6)

    plt.tight_layout()

    # 保存输出
    plt.savefig("examples/pipeline/images/context_len.png", 
               dpi=300, 
               bbox_inches='tight',
               facecolor='white')
    plt.savefig("examples/pipeline/images/context_len.pdf", 
               dpi=300, 
               bbox_inches='tight',
               facecolor='white')
    plt.close()

    


if __name__ == "__main__":
    draw()