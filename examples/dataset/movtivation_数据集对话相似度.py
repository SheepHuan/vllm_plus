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

def dataset_conversation_sim():
    
    x = ["WildChat-1M","ShareGPT-90K","LMSYSChat-1M"]
    total = 20000
    wildchat_data = np.array( [7391, 4357, 2899, 1764]) / total * 100
    sharegpt90k_data = np.array( [7262, 1088, 374, 114]) / total * 100
    lmsys_data = np.array( [4059, 550, 180, 73]) / total * 100
    # 可视化横向柱状图，左右分布
    bar_width = 0.2  # 柱状图的宽度
    index = np.arange(len(x))  # x轴的位置

    plt.figure(figsize=(8, 6))  # 设置图像的宽度和高度
    # 绘制每个数据集的4个柱状图
    for i in range(4):  # 4个值
        plt.bar(index + i * bar_width, [wildchat_data[i], sharegpt90k_data[i], lmsys_data[i]], 
                width=bar_width, label=f"Each prompt has {i+1} similar { 'prompt' if i == 0 else 'prompts'}.")  # 只在第一个柱状图上显示标签
    plt.yticks(fontsize=12) 
    plt.xticks(index + bar_width, x, fontsize=14)  # 设置x轴刻度和字体大小
    plt.ylabel('Proportion (%)', fontsize=14)  # 设置y轴标签和字体大小
    plt.xlabel('Real dataset of user conversation with LLM', fontsize=14)  # 设置x轴标签和字体大小
    # plt.title('The proportion of similar conversations in the dataset')  # 添加标题
    plt.legend(fontsize=12)  # 设置图例的字体大小
    
    # 导出为PDF
    plt.savefig('examples/dataset/data/dataset_conversation_sim.pdf', format='pdf')
    # plt.show()
    
dataset_conversation_sim()