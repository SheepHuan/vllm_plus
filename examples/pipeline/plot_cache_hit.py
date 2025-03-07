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
    
    x = ["WildChat-Similar","ShareGPT-Similar","LMSYSChat-Similar"]
    wildchat_data = np.array( [95.64, 94.26, 91.57, 85.90, 64.72])
    sharegpt90k_data = np.array( [92.08, 89.58, 86.22, 77.37,52.30])
    lmsys_data = np.array( [90.83, 79.94, 72.22, 51.85, 20.36])
    rate_label =[">10%",">30%",">50%",">70%",">90%"]
    # 可视化横向柱状图，左右分布
    bar_width = 0.15  # 柱状图的宽度
    index = np.arange(len(x))  # x轴的位置

    plt.figure(figsize=(8, 6))  # 设置图像的宽度和高度
    # 绘制每个数据集的4个柱状图
    for i in range(0,5):  # 4个值
        bar_index = i - 1
        plt.bar(index + bar_index * bar_width, [wildchat_data[i], sharegpt90k_data[i], lmsys_data[i]], 
                width=bar_width, label=f"requests with KVcache hit rate {rate_label[i]}.")  # 只在第一个柱状图上显示标签
    plt.yticks(fontsize=14) 
    plt.xticks(index + bar_width, x, fontsize=14)  # 设置x轴刻度和字体大小
    plt.ylabel('Proportion (%)', fontsize=14)  # 设置y轴标签和字体大小
    # plt.xlabel('Real dataset of user conversation with LLM', fontsize=14)  # 设置x轴标签和字体大小
    # plt.title('The proportion of similar conversations in the dataset')  # 添加标题
    # plt.legend()  # 设置图例的字体大小
    # ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.102), loc='lower left',
    #       ncol=3, mode="expand", borderaxespad=0)
    
    
    plt.legend(bbox_to_anchor=(0, 1.07, 1, 0.102),ncol=2, loc='upper left',mode="expand", borderaxespad=0, fontsize=12)

    # 调整图形布局，防止图例被裁剪
    plt.tight_layout()
    # 导出为PDF
    plt.savefig('examples/dataset/data/dataset_kvcache_hit_rate.pdf', format='pdf')
    # plt.show()
    
dataset_conversation_sim()