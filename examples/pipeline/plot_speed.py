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

def draw():
    # 模型名称和参数
    embed_model_name = [
        "Linq-Embed-Mistral-7B Embed",
        "gte-Qwen2-1.5B Embed",
        "bge-m3-0.5B Embed",
        "all-MiniLM-L6-v2 Embed"
    ]
    
    prefill_model = [
        "Qwen2.5-1.5B",
        "LLama3.1-8B"
    ]
    
    # Embed耗时数据（左侧Y轴）
    x = [20,120,180,400,800,1200,1400,1600]
    # y1 = [53.68,85.77,125.23,262.12,492.30,686.43,805.48,946.00]  # 7B模型
    y2 = [27.26,27.62,29.98,55.62,98.34,160.00,160.00,193.25]    # 1.5B模型
    y3 = [15.27,14.90,15.42,21.92,42.70,66.36,73.97,85.58]       # 0.5B模型
    y4 = [4.55,4.71,4.91,5.28,6.17,6.81,7.20,7.73]
    # Prefill耗时数据（右侧Y轴，示例数据需替换实际值）
    z1 = [19.47,19.34,19.50,19.72,20.23,21.80,25.06,28.88]  # 7B模型prefill
    z2 = [25.07,27.57,29.54,41.40,70.11,99.93,110.11,132.13]     # 1.5B模型prefill

    # 初始化画布和双轴
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴[1,5](@ref)

    # 绘制左侧Y轴数据（Embed耗时）
    embed_color = '#1f77b4'
    # line11, = ax1.plot(x, y1, color=embed_color, linestyle='-', linewidth=2, 
    #                  marker='o', markersize=6, label=embed_model_name[0])
    line12, = ax1.plot(x, y2, color=embed_color, linestyle='-', linewidth=2,
                     marker='s', markersize=6, label=embed_model_name[1])
    line13, = ax1.plot(x, y3, color=embed_color, linestyle='-', linewidth=2,
                     marker='^', markersize=6, label=embed_model_name[2])
    line14, = ax1.plot(x, y4, color=embed_color, linestyle='-', linewidth=2,
                     marker='x', markersize=6, label=embed_model_name[3])
    # 绘制右侧Y轴数据（Prefill耗时）
    prefill_color = '#FF1493'  # 洋红色系
    line21, = ax2.plot(x, z1, color=prefill_color, linestyle='--',
                      linewidth=2, marker='D', markersize=5, 
                      label=f"{prefill_model[0]} Prefill")
    line22, = ax2.plot(x, z2, color=prefill_color, linestyle='--',
                      linewidth=2, marker='*', markersize=7, 
                      label=f"{prefill_model[1]} Prefill")

    # 坐标轴设置
    ax1.set_xlabel('Input Token Number', fontsize=14)
    ax1.set_ylabel('Embed Time (ms)', fontsize=14, color='#1f77b4')
    ax2.set_ylabel('Prefill Time (ms)', fontsize=14, color=prefill_color)
    ax1.tick_params(axis='both', which='major', labelsize=12)  # X轴和左Y轴
    ax2.tick_params(axis='y', labelsize=12)  # 右Y轴
    
    ax1.spines['left'].set_color('#1f77b4')      # 左侧轴线颜色[4](@ref)
    ax2.spines['right'].set_color(prefill_color) # 右侧轴线颜色

    # 合并图例（关键步骤）
    lines = [ line12, line13, line14, line21,line22]
   
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9,
              bbox_to_anchor=(0, 1.22), ncol=2,fontsize=14)  # 调整图例位置[7,10](@ref)


    embed_max = max(y2)
    prefill_max = max(z2)
    ratio = embed_max / prefill_max

    # 设置右侧Y轴显示范围与左侧对齐
    ax1.set_yticks([0,5,10,20,30,50,100,200])
    ax2.set_yticks([0,5,10,20,30,50,100,200])
    # 辅助元素
    # ax1.grid(True, linestyle='--', alpha=0.6)  # 网格线保留[4](@ref)
    
    # 保存输出
    plt.savefig("examples/pipeline/images/speed.png", bbox_inches='tight')
draw()