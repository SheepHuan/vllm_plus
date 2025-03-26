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
        "ShareGPT-90k",
        "LMSYS-1M",
        "WildChat-1M",
        "MOSS",
        "BELLE-0.8M",
        "InstructionWild v2"
    ]
    total_instructions = [
        364544,
        765472,
        1146111,
       
        6234056,
        2244995,
        110904
    ]
    
    duplicate_instructions = [
        160447,
        41644,
        462105,
         2753298,
        1384085,
        21458
    ]
    
    # 创建图表
    plt.figure(figsize=(8, 6))
    
    # 设置柱子的位置
    x = np.arange(len(embed_model_name))
    width = 0.35  # 柱子的宽度
    
    # 创建两组柱子
    bars1 = plt.bar(x - width/2, total_instructions, width, 
                   label='Total Instructions',
                   color='#1f77b4',
                   hatch='/')  # 添加斜线填充
    
    bars2 = plt.bar(x + width/2, duplicate_instructions, width,
                   label='High-Similar Instructions',
                   color='#2ca02c',
                   hatch='\\')  # 添加反斜线填充
    
    # 设置对数刻度
    plt.yscale('log')
    
    # 设置图表属性
    plt.xlabel('Real-World LLM Conversation Dataset', fontsize=14)
    plt.ylabel('Instruction Count', fontsize=14)
    # plt.title('Instructions Distribution', fontsize=14)
    
    # 设置x轴刻度
    plt.xticks(x, embed_model_name, rotation=10, ha='right',fontsize=12)
    # 设置Y轴刻度字体大小
    plt.yticks(fontsize=12)
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加图例
    plt.legend(loc='upper left', fontsize=12)
    
    # 在每个柱子上添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("examples/pipeline/images/instructions_distribution.png", 
                dpi=300, bbox_inches='tight')
    plt.savefig("examples/pipeline/images/instructions_distribution.pdf", 
                dpi=300, bbox_inches='tight')

draw()