import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 数据结构：每个模型的数据
# x轴: TTFT (s)
# y轴: F1-score 或 Rouge-L-score
# 格式: [model_name][dataset][metric_type][data_points]
# 你需要提供实际数据

# 示例数据结构 - 请替换为你的实际数据
models = ['Qwen2.5-72B', 'Llama3.1-8B']
datasets = ['GSM8K', 'SAMSum',]
metrics = ['acc', 'rougeL']

# 每个模型、数据集和缓存方法的数据点
# 格式: [x, y] 代表 [TTFT(s), 分数]
data = {
    'Qwen2.5-72B': {
        'GSM8K': {
            'CacheBlend': [0.1487, 0.7734],
            'KVShare': [0.156, 0.7978],
            'Naive': [0.12, 0.4609],
            'Full KV recompute': [0.2938, 0.8593]
        },
        'SAMSum': {
            'CacheBlend': [0.1487, 0.249],
            'KVShare': [0.156, 0.253],
            'Naive': [0.12, 0.208],
            'Full KV recompute': [0.2938, 0.267]
        },
    },
    'Llama3.1-8B': {
        'GSM8K': {
            'CacheBlend': [0.1487, 0.703],
            'KVShare': [0.156, 0.774],
            'Naive': [0.12, 0.1398],
            'Full KV recompute': [0.2938, 0.8359]
        },
        'SAMSum': {
            'CacheBlend': [0.1487, 0.249],
            'KVShare': [0.156, 0.253],
            'Naive': [0.12, 0.208],
            'Full KV recompute': [0.2938, 0.267]
        },
    }
}

# 绘图设置
cache_methods = ['CacheBlend', 'KVShare', 'Naive', 'Full KV recompute']
colors = ['red', 'orange', 'deepskyblue', 'navy']  # 增加对比度
markers = ['s', '*', 'o', '^']
marker_size = 140  # 进一步增大数据点大小

# 设置绘图样式和参数
plt.style.use('seaborn-v0_8-whitegrid')  # 使用带网格的样式
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# 创建子图 - 设置更合理的比例
fig, axs = plt.subplots(len(datasets), len(models), figsize=(len(models)*6, len(datasets)*6), 
                       constrained_layout=False, sharex=True)

# 处理单行或单列的情况
if len(datasets) == 1 and len(models) == 1:
    axs = np.array([[axs]])
elif len(datasets) == 1:
    axs = np.array([axs])
elif len(models) == 1:
    axs = np.array([[ax] for ax in axs])

# 为每个子图设置固定宽高比和一致的刻度范围
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        ax = axs[i, j]
        
        # 设置固定的宽高比
        if dataset == 'GSM8K':
            ax.set_ylim(0.1, 0.9)  # GSM8K 数据集准确率范围
            ax.set_xlim(0.08, 0.32)  # 调整x轴范围，更好地展示数据点
        else:
            ax.set_ylim(0.20, 0.30)  # SAMSum 数据集 Rouge-L 分数范围
            ax.set_xlim(0.08, 0.32)  # 调整x轴范围，更好地展示数据点
        
        # 绘制每个缓存方法的数据点
        for k, method in enumerate(cache_methods):
            try:
                if method in data[model][dataset]:
                    x, y = data[model][dataset][method]
                    ax.scatter(x, y, color=colors[k % len(colors)], 
                             marker=markers[k % len(markers)], s=marker_size, 
                             edgecolors='black', linewidth=1.5, alpha=1.0, zorder=10)  # 增加透明度为1
            except (KeyError, IndexError):
                # 如果数据不存在，则跳过
                continue
        
        # 设置轴标签和标题
        if i == len(datasets) - 1:  # 只在底部行添加 x 轴标签
            ax.set_xlabel('TTFT (s)')
        if j == 0:  # 只在左侧列添加 y 轴标签
            try:
                ax.set_ylabel(metrics[i])
            except IndexError:
                ax.set_ylabel('Score')
        
        # 设置每个子图的标题
        if i == 0:  # 只在顶部行添加模型名称
            ax.set_title(model, fontsize=18, fontweight='bold')
        
        # 添加数据集标签在左边
        if j == 0:
            ax.text(-0.4, 0.5, f"{dataset}\nDataset", transform=ax.transAxes, 
                    va='center', ha='center', rotation=90, fontsize=16, fontweight='bold')
        
        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # 设置轴刻度字体大小和粗细
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
        
        # 设置更精确的刻度
        if dataset == 'GSM8K':
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))  # 每0.2一个刻度
        else:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))  # 每0.02一个刻度
            
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))  # 每0.05一个刻度
        
        # 设置刻度格式
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        if dataset == 'GSM8K':
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        else:
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 创建图例
handles = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', 
                     markerfacecolor=colors[i % len(colors)], markersize=14) 
           for i, method in enumerate(cache_methods)]

# 将图例放在图表顶部中央
legend = fig.legend(handles=handles, labels=cache_methods, loc='upper center', 
           bbox_to_anchor=(0.5, 0.99), ncol=min(4, len(cache_methods)),
           fontsize=14, frameon=True, framealpha=0.9, 
           handletextpad=0.5, columnspacing=1.0)

# 为图例添加边框
legend.get_frame().set_linewidth(1.5)
legend.get_frame().set_edgecolor('black')

# 添加总标题
# fig.suptitle('TTFT vs 准确率比较', fontsize=20, y=1.05)

# 调整子图布局 - 子图之间添加更多空间
plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.95)

# 保存并显示图表
plt.savefig('ttft_vs_accuracy.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.show()

# def plot_with_custom_data(custom_data=None, models=None, datasets=None, metrics=None, 
#                          cache_methods=None, output_file='ttft_vs_accuracy.png'):
#     """
#     使用用户提供的自定义数据绘制图表
    
#     参数:
#     custom_data: 字典格式的数据，结构与上面的 'data' 变量相同
#     models: 模型名称列表，如果为None则使用默认值
#     datasets: 数据集名称列表，如果为None则使用默认值
#     metrics: 评估指标列表，如果为None则使用默认值
#     cache_methods: 缓存方法列表，如果为None则使用默认值
#     output_file: 输出图片文件名
#     """
#     # 使用提供的数据或默认数据
#     plot_data = custom_data if custom_data is not None else data
#     plot_models = models if models is not None else ['Mistral-7B', 'Yi-34B', 'Llama-70B']
#     plot_datasets = datasets if datasets is not None else ['2WikiMOA', 'Musique', 'SAMSum', 'MultiNews']
#     plot_metrics = metrics if metrics is not None else ['F1-score', 'F1-score', 'Rouge-L-score', 'Rouge-L-score']
#     plot_cache_methods = cache_methods if cache_methods is not None else ['CacheBlend', 'Full KV reuse', 'Prefix Caching', 'Full KV recompute']
    
#     # 颜色和标记
#     colors = ['red', 'orange', 'skyblue', 'navy']
#     markers = ['s', 'x', 'o', '^']
    
#     # 创建子图
#     fig, axs = plt.subplots(len(plot_datasets), len(plot_models), 
#                            figsize=(len(plot_models)*4, len(plot_datasets)*3.5), 
#                            constrained_layout=True)
    
#     # 处理单行或单列的情况
#     if len(plot_datasets) == 1 and len(plot_models) == 1:
#         axs = np.array([[axs]])
#     elif len(plot_datasets) == 1:
#         axs = np.array([axs])
#     elif len(plot_models) == 1:
#         axs = np.array([[ax] for ax in axs])
    
#     # 迭代所有子图
#     for i, dataset in enumerate(plot_datasets):
#         for j, model in enumerate(plot_models):
#             ax = axs[i, j]
            
#             # 绘制每个缓存方法的数据点
#             for k, method in enumerate(plot_cache_methods):
#                 try:
#                     if method in plot_data[model][dataset]:
#                         x, y = plot_data[model][dataset][method]
#                         ax.scatter(x, y, color=colors[k % len(colors)], 
#                                  marker=markers[k % len(markers)], s=80)
#                 except (KeyError, IndexError):
#                     # 如果数据不存在，则跳过
#                     continue
            
#             # 设置轴标签和标题
#             if i == len(plot_datasets) - 1:  # 只在底部行添加 x 轴标签
#                 ax.set_xlabel('TTFT (s)')
#             if j == 0:  # 只在左侧列添加 y 轴标签
#                 try:
#                     ax.set_ylabel(plot_metrics[i])
#                 except IndexError:
#                     ax.set_ylabel('Score')
            
#             # 设置每个子图的标题
#             if i == 0:  # 只在顶部行添加模型名称
#                 ax.set_title(model)
            
#             # 添加数据集标签在左边
#             if j == 0:
#                 ax.text(-0.3, 0.5, f"{dataset}\nDataset", transform=ax.transAxes, 
#                         va='center', ha='center', rotation=90)
            
#             # 添加 "Better" 箭头 - 从右下指向左上
#             ax.annotate("Better", xy=(0.3, 0.7), xytext=(0.6, 0.3),
#                         arrowprops=dict(facecolor='black', shrink=0.05, width=2),
#                         transform=ax.transAxes, fontsize=12)
            
#             # 设置 y 轴范围为 0 到 0.4
#             ax.set_ylim(0, 0.4)
            
#             # 设置 x 轴范围
#             if model == 'Mistral-7B':
#                 ax.set_xlim(0, 0.8)
#             elif model == 'Yi-34B':
#                 ax.set_xlim(0, 2.5)
#             elif model == 'Llama-70B':
#                 ax.set_xlim(0, 3.0)
#             else:
#                 # 对于其他模型，根据数据自动设置范围
#                 all_x = []
#                 for method in plot_cache_methods:
#                     try:
#                         if method in plot_data[model][dataset]:
#                             all_x.append(plot_data[model][dataset][method][0])
#                     except (KeyError, IndexError):
#                         continue
                
#                 if all_x:
#                     max_x = max(all_x)
#                     ax.set_xlim(0, max_x * 1.2)  # 设置一个稍大的范围
    
#     # 创建图例
#     handles = [plt.Line2D([0], [0], marker=markers[i % len(markers)], color='w', 
#                          markerfacecolor=colors[i % len(colors)], markersize=10) 
#                for i, method in enumerate(plot_cache_methods)]
    
#     fig.legend(handles=handles, labels=plot_cache_methods, loc='upper center', 
#                bbox_to_anchor=(0.5, 0.98), ncol=min(4, len(plot_cache_methods)))
    
#     # 调整子图布局
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为图例留出空间
    
#     # 保存并显示图表
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     plt.show()
    
#     return fig

# # 如果直接运行脚本
# if __name__ == "__main__":
#     import json
#     import sys
    
#     # if len(sys.argv) > 1:
#     #     # 如果提供了数据文件路径
#     #     try:
#     #         with open(sys.argv[1], 'r') as f:
#     #             custom_data = json.load(f)
#     #         output_file = sys.argv[2] if len(sys.argv) > 2 else 'ttft_vs_accuracy.png'
#     #         plot_with_custom_data(custom_data=custom_data, output_file=output_file)
#     #     except Exception as e:
#     #         print(f"加载数据文件时出错: {e}")
#     #         print("使用示例数据绘图...")
#     #         plot_with_custom_data()
#     # else:
#     print("未提供数据文件，使用示例数据绘图...")
#     # 使用脚本中的示例数据
#     plot_with_custom_data()
