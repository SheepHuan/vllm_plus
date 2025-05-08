import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 数据集名称
datasets = ['GSM8K', 'SAMSum']
# 评估指标类型
metrics = ['acc', 'rougeL']
# 缓存方法
cache_methods = ['CacheBlend', 'KVShare (our)', 'Naive', 'FULL COMPUTE']

# 示例数据 - 使用更接近图片中的数据
# 格式: [x, y] 代表 [重计算比例(%), 评分]
data = {
    'GSM8K': {
        'CacheBlend': [[10, 0.55468], [20, 0.6953], [30, 0.7734], [40, 0.8203125],[50,0.859375]],
        'KVShare (our)': [[10, 0.734375], [20, 0.7609375], [30,0.7978], [40, 0.8223],[50,0.8578]],
        'Naive': [[90, 0.4609375]],
        'FULL COMPUTE': [[100, 0.859375]]
    },
    'SAMSum': {
        'CacheBlend': [[10, 0.2445], [20, 0.2563], [30, 0.2663]],
        'KVShare (our)': [[10, 0.2471], [20, 0.2596], [30, 0.2701]],
        'Naive': [[90, 0.2407]],
        'FULL COMPUTE': [[100, 0.2740]]
    }
}

# 颜色和标记样式 - 使用更明显的对比色
colors = ['#ff5252', '#4285F4', '#34A853', '#7B1FA2']  # 红, 蓝, 绿, 紫
markers = ['s', 'o', 'x', '^']
marker_size = 150  # 更大的标记尺寸
line_width = 2.5   # 加粗的线条宽度

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

# 创建子图布局 - 只有两个子图
fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
axs = axs.flatten()

# 为每个数据集绘制子图
for i, dataset in enumerate(datasets):
    ax = axs[i]
    
    # 设置y轴范围
    if dataset == 'GSM8K':
        ax.set_ylim(0.4, 0.9)  # 根据新数据调整范围
        metric_name = 'acc'
    else:  # SAMSum
        ax.set_ylim(0.23, 0.28)  # 根据新数据调整范围
        metric_name = 'Rouge-L-Score'
    
    # 设置x轴范围
    ax.set_xlim(0, 105)  # 调整X轴范围
    
    # 绘制数据点
    for j, method in enumerate(cache_methods):
        if method in data[dataset]:
            if method in ['CacheBlend', 'KVShare (our)']:
                # 对于有多个数据点的方法，需要连线
                points = data[dataset][method]
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                ax.plot(x_values, y_values, color=colors[j], marker=markers[j], 
                       markersize=10, linewidth=line_width, alpha=0.9)
                
                # 添加数据点
                for x, y in points:
                    ax.scatter(x, y, color=colors[j], marker=markers[j], s=marker_size,
                              edgecolors='black', linewidth=1.5, zorder=10)
            else:
                # 对于其他方法，只有单个数据点
                x, y = data[dataset][method][0]
                ax.scatter(x, y, color=colors[j], marker=markers[j], s=marker_size,
                          edgecolors='black', linewidth=1.5, zorder=10)
    
    # 设置标题
    ax.set_title(dataset, fontsize=20, fontweight='bold', pad=15)
    
    # 设置轴标签
    if i == 0:
        if dataset == 'GSM8K':
            ax.set_ylabel('acc', fontsize=18, fontweight='bold')
        else:
            ax.set_ylabel(metric_name, fontsize=18, fontweight='bold')
    
    # 在每个子图底部添加x轴标签
    ax.set_xlabel('Re-compute Ratio (%)', fontsize=16, fontweight='bold')
    
    # 设置刻度
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))  # 调整为每25个单位一个刻度
    
    if dataset == 'GSM8K':
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    else:  # SAMSum
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax.set_yticks([0.23, 0.24, 0.25, 0.26, 0.27, 0.28])
    
    # 美化刻度标签
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 增加边框粗细

# 创建图例
handles = []
labels = []
for i, method in enumerate(cache_methods):
    if method in ['CacheBlend', 'KVShare (our)']:
        handle = plt.Line2D([0], [0], color=colors[i], marker=markers[i], 
                           markersize=12, linewidth=line_width, 
                           markerfacecolor=colors[i], markeredgecolor='black')
    else:
        handle = plt.Line2D([0], [0], marker=markers[i], color='w',
                           markerfacecolor=colors[i], markersize=12, 
                           markeredgecolor='black', markeredgewidth=1.5)
    handles.append(handle)
    labels.append(method)

# 将图例放在图表顶部
leg = fig.legend(handles=handles, labels=labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.15), ncol=len(cache_methods),
           fontsize=14, frameon=True, framealpha=0.95,
           handletextpad=0.5, columnspacing=1.5, 
           fancybox=True, shadow=True)

# 美化图例
leg.get_frame().set_linewidth(2)

# 调整子图间距
plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.80, bottom=0.15, left=0.08, right=0.95)

# 保存高分辨率图表
plt.savefig('acc_ratio.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
# 保存一个更高分辨率的版本，适合发表
plt.savefig('acc_ratio_high_res.png', dpi=600, bbox_inches='tight', pad_inches=0.2)
plt.savefig('acc_ratio.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
