import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 数据集名称
datasets = ['GSM8K', 'DROP']
# 评估指标类型
metrics = ['Accuracy', 'EM Score']
# 缓存方法
cache_methods = ['CacheBlend-prefill', 'CacheBlend-prefill+decode', 'KVShare-prefill (our)', 'KVShare-prefill+decode (our)', 'Naive', 'FR']
# 模型名称
models = ['Qwen2.5-7B', 'Llama-3.1-8B']

# 示例数据 - 使用更接近图片中的数据
# 格式: [x, y] 代表 [重计算比例(%), 评分]
data = {
    'Qwen2.5-7B': {
     'GSM8K': {
        'CacheBlend-prefill': [[10, 0.42], [20, 0.511], [30, 0.54], [40, 0.609]],
        'CacheBlend-prefill+decode': [[10, 0.722], [20, 0.718], [30, 0.7187], [40, 0.7265]],
        'KVShare-prefill (our)': [[10, 0.5], [20, 0.523], [30,0.554], [40, 0.6328]],
        'KVShare-prefill+decode (our)': [[10, 0.757], [20, 0.7187], [30,0.7148], [40, 0.7148]],
        'Naive': [[90, 0.5]],
        'FR': [[100, 0.828125]]
    },
    'DROP': {
        'CacheBlend-prefill': [[10, 0.367], [20, 0.3828], [30, 0.40625], [40, 0.3906]],
        'CacheBlend-prefill+decode': [[10, 0.41406], [20, 0.42968], [30, 0.4765], [40, 0.4609]],
        'KVShare-prefill (our)': [[10, 0.41406], [20, 0.45312], [30, 0.4375], [40, 0.4843]],
        'KVShare-prefill+decode (our)': [[10, 0.46875], [20, 0.5390], [30, 0.49218], [40, 0.5515]],
        'Naive': [[90, 0.3125]],
        'FR': [[100, 0.6875]]
    }
    }, 
    'Llama-3.1-8B': {
     'GSM8K': {
        'CacheBlend-prefill': [[10, 0.37], [20, 0.396], [30, 0.425], [40, 0.443]],
        'CacheBlend-prefill+decode': [[10, 0.507], [20, 0.511], [30, 0.466], [40, 0.5019]],
        'KVShare-prefill (our)': [[10, 0.335], [20, 0.357], [30,0.4023], [40, 0.4435]],
        'KVShare-prefill+decode (our)': [[10, 0.558], [20, 0.53515], [30,0.58203], [40, 0.57013]],
        'Naive': [[90, 0.337]],
        'FR': [[100, 0.54492]]
    },
    'DROP': {
        'CacheBlend-prefill': [[10, 0.453], [20, 0.5], [30, 0.578], [40, 0.5625]],
        'CacheBlend-prefill+decode': [[10, 0.468], [20, 0.546], [30, 0.5781], [40, 0.5703]],
        'KVShare-prefill (our)': [[10, 0.52], [20, 0.5625], [30,0.58593], [40, 0.6328]],
        'KVShare-prefill+decode (our)': [[10, 0.5468], [20, 0.625], [30,0.664], [40, 0.6875]],
        'Naive': [[90, 0.429]],
        'FR': [[100, 0.7265]]
    }
    }
}

# 颜色和标记样式 - 使用更明显的对比色
colors = ['#ff5252', '#ff8a80', '#4285F4', '#82b1ff', '#7B1FA2', '#FFA000']  # 红, 浅红, 蓝, 浅蓝, 紫, 橙
markers = ['s', 's', 'o', 'o', '^', 'D']  # 方形, 方形, 圆形, 圆形, 三角形, 菱形
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

# 创建子图布局 - 2x2布局
fig, axs = plt.subplots(2, 2, figsize=(13, 10), dpi=100)

# 在循环前加
ylabels = {'GSM8K': 'Accuracy', 'DROP': 'EM Score'}

# 为每个模型和数据集组合绘制子图
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        ax = axs[i, j]
        
        # 设置y轴范围
        if dataset == 'GSM8K':
            ax.set_ylim(0.3, 0.9)
        else:  # DROP
            ax.set_ylim(0.3, 0.8)
        
        # 设置x轴范围
        ax.set_xlim(0, 105)
        
        # 绘制数据点
        for k, method in enumerate(cache_methods):
            if method in data[model][dataset]:
                if method in ['CacheBlend-prefill', 'CacheBlend-prefill+decode', 'KVShare-prefill (our)', 'KVShare-prefill+decode (our)']:
                    # 对于有多个数据点的方法，需要连线
                    points = data[model][dataset][method]
                    x_values = [point[0] for point in points]
                    y_values = [point[1] for point in points]
                    ax.plot(x_values, y_values, color=colors[k], marker=markers[k], 
                           markersize=10, linewidth=line_width, alpha=0.9)
                    
                    # 添加数据点
                    for x, y in points:
                        ax.scatter(x, y, color=colors[k], marker=markers[k], s=marker_size,
                                  edgecolors='black', linewidth=1.5, zorder=10)
                else:
                    # 对于Naive和FR，只创建水平线
                    x, y = data[model][dataset][method][0]
                    # 创建从0到x的水平线
                    ax.plot([0, x], [y, y], color=colors[k], linewidth=line_width, 
                           alpha=0.9, linestyle='--')
        
        # 设置轴标签
        if j == 0:  # 第一列
            ax.set_ylabel(ylabels[dataset], fontsize=14, fontweight='bold')
        
        # 在每个子图底部添加x轴标签
        if i == 1:  # 最后一行
            ax.set_xlabel('Re-compute Ratio (%)', fontsize=14, fontweight='bold')
        
        # 设置刻度
        ax.xaxis.set_major_locator(plt.MultipleLocator(25))
        
        if dataset == 'GSM8K':
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        else:  # DROP
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        
        # 美化刻度标签
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1, length=3)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

# 设置列标题（模型名）
for j, model in enumerate(models):
    axs[0, j].set_title(model, fontsize=18, fontweight='bold', pad=15)

# 设置行标签（数据集名）
for i, dataset in enumerate(datasets):
    # 在每行最左侧居中加标签
    fig.text(0.01, 0.75-0.5*i, dataset, va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# 创建图例
handles = []
labels = []
for i, method in enumerate(cache_methods):
    if method in ['CacheBlend-prefill', 'CacheBlend-prefill+decode', 'KVShare-prefill (our)', 'KVShare-prefill+decode (our)']:
        handle = plt.Line2D([0], [0], color=colors[i], marker=markers[i], 
                           markersize=12, linewidth=line_width, 
                           markerfacecolor=colors[i], markeredgecolor='black')
    else:
        # Naive 和 FR 用虚线线条，不加marker
        handle = plt.Line2D([0], [0], color=colors[i], linestyle='--',
                            linewidth=line_width)
    handles.append(handle)
    labels.append(method)

# 将图例放在图的最上方
leg = fig.legend(
    handles=handles, labels=labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),  # 横向居中，纵向略高于图
    ncol=len(cache_methods),      # 横向一行展示
    fontsize=14, frameon=True, framealpha=0.95,
    handletextpad=0.5, columnspacing=1.5,
    fancybox=True, shadow=True
)
leg.get_frame().set_linewidth(2)

# 调整子图间距，给顶部留出空间
plt.subplots_adjust(wspace=0.25, hspace=0.25, top=0.88, bottom=0.10, left=0.08, right=0.95)

# 保存高分辨率图表
plt.savefig('acc_ratio.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
# 保存一个更高分辨率的版本，适合发表
plt.savefig('acc_ratio_high_res.png', dpi=600, bbox_inches='tight', pad_inches=0.2)
plt.savefig('acc_ratio.pdf', format='pdf', bbox_inches='tight', pad_inches=0.2)
plt.show()
