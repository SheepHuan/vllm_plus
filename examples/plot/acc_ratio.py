import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 数据集名称
datasets = ['GSM8K', 'DROP', 'SAMSum']
# 评估指标类型
metrics = ['Accuracy', 'EM Score', 'F1 Score']
# 缓存方法
cache_methods = ['CacheBlend-prefill', 'CacheBlend-prefill+decode', 'KVShare-prefill (our)', 'KVShare-prefill+decode (our)', 'EPIC-prefill', 'EPIC-prefill+decode', 'Naive', 'FR']
# 模型名称
models = ['Qwen2.5-7B', 'Llama-3.1-8B', 'Yi-1.5-9B']

# 示例数据 - 使用更接近图片中的数据
# 格式: [x, y] 代表 [重计算比例(%), 评分]
data = {
    'Qwen2.5-7B': {
     'GSM8K': {
        'CacheBlend-prefill': [[10, 0.42], [20, 0.511], [30, 0.54], [40, 0.609]],
        'CacheBlend-prefill+decode': [[10, 0.722], [20, 0.718], [30, 0.7187], [40, 0.7265]],
        'KVShare-prefill (our)': [[10, 0.5], [20, 0.523], [30,0.554], [40, 0.6328]],
        'KVShare-prefill+decode (our)': [[10, 0.757], [20, 0.7187], [30,0.7148], [40, 0.7148]],
        'EPIC-prefill': [[10,0.33984], [20, 0.339843], [30,0.326171], [40, 0.33398]],
        'EPIC-prefill+decode': [[10, 0.50785], [20, 0.470703], [30,0.470703], [40, 0.46093]],
        'Naive': [[10, 0.5]],
        'FR': [[100, 0.828125]]
    },
    'DROP': {
        'CacheBlend-prefill': [[10, 0.367], [20, 0.3828], [30, 0.40625], [40, 0.3906]],
        'CacheBlend-prefill+decode': [[10, 0.41406], [20, 0.42968], [30, 0.4765], [40, 0.4609]],
        'KVShare-prefill (our)': [[10, 0.41406], [20, 0.45312], [30, 0.4375], [40, 0.4843]],
        'KVShare-prefill+decode (our)': [[10, 0.46875], [20, 0.5390], [30, 0.49218], [40, 0.5515]],
        'EPIC-prefill': [[10,0.453125], [20,0.46093], [30,0.45312], [40, 0.5]],
        'EPIC-prefill+decode': [[10, 0.546875], [20, 0.453125], [30,0.5390], [40, 0.55468]],
        'Naive': [[10, 0.3125]],
        'FR': [[100, 0.6875]]
    },
    'SAMSum': {
        'CacheBlend-prefill': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
        'CacheBlend-prefill+decode': [[10, 0.147413], [20,0.147942], [30,0.15922], [40, 0.16200]],
        'KVShare-prefill (our)': [[10, 0.157595], [20, 0.1613373], [30, 0.167983], [40, 0.17214]],
        'KVShare-prefill+decode (our)': [[10, 0.17187], [20, 0.178930], [30, 0.181036], [40, 0.190586]],
        'EPIC-prefill': [[10,0.14290], [20, 0.14547], [30,0.14708090], [40, 0.1478397]],
        'EPIC-prefill+decode': [[10, 0.15166], [20,0.1564341], [30,0.1562389], [40,0.157159]],
        'Naive': [[10, 0.08397565]],
        'FR': [[100, 0.201769]]
    },
    }, 
    'Llama-3.1-8B': {
     'GSM8K': {
        'CacheBlend-prefill': [[10, 0.37], [20, 0.396], [30, 0.425], [40, 0.443]],
        'CacheBlend-prefill+decode': [[10, 0.507], [20, 0.511], [30, 0.466], [40, 0.5019]],
        'KVShare-prefill (our)': [[10, 0.335], [20, 0.357], [30,0.4023], [40, 0.4435]],
        'KVShare-prefill+decode (our)': [[10, 0.558], [20, 0.53515], [30,0.58203], [40, 0.57013]],
        'EPIC-prefill': [[10,0.3398], [20, 0.3398], [30,0.32617], [40, 0.33398]],
        'EPIC-prefill+decode': [[10, 0.50781], [20, 0.47070], [30,0.470703], [40, 0.46093]],
        'Naive': [[10, 0.337]],
        'FR': [[100, 0.54492]]
    },
    'DROP': {
        'CacheBlend-prefill': [[10, 0.453], [20, 0.5], [30, 0.578], [40, 0.5625]],
        'CacheBlend-prefill+decode': [[10, 0.468], [20, 0.546], [30, 0.5781], [40, 0.5703]],
        'KVShare-prefill (our)': [[10, 0.52], [20, 0.5625], [30,0.58593], [40, 0.6328]],
        'KVShare-prefill+decode (our)': [[10, 0.5468], [20, 0.625], [30,0.664], [40, 0.6875]],
        'EPIC-prefill': [[10,0.453125], [20,0.46093], [30,0.45312], [40, 0.5]],
        'EPIC-prefill+decode': [[10, 0.546875], [20, 0.453125], [30,0.5390], [40, 0.55468]],
        'Naive': [[10, 0.429]],
        'FR': [[100, 0.7265]]
    },
    'SAMSum': {
        'CacheBlend-prefill': [[10, 0.121071], [20, 0.1305], [30, 0.132118], [40, 0.147089]],
        'CacheBlend-prefill+decode': [[10, 0.120188], [20,0.1323895], [30,0.1363604], [40,0.149020]],
        'KVShare-prefill (our)': [[10, 0.120188], [20, 0.146217], [30, 0.150114], [40, 0.15214]],
        'KVShare-prefill+decode (our)': [[10, 0.1405901], [20, 0.157698], [30,0.16348], [40, 0.171663]],
        'EPIC-prefill': [[10,0.15200], [20, 0.1522], [30,0.15666], [40, 0.15664]],
        'EPIC-prefill+decode': [[10, 0.153191], [20,0.1534271], [30,0.1555444], [40,0.15770632]],
        'Naive': [[10, 0.05059]],
        'FR': [[100, 0.18888]]
    }
    },
    'Yi-1.5-9B': {
        'DROP': {
            'CacheBlend-prefill': [[10, 0.4375], [20, 0.539], [30, 0.5390], [40, 0.59375]],
            'CacheBlend-prefill+decode': [[10, 0.5], [20, 0.5090625], [30, 0.570312], [40, 0.601562]],
            'KVShare-prefill (our)': [[10, 0.53906], [20, 0.4765], [30, 0.507812], [40, 0.5625]],
            'KVShare-prefill+decode (our)': [[10, 0.5859375], [20, 0.5990625], [30, 0.6328125], [40,0.65625]],
            'EPIC-prefill': [[10, 0.4609375], [20, 0.453125], [30, 0.453125], [40, 0.5]],
            'EPIC-prefill+decode': [[10, 0.5390625], [20,0.5078125], [30,0.5546875], [40, 0.5234375]],
            'Naive': [[10, 0.4078]],
            'FR': [[100, 0.684375]]
        },
        'GSM8K': {
            'CacheBlend-prefill': [[10, 0.24804], [20, 0.3125], [30, 0.32226], [40,0.3535]],
            'CacheBlend-prefill+decode': [[10, 0.453], [20, 0.5], [30, 0.578], [40, 0.5625]],
            'KVShare-prefill (our)': [[10, 0.515625], [20, 0.4574], [30, 0.4984], [40, 0.44335]],
            'KVShare-prefill+decode (our)': [[10, 0.51562], [20, 0.58398], [30, 0.587890], [40, 0.62304]],
            'EPIC-prefill': [[10,0.25585], [20,0.2421875], [30, 0.21484], [40, 0.25]],
            'EPIC-prefill+decode': [[10, 0.45898], [20,0.41406], [30,0.36914], [40,0.369140]],
            'Naive': [[10, 0.20703]],
            'FR': [[100, 0.69921]]
        },
        'SAMSum': {
            'CacheBlend-prefill': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
            'CacheBlend-prefill+decode': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
            'KVShare-prefill (our)': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
            'KVShare-prefill+decode (our)': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
            'EPIC-prefill': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
            'EPIC-prefill+decode': [[10, 0.13924], [20, 0.13921], [30, 0.15356], [40, 0.1580]],
            'Naive': [[10, 0.13924]],
            'FR': [[100, 0.13924]]
        }
    }
}

# 颜色和标记样式
# 为每个方法系列使用独特的颜色
colors = {
    'CacheBlend': '#ff5252',  # 红色
    'KVShare': '#4285F4',    # 蓝色
    'EPIC': '#7B1FA2',       # 紫色
    'Naive': '#FFA000',      # 橙色
    'FR': '#00C853'          # 绿色
}

# 为prefill和decode使用不同的标记
markers = {
    'prefill': '^',          # 三角形
    'decode': 'D'            # 菱形
}

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

# 创建子图布局 - 3x3布局
fig, axs = plt.subplots(3, 3, figsize=(15, 15), dpi=100)

# 在循环前加
ylabels = {'GSM8K': 'Accuracy', 'DROP': 'EM Score', 'SAMSum': 'F1 Score'}

# 为每个数据集和模型组合绘制子图
for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        # 跳过不存在的模型-数据集组合
        if model not in data or dataset not in data[model]:
            # 隐藏空的子图
            axs[i, j].set_visible(False)
            continue
            
        ax = axs[i, j]
        
        # 收集当前子图的所有y值
        all_y_values = []
        for method in cache_methods:
            if method in data[model][dataset]:
                if method in ['CacheBlend-prefill', 'CacheBlend-prefill+decode', 'KVShare-prefill (our)', 'KVShare-prefill+decode (our)', 'EPIC-prefill', 'EPIC-prefill+decode']:
                    points = data[model][dataset][method]
                    y_values = [point[1] for point in points]
                    all_y_values.extend(y_values)
                else:
                    # 对于Naive和FR，只取单个y值
                    y = data[model][dataset][method][0][1]
                    all_y_values.append(y)
        
        # 如果没有数据，跳过这个子图
        if not all_y_values:
            ax.set_visible(False)
            continue
            
        # 计算y轴范围
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        y_range = y_max - y_min
        # 恢复为自动设置y轴范围
        y_min = max(0, y_min - y_range * 0.1)  # 向下扩展10%，但不小于0
        y_max = min(1, y_max + y_range * 0.1)  # 向上扩展10%，但不大于1
        ax.set_ylim(y_min, y_max)
        
        # 设置x轴范围
        ax.set_xlim(0, 60)
        
        # 绘制数据点
        for k, method in enumerate(cache_methods):
            if method in data[model][dataset]:
                if method in ['CacheBlend-prefill', 'CacheBlend-prefill+decode', 'KVShare-prefill (our)', 'KVShare-prefill+decode (our)', 'EPIC-prefill', 'EPIC-prefill+decode']:
                    # 确定颜色和标记
                    if 'CacheBlend' in method:
                        color = colors['CacheBlend']
                    elif 'KVShare' in method:
                        color = colors['KVShare']
                    elif 'EPIC' in method:
                        color = colors['EPIC']
                    else:
                        continue
                    
                    marker = markers['decode'] if 'decode' in method else markers['prefill']
                    
                    # 对于有多个数据点的方法，需要连线
                    points = data[model][dataset][method]
                    x_values = [point[0] for point in points]
                    y_values = [point[1] for point in points]
                    
                    # 打印调试信息
                    print(f"Drawing {method} for {model} on {dataset}")
                    print(f"Points: {points}")
                    print(f"Color: {color}, Marker: {marker}")
                    
                    ax.plot(x_values, y_values, color=color, marker=marker, 
                           markersize=10, linewidth=line_width, alpha=0.5)
                    
                    # 添加数据点
                    for x, y in points:
                        ax.scatter(x, y, color=color, marker=marker, s=marker_size,
                                  edgecolors='black', linewidth=1.5, zorder=10)
                else:
                    # 对于Naive和FR，只创建水平线
                    x, y = data[model][dataset][method][0]
                    color = colors['Naive'] if method == 'Naive' else colors['FR']
                    # 创建从0到x的水平线
                    if method == 'Naive':
                        ax.plot([0, 100], [y, y], color=color, linewidth=line_width, 
                               alpha=0.9, linestyle='--')
                    else:
                        ax.plot([0, x], [y, y], color=color, linewidth=line_width, 
                               alpha=0.9, linestyle='--')
        
        # 设置轴标签
        if j == 0:  # 第一列
            ax.set_ylabel(ylabels[dataset], fontsize=14, fontweight='bold')
        
        # 在每个子图底部添加x轴标签
        if i == 2:  # 最后一行
            ax.set_xlabel('Re-compute Ratio (%)', fontsize=14, fontweight='bold')
        
        # 设置刻度
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))
        
        # 根据y轴范围动态设置y轴刻度间隔
        y_range = y_max - y_min
        if y_range <= 0.2:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        elif y_range <= 0.4:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        else:
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        
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
    fig.text(0.02, 0.75-0.25*i, dataset, va='center', ha='center', rotation='vertical', fontsize=18, fontweight='bold')

# 创建图例
handles = []
labels = []

# 添加方法系列的图例
for method in ['CacheBlend', 'KVShare', 'EPIC']:
    # 添加prefill
    handle = plt.Line2D([0], [0], color=colors[method], marker=markers['prefill'],
                      markersize=12, linewidth=line_width, 
                      markerfacecolor=colors[method], markeredgecolor='black')
    handles.append(handle)
    labels.append(f'{method}-prefill')
    
    # 添加decode
    handle = plt.Line2D([0], [0], color=colors[method], marker=markers['decode'],
                      markersize=12, linewidth=line_width, 
                      markerfacecolor=colors[method], markeredgecolor='black')
    handles.append(handle)
    labels.append(f'{method}-decode')

# 添加Naive和FR的图例
for method in ['Naive', 'FR']:
    handle = plt.Line2D([0], [0], color=colors[method], linestyle='--',
                      linewidth=line_width)
    handles.append(handle)
    labels.append(method)

# 将图例放在图的最上方
leg = fig.legend(
    handles=handles, labels=labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),  # 横向居中，纵向略高于图
    ncol=4,      # 每行4个
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
