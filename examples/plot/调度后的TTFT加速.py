import matplotlib.pyplot as plt
import numpy as np

# Data preparation
methods = ['FC', 'CacheBlend', 'EPIC', 'KVShare']
schemes = ['Naive schedule', 'Optimized schedule']
qps_values = [24, 28, 30, 32]  # QPS values

# 每个方法/模式/QPS下都不同的数据
fc_ttft = [2484.27, 2718.93, 2930.70, 3080.16]  # FC 4个QPS
blend_naive_ttft = [2829.58, 3074.13, 3307.69, 3608.05]
blend_opt_ttft = [2141.15, 2325.44, 2403.53, 2478.83]
epic_naive_ttft = [2634.92, 2896.32, 3174.95, 3401.57]
epic_opt_ttft = [2152.93, 2476.39, 2621.42, 2856.71]
kv_naive_ttft = [367.71, 599.37, 1079.05, 1280.50]
kv_opt_ttft = [330.71, 576.00, 880.39, 847.04]

# 颜色和图案
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
hatches = [None, '////', '....']  # None for FC, '////' for Naive, 'o' for Optimized

n_qps = len(qps_values)
bar_width = 0.03
# 新的分组：FC, (空), BLEND-Naive, BLEND-Opt, (空), EPIC-Naive, EPIC-Opt, (空), KVSHARE-Naive, KVSHARE-Opt
gap_width = 0.005  # 空白间隔宽度
bars_per_group = 10  # 7根柱子+3个空槽
# 组间隔
group_gap = 0.02

# 计算每个柱子的x坐标
x = []
for i in range(n_qps):
    group_start = i * (bars_per_group * bar_width + 3 * gap_width + group_gap)
    # FC
    x.append(group_start)
    # 空槽1
    x.append(group_start + bar_width)
    # BLEND-Naive
    x.append(group_start + bar_width + gap_width)
    # BLEND-Opt
    x.append(group_start + 2 * bar_width + gap_width)
    # 空槽2
    x.append(group_start + 2 * bar_width + 2 * gap_width)
    # EPIC-Naive
    x.append(group_start + 2 * bar_width + 2 * gap_width + bar_width)
    # EPIC-Opt
    x.append(group_start + 2 * bar_width + 2 * gap_width + 2 * bar_width)
    # 空槽3
    x.append(group_start + 2 * bar_width + 3 * gap_width + 2 * bar_width)
    # KVSHARE-Naive
    x.append(group_start + 2 * bar_width + 3 * gap_width + 3 * bar_width)
    # KVSHARE-Opt
    x.append(group_start + 2 * bar_width + 3 * gap_width + 4 * bar_width)
x = np.array(x)

fig, ax = plt.subplots(figsize=(20, 6))

bars = []
for i in range(n_qps):
    base = i * bars_per_group
    # FC
    idx = base
    bar = ax.bar(x[idx], fc_ttft[i], width=bar_width, color=colors[0], hatch=None,
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)
    # BLEND
    idx = base + 2
    bar = ax.bar(x[idx], blend_naive_ttft[i], width=bar_width, color=colors[1], hatch='////',
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)
    idx = base + 3
    bar = ax.bar(x[idx], blend_opt_ttft[i], width=bar_width, color=colors[1], hatch='o',
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)
    # EPIC
    idx = base + 5
    bar = ax.bar(x[idx], epic_naive_ttft[i], width=bar_width, color=colors[2], hatch='////',
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)
    idx = base + 6
    bar = ax.bar(x[idx], epic_opt_ttft[i], width=bar_width, color=colors[2], hatch='o',
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)
    # KVSHARE
    idx = base + 8
    bar = ax.bar(x[idx], kv_naive_ttft[i], width=bar_width, color=colors[3], hatch='////',
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)
    idx = base + 9
    bar = ax.bar(x[idx], kv_opt_ttft[i], width=bar_width, color=colors[3], hatch='o',
                 alpha=0.8, edgecolor='black', linewidth=1.2)
    bars.append(bar)

# 添加数值标签
# def autolabel(barlist):
#     for bar in barlist:
#         for rect in bar:
#             height = rect.get_height()
#             ax.annotate(f'{height:.0f}',
#                         xy=(rect.get_x() + rect.get_width()/2, height),
#                         xytext=(0, 3),
#                         textcoords="offset points",
#                         ha='center', va='bottom', fontsize=14)

# autolabel(bars)

# 设置x轴：每组QPS的中心位置
xtick_positions = []
for i in range(n_qps):
    group_start = i * (bars_per_group * bar_width + 3 * gap_width + group_gap)
    # 取BLEND-Naive和KVSHARE-Naive之间的中点作为QPS标签位置
    left = x[i * bars_per_group + 2]
    right = x[i * bars_per_group + 8]
    xtick_positions.append((left + right) / 2)
ax.set_xticks(xtick_positions)
ax.set_xticklabels(qps_values, fontsize=16)
ax.set_xlabel('QPS (queries per second)', fontsize=18)
ax.set_ylabel('TTFT (ms)', fontsize=18)
# ax.set_title('TTFT for Each Method and Scheme at Different QPS', fontsize=20)

# 图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], label='FC', alpha=0.8, edgecolor='black'),
    Patch(facecolor=colors[1], label='BLEND', alpha=0.8, edgecolor='black'),
    Patch(facecolor=colors[2], label='EPIC', alpha=0.8, edgecolor='black'),
    Patch(facecolor=colors[3], label='KVSHARE', alpha=0.8, edgecolor='black'),
    Patch(facecolor='white', label='Naive schedule', hatch='////', edgecolor='black', alpha=0.8),
    Patch(facecolor='white', label='Optimized schedule', hatch='o', edgecolor='black', alpha=0.8)
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=14, ncol=2)

ax.tick_params(axis='y', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ttft_vs_qps_methods.pdf', dpi=300, bbox_inches='tight')
plt.close()
