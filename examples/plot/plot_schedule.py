import matplotlib.pyplot as plt
import numpy as np

# 第一个图的数据
token_hit_rates = np.array([2.52,5.12,12.85,25.33,49.94,64.29,76.04,99.97])
ttft_ms = np.array([1458,1445,1442,1428,1106,812,595,158])

# 创建第一个图
plt.figure(figsize=(4,3))
plt.plot(token_hit_rates, ttft_ms, 'b-', linewidth=2,label='naive schedule')
plt.scatter(token_hit_rates, ttft_ms, color='blue', s=100, zorder=5)

# 添加连接第一个点和最后一个点的虚线
plt.plot([token_hit_rates[0], token_hit_rates[-1]], 
         [ttft_ms[0], ttft_ms[-1]], 
         'r--', linewidth=2, label='ideal schedule')

# 设置图表标题和轴标签
plt.xlabel('Avg. Hit Rate in Batch', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.3)

# 设置坐标轴范围
plt.xlim(0, max(token_hit_rates) * 1.1)
plt.ylim(0, max(ttft_ms) * 1.1)

# 添加图例
plt.legend(fontsize=10)

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 优化布局
plt.tight_layout()

# 保存第一个图
plt.savefig('ttft_vs_token_hit_rate.pdf', dpi=300, bbox_inches='tight')

# # 创建第二个图（柱状图）
# plt.figure(figsize=(8, 6))

# # 数据
# batch_sizes = np.array([1, 2, 3, 4, 5, 6, 10,11, 13,14, 16,17])

# # 105,105, 163,163, 209, 209, 105,105,163,163, 209, 209
# # 80%      40%      0%
# ttft_times = np.array([105,105, 163,163, 209, 209, 105,105,163,163, 209, 209])

# # 定义颜色
# colors = ['#1f77b4', '#1f77b4',  # 蓝色 
#           '#ff7f0e', '#ff7f0e',  # 橙色 
#           '#2ca02c', '#2ca02c',  # 绿色 
#           '#1f77b4', '#1f77b4',  # 蓝色 
#           '#ff7f0e', '#ff7f0e',  # 橙色 
#           '#2ca02c', '#2ca02c']  # 绿色 

# # 创建柱状图
# bars = plt.bar(batch_sizes, ttft_times, width=0.6, color=colors)

# # 设置x轴刻度和标签
# plt.xticks([])  # 移除x轴刻度标签
# plt.ylabel('TTFT (ms)', fontsize=14)

# # 添加图例
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='#1f77b4', label='80% hit rate'),
#     Patch(facecolor='#ff7f0e', label='40% hit rate'),
#     Patch(facecolor='#2ca02c', label='0% hit rate')
# ]
# plt.legend(handles=legend_elements, fontsize=12)

# # 添加网格线
# plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# # 优化布局
# plt.tight_layout()

# # 保存第二个图
# plt.savefig('batch_ttft_hitrate.png', dpi=300, bbox_inches='tight')