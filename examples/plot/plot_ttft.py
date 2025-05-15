import matplotlib.pyplot as plt
import numpy as np

# 数据
qps = [2, 8, 16, 18, 20, 22, 24, 26, 28, 30, 32]
fr = [69.91, 70.37, 267.63, 627.56, 1369.49, 1766.90, 2212.31, 2484.27, 2718.93, 2930.70,3080.16]
cache_blend = [89.79, 98.60, 476.59, 1462.79, 1609.43, 2131.88, 2391.83, 2829.58, 3074.13, 3307.69,3608.05]
epic = [88.21, 97.03, 449.93, 977.35, 1480.86, 1817.44, 2238.26, 2634.92, 2896.32, 3174.95,3401.57]
kv_share = [74.22, 75.26, 119.92, 140.67, 159.09, 188.03, 271.77, 330.71, 576.00,847.04, 880.39]

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图表
plt.figure(figsize=(3,4))

# 绘制曲线
plt.plot(qps, fr, 'o-', label='FR', linewidth=2)
plt.plot(qps, cache_blend, 's-', label='CacheBlend', linewidth=2)
plt.plot(qps, epic, '^-', label='EPIC', linewidth=2)
plt.plot(qps, kv_share, 'd-', label='KVShare', linewidth=2)

# 设置图表属性
plt.xlabel('QPS', fontsize=12)
plt.ylabel('TTFT (ms)', fontsize=12)
# plt.title('不同方法在不同QPS下的TTFT性能对比', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

# 设置坐标轴范围
plt.xlim(0, 32)
plt.ylim(0, 3500)

# 保存图表
plt.savefig('ttft_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()