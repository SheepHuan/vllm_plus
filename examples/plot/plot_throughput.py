import matplotlib.pyplot as plt
import numpy as np
print(plt.style.available)
# 数据
qps = [2, 8, 16, 20, 22, 24, 26, 28, 30, 32]
fr = [1248.36, 1852.30, 1929.76, 1938.55, 1929.85, 1918.35, 1918.00, 1921.47, 1911.24, 1919.45]
cache_blend = [1281.35, 1875.27, 1941.29, 1995.61, 2021.08, 2017.91, 2010.39, 2002.06, 1992.24, 1990.49]
epic = [1283.42, 1907.09, 1997.86, 1996.94,  2005.1, 2002.01, 1996.14, 1979.01, 1942.83, 1973.05]
kv_share = [1182.12, 2108.29, 2266.41, 2262.88, 2287.16, 2316.16, 2349.75, 2375.09, 2394.05, 2332.09]

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 设置plot风格
# plt.style.use('seaborn-v0_8-whitegrid')
# 创建图表
plt.figure(figsize=(3, 4))

# 绘制曲线
plt.plot(qps, fr, 'o-', label='FR', linewidth=2)
plt.plot(qps, cache_blend, 's-', label='CacheBlend', linewidth=2)
plt.plot(qps, epic, '^-', label='EPIC', linewidth=2)
plt.plot(qps, kv_share, 'd-', label='KVShare', linewidth=2)

# 设置图表属性
plt.xlabel('QPS', fontsize=12)
plt.ylabel('Throughput (tokens/s)', fontsize=12)
# plt.title('不同方法在不同QPS下的吞吐量性能对比', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

# 设置坐标轴范围
plt.xlim(0, 32)
plt.ylim(1000, 2500)

# 保存图表
plt.savefig('throughput_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()