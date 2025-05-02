import matplotlib.pyplot as plt
import numpy as np
import numpy as np
# 模拟数据
np.random.seed(42)
seq_lengths = [1, 2, 3, 4, 6, 8, 16, 32,64]

# 三个模型的decode时间 (单位: us)
model1_times = [0.0423,0.0476,0.0471,0.0479,0.0490,0.0491,0.0505,0.0548,0.0597] 
model2_times = [0.0514,0.0522,0.0525,0.0526,0.0533,0.0536,0.0562,0.0603,0.0856]   
model3_times = [0.8515,0.8552,0.8556,0.8560,0.8563,0.8568,0.8585,0.8613,0.8664] 

model1_times = np.array(model1_times)*1000
model2_times = np.array(model2_times)*1000 
model3_times = np.array(model3_times)*1000



# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, model1_times, 'o-', label='Mixtral-7B', linewidth=2)
plt.plot(seq_lengths, model2_times, 's-', label='Llama3.1-8B', linewidth=2)
plt.plot(seq_lengths, model3_times, '^-', label='Qwen2.5-32B', linewidth=2)

plt.xlabel('Sequence Length', fontsize=14)
plt.ylabel('Decode Time (us)', fontsize=14)
# plt.title('Decode Time Comparison of Different Models', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 添加网格和优化视觉效果
plt.tight_layout()
plt.savefig('decode_time_comparison.png', dpi=300)
plt.show()




