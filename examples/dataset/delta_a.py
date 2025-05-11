import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def attention_score(q,k):
    return torch.softmax(torch.matmul(q,k.transpose(-2,-1)/math.sqrt(dim)),dim=-1)

def attention(q, k, v):
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
    return torch.matmul(attn, v)

def delta_a_approx(q, k, v, delta_k, delta_v):
    # q, k, v, delta_k, delta_v: (head, seq, dim)
    head, seq, dim = q.shape
    sqrt_d = math.sqrt(dim)
    # 下三角mask
    mask = torch.tril(torch.ones(seq, seq, device=q.device)).unsqueeze(0)  # (1, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt_d  # (head, seq, seq)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    S = torch.softmax(scores, dim=-1)  # (head, seq, seq)

    # ΔV对ΔA的影响
    # 只考虑mask范围内的query
    # S: (head, seq, seq), delta_v: (head, seq, dim)
    # 只保留mask范围内的query
    mask_bool = mask.bool()[0]  # (seq, seq)
    # S_masked: (head, seq, seq)
    S_masked = S * mask  # 非mask位置为0
    # (head, seq, seq, dim)
    wv = S_masked.unsqueeze(-1) * delta_v.unsqueeze(1)  # 广播到 (head, seq, seq, dim)
    wv_norm = wv.norm(dim=(0,3))  # (seq, seq)
    impact_v = wv_norm.sum(dim=0)  # (seq,)

    # ΔK对ΔA的影响
    # 先计算 ∂A/∂K: (head, seq, dim)
    diag_WV = torch.einsum('hij,hjd->hid', S, v)  # (head, seq, dim)
    tmp2 = torch.matmul(S.transpose(-2, -1), q)  # (head, seq, dim)
    SSVTQ = torch.matmul(S, tmp2)  # (head, seq, dim)
    dA_dK = (diag_WV - SSVTQ) / sqrt_d  # (head, seq, dim)

    # dA_dK: (head, seq, dim), delta_k: (head, seq, dim)
    # mask: (seq, seq)
    # 只保留mask范围内的query
    dA_dK_expand = dA_dK.unsqueeze(2)  # (head, seq, 1, dim)
    delta_k_expand = delta_k.unsqueeze(1)  # (head, 1, seq, dim)
    # mask: (1, seq, seq, 1)
    mask_expand = mask.unsqueeze(0).unsqueeze(-1)
    # 只在mask范围内相乘
    kk = dA_dK_expand * delta_k_expand * mask_expand  # (head, seq, seq, dim)
    kk_norm = kk.norm(dim=(0,3))  # (seq, seq)
    impact_k = kk_norm.sum(dim=(0,-1))  # (seq,)

    # z-score标准化
    impact_k = (impact_k - impact_k.min()) / (impact_k.max() - impact_k.min() + 1e-8)
    impact_v = (impact_v - impact_v.min()) / (impact_v.max() - impact_v.min() + 1e-8)
    
    total_impact = impact_v + impact_k
    return impact_v, impact_k, total_impact

def reduce_top_influence_tokens(q, k, v, delta_k, delta_v, delta_v_influence, delta_k_influence, topn=None):
    seq_kv = k.shape[1]
    if topn is None:
        topn = max(1, int(seq_kv * 0.1))  # 至少为1
    # 1. 计算总影响力
    total_influence = delta_v_influence + delta_k_influence  # (seq_kv,)
    # 2. 找到topn个影响最大的token索引
    top_indices = torch.topk(total_influence, topn).indices.cpu().numpy()
    print("影响力最大的token索引:", top_indices)

    # 3. 备份原始扰动
    delta_k_new = delta_k.clone()
    delta_v_new = delta_v.clone()
    # 4. 将topn token的扰动置零
    delta_k_new[:, top_indices, :] = 0
    delta_v_new[:, top_indices, :] = 0

    # 5. 重新计算attention误差
    a1 = attention(q, k, v)
    a2 = attention(q, k + delta_k_new, v + delta_v_new)
    delta_a_true_new = a2 - a1

    # 6. 返回新误差和top token索引
    return delta_a_true_new, top_indices

def compare_zeroing_strategies(q, k, v, delta_k, delta_v, ratios, methods):
    seq_kv = k.shape[1]
    a1 = attention(q, k, v)
    delta_a_norm, delta_v_influence, delta_k_influence = delta_a_approx_autograd(q, k, v, delta_k, delta_v)
    delta_v_norm_token = torch.norm(delta_v, dim=-1).sum(dim=0)  # (seq_kv,)
    delta_k_norm_token = torch.norm(delta_k, dim=-1).sum(dim=0)  # (seq_kv,)
    # 生成下三角mask
    mask = torch.tril(torch.ones(q.shape[1], k.shape[1], device=q.device)).unsqueeze(0)  # (1, seq_q, seq_kv)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(v.shape[-1])  # (head, seq_q, seq_kv)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    S = torch.softmax(scores, dim=-1)  # (head, seq_q, seq_kv)
    S_sum_token = S.sum(dim=(0,1))  # (seq_kv,)

    error_dict = {m: [] for m in methods}

    for ratio in ratios:
        topn = max(1, int(seq_kv * ratio))
        # 1. K+V综合影响力
        if 'kv' in methods:
            total_influence = delta_v_influence + delta_k_influence
            top_indices = torch.topk(total_influence, topn).indices.cpu().numpy()
            delta_k_new = delta_k.clone()
            delta_v_new = delta_v.clone()
            delta_k_new[:, top_indices, :] = 0
            delta_v_new[:, top_indices, :] = 0
            a2 = attention(q, k + delta_k_new, v + delta_v_new)
            delta_a_true = torch.abs(a2 - a1)
            error = torch.norm(delta_a_true, dim=-1).sum().item()
            error_dict['kv'].append(error)
        # 2. |ΔV| only
        if 'vnorm' in methods:
            top_indices = torch.topk(delta_v_norm_token, topn).indices.cpu().numpy()
            delta_k_new = delta_k.clone()
            delta_v_new = delta_v.clone()
            delta_k_new[:, top_indices, :] = 0
            delta_v_new[:, top_indices, :] = 0
            a2 = attention(q, k + delta_k_new, v + delta_v_new)
            delta_a_true = torch.abs(a2 - a1)
            error = torch.norm(delta_a_true, dim=-1).sum().item()
            error_dict['vnorm'].append(error)
        # 3. |ΔK| only
        if 'knorm' in methods:
            top_indices = torch.topk(delta_k_norm_token, topn).indices.cpu().numpy()
            delta_k_new = delta_k.clone()
            delta_v_new = delta_v.clone()
            delta_k_new[:, top_indices, :] = 0
            delta_v_new[:, top_indices, :] = 0
            a2 = attention(q, k + delta_k_new, v + delta_v_new)
            delta_a_true = torch.abs(a2 - a1)
            error = torch.norm(delta_a_true, dim=-1).sum().item()
            error_dict['knorm'].append(error)
        # 4. Attention weight only
        if 's' in methods:
            top_indices = torch.topk(S_sum_token, topn).indices.cpu().numpy()
            delta_k_new = delta_k.clone()
            delta_v_new = delta_v.clone()
            delta_k_new[:, top_indices, :] = 0
            delta_v_new[:, top_indices, :] = 0
            a2 = attention(q, k + delta_k_new, v + delta_v_new)
            delta_a_true = torch.abs(a2 - a1)
            error = torch.norm(delta_a_true, dim=-1).sum().item()
            error_dict['s'].append(error)
        # 4. 只用 impact_v 排序
        if 'vimpact' in methods:
            top_indices = torch.topk(delta_v_influence, topn).indices.cpu().numpy()
            delta_k_new = delta_k.clone()
            delta_v_new = delta_v.clone()
            delta_v_new[:, top_indices, :] = 0  # 只置零 ΔV
            a2 = attention(q, k + delta_k_new, v + delta_v_new)
            delta_a_true = torch.abs(a2 - a1)
            error = torch.norm(delta_a_true, dim=-1).sum().item()
            error_dict['vimpact'].append(error)
        # 5. 只用 impact_k 排序
        if 'kimpact' in methods:
            top_indices = torch.topk(delta_k_influence, topn).indices.cpu().numpy()
            delta_k_new = delta_k.clone()
            delta_v_new = delta_v.clone()
            delta_k_new[:, top_indices, :] = 0  # 只置零 ΔK
            a2 = attention(q, k + delta_k_new, v + delta_v_new)
            delta_a_true = torch.abs(a2 - a1)
            error = torch.norm(delta_a_true, dim=-1).sum().item()
            error_dict['kimpact'].append(error)
    return error_dict

def delta_a_approx_autograd(q, k, v, delta_k, delta_v):
    """
    用自动求导分析ΔK和ΔV中哪些token对A影响最大
    输入:
        q, k, v, delta_k, delta_v: (head, seq, dim)
    输出:
        impact_v, impact_k, total_impact: (seq,)
    """
    head, seq, dim = q.shape
    sqrt_d = math.sqrt(dim)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)

    # 下三角mask
    mask = torch.tril(torch.ones(seq, seq, device=q.device))
    scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt_d
    scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    A = torch.matmul(attn, v)  # (head, seq, dim)

    # 只分析A的L2范数（也可以分析A.sum()或其它目标）
    loss = A.norm()
    loss.backward()

    # v.grad, k.grad: (head, seq, dim)
    # ΔA ≈ grad * ΔV/ΔK，统计每个token的影响
    impact_v = (v.grad * delta_v).norm(dim=(0,2))  # (seq,)
    impact_k = (k.grad * delta_k).norm(dim=(0,2))  # (seq,)
    # 归一化到[0,1]
    impact_v = (impact_v - impact_v.min()) / (impact_v.max() - impact_v.min() + 1e-8)
    impact_k = (impact_k - impact_k.min()) / (impact_k.max() - impact_k.min() + 1e-8)
    total_impact = impact_v + impact_k

    return impact_v, impact_k, total_impact

seq_q = 100
seq_kv = 100
head = 4
dim = 64

q = torch.randn(head, seq_q, dim)
k = torch.randn(head, seq_kv, dim)
v = torch.randn(head, seq_kv, dim)

# 1. 生成mask
mask = np.ones(seq_kv, dtype=bool)  # (seq_kv,)
mask_tensor = torch.tensor(mask, device=q.device).float()  # (seq_kv,)

# 2. 获取有误差和没误差的token索引
idx_kv = np.where(mask)[0]      # 有误差的token索引
idx_q = np.where(~mask)[0]      # 没误差的token索引


ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
methods = ['kv', 'vnorm', 'knorm', 'vimpact', 'kimpact']
N = 100  # 实验次数
all_errors = {m: [] for m in methods}

for i in range(N):
    # 随机扰动
    mask = np.ones(seq_kv, dtype=bool)  # 所有位置都为True
    mask_tensor = torch.tensor(mask, device=q.device).float()
    delta_k = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None] 
    delta_v = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None] * 2
    errors = compare_zeroing_strategies(q, k, v, delta_k, delta_v, ratios, methods)
    for m in methods:
        all_errors[m].append(errors[m])

# 计算均值和方差
mean_errors = {m: np.mean(all_errors[m], axis=0) for m in methods}
std_errors = {m: np.std(all_errors[m], axis=0) for m in methods}

# 绘制曲线
plt.figure(figsize=(6, 6))
labels = {
    'kv': r'$\partial ΔA/\partial \Delta K + \partial ΔA/\partial \Delta V$',
    'vnorm': r'$|\Delta V|$ only',
    'knorm': r'$|\Delta K|$ only',
    'vimpact': r'$\partial ΔA/\partial \Delta V$',
    'kimpact': r'$\partial ΔA/\partial \Delta K$',
}
for m in methods:
    plt.errorbar([int(r*100) for r in ratios], mean_errors[m], yerr=std_errors[m], marker='o', label=labels[m])
plt.xlabel('Top affected token ratio (%)')
plt.ylabel('Total ΔA error')
plt.title('ΔA error after zeroing top affected tokens (mean±std, N={})'.format(N))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('delta_a_influence_multi.png')
plt.show()

# 多次实验的统计分析
N = 100  # 实验次数
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
all_overlaps = {r: [] for r in ratios}

print(f"\n开始进行{N}次实验的统计分析...")
for exp in range(N):
    # 生成新的随机扰动
    delta_k = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None]
    delta_v = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None] * 2
    
    # 计算impact
    impact_v, impact_k, total_impact = delta_a_approx_autograd(q, k, v, delta_k, delta_v)
    
    # 计算每个比例的重叠
    seq = impact_v.shape[0]
    for ratio in ratios:
        top_n = int(seq * ratio)
        top_v_idx = np.argpartition(-impact_v.cpu().numpy(), top_n)[:top_n]
        top_k_idx = np.argpartition(-impact_k.cpu().numpy(), top_n)[:top_n]
        overlap = len(set(top_v_idx) & set(top_k_idx)) / top_n
        all_overlaps[ratio].append(overlap)

# 计算统计结果
mean_overlaps = {r: np.mean(all_overlaps[r]) for r in ratios}
std_overlaps = {r: np.std(all_overlaps[r]) for r in ratios}

# 打印统计结果
print("\n重叠比例统计结果（平均值±标准差）：")
for ratio in ratios:
    print(f"Top {int(ratio*100)}%: {mean_overlaps[ratio]:.2%} ± {std_overlaps[ratio]:.2%}")

# 绘制带误差棒的统计结果
plt.figure(figsize=(10, 6))
plt.errorbar([int(r*100) for r in ratios], 
             [mean_overlaps[r] for r in ratios],
             yerr=[std_overlaps[r] for r in ratios],
             fmt='o-', linewidth=2, capsize=5)
plt.xlabel('Top r%')
plt.ylabel('Overlap Ratio')
plt.title(f'Overlap Ratio between Top r% impact_v and impact_k (N={N})')
plt.grid(True)
plt.xticks([int(r*100) for r in ratios])
plt.tight_layout()
plt.savefig('overlap_ratios_statistics.png')
plt.show()

# 计算皮尔森相关系数的统计结果
all_correlations = []
for exp in range(N):
    delta_k = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None]
    delta_v = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None] * 2
    impact_v, impact_k, _ = delta_a_approx_autograd(q, k, v, delta_k, delta_v)
    r, _ = pearsonr(impact_v.cpu().numpy(), impact_k.cpu().numpy())
    all_correlations.append(r)

print(f"\n皮尔森相关系数统计结果（N={N}）：")
print(f"平均值: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")

# 创建一个大图，包含所有分析结果
plt.figure(figsize=(20, 15))

# 1. 重叠比例分析子图
plt.subplot(2, 2, 1)
plt.errorbar([int(r*100) for r in ratios], 
             [mean_overlaps[r] for r in ratios],
             yerr=[std_overlaps[r] for r in ratios],
             fmt='o-', linewidth=2, capsize=5)
plt.xlabel('Top r%')
plt.ylabel('Overlap Ratio')
plt.title(f'Overlap Ratio between Top r% impact_v and impact_k (N={N})')
plt.grid(True)
plt.xticks([int(r*100) for r in ratios])

# 2. 最后一次实验的散点图
plt.subplot(2, 2, 2)
plt.scatter(impact_v.cpu().numpy(), impact_k.cpu().numpy(), alpha=0.5)
plt.xlabel('impact_v')
plt.ylabel('impact_k')
plt.title(f'impact_v vs impact_k (Pearson r={np.mean(all_correlations):.2f})')
plt.grid(True)

# 3. Top 10%重叠可视化
plt.subplot(2, 2, 3)
v_mask = np.zeros(seq)
k_mask = np.zeros(seq)
v_mask[top_v_idx] = 1
k_mask[top_k_idx] = 1
plt.plot(v_mask, label='Top 10% impact_v')
plt.plot(k_mask, label='Top 10% impact_k')
plt.plot(v_mask * k_mask, label='Overlap', linewidth=3)
plt.xlabel('Token Index')
plt.ylabel('Top 10% Flag')
plt.title(f'Top 10% Overlap Visualization')
plt.legend()

# 4. 皮尔森相关系数分布
plt.subplot(2, 2, 4)
plt.hist(all_correlations, bins=20, alpha=0.7)
plt.axvline(np.mean(all_correlations), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(all_correlations):.3f}')
plt.xlabel('Pearson Correlation')
plt.ylabel('Frequency')
plt.title('Distribution of Pearson Correlation (N={})'.format(N))
plt.legend()
plt.grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印统计结果
print("\n=== 统计分析结果 ===")
print("\n重叠比例统计结果（平均值±标准差）：")
for ratio in ratios:
    print(f"Top {int(ratio*100)}%: {mean_overlaps[ratio]:.2%} ± {std_overlaps[ratio]:.2%}")

print(f"\n皮尔森相关系数统计结果（N={N}）：")
print(f"平均值: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")
print(f"最小值: {np.min(all_correlations):.3f}")
print(f"最大值: {np.max(all_correlations):.3f}")

# 1. 重叠比例分析图
plt.figure(figsize=(10, 6))
plt.errorbar([int(r*100) for r in ratios], 
             [mean_overlaps[r] for r in ratios],
             yerr=[std_overlaps[r] for r in ratios],
             fmt='o-', linewidth=2, capsize=5)
plt.xlabel('Top r%')
plt.ylabel('Overlap Ratio')
plt.title(f'Overlap Ratio between Top r% impact_v and impact_k (N={N})')
plt.grid(True)
plt.xticks([int(r*100) for r in ratios])
plt.tight_layout()
plt.savefig('overlap_ratio_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 最后一次实验的散点图
plt.figure(figsize=(10, 6))
plt.scatter(impact_v.cpu().numpy(), impact_k.cpu().numpy(), alpha=0.5)
plt.xlabel('impact_v')
plt.ylabel('impact_k')
plt.title(f'impact_v vs impact_k (Pearson r={np.mean(all_correlations):.2f})')
plt.grid(True)
plt.tight_layout()
plt.savefig('impact_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Top 10%重叠可视化
plt.figure(figsize=(10, 6))
v_mask = np.zeros(seq)
k_mask = np.zeros(seq)
v_mask[top_v_idx] = 1
k_mask[top_k_idx] = 1
plt.plot(v_mask, label='Top 10% impact_v')
plt.plot(k_mask, label='Top 10% impact_k')
plt.plot(v_mask * k_mask, label='Overlap', linewidth=3)
plt.xlabel('Token Index')
plt.ylabel('Top 10% Flag')
plt.title(f'Top 10% Overlap Visualization')
plt.legend()
plt.tight_layout()
plt.savefig('top10_overlap_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 皮尔森相关系数分布
plt.figure(figsize=(10, 6))
plt.hist(all_correlations, bins=20, alpha=0.7)
plt.axvline(np.mean(all_correlations), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(all_correlations):.3f}')
plt.xlabel('Pearson Correlation')
plt.ylabel('Frequency')
plt.title('Distribution of Pearson Correlation (N={})'.format(N))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pearson_correlation_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印统计结果
print("\n=== 统计分析结果 ===")
print("\n重叠比例统计结果（平均值±标准差）：")
for ratio in ratios:
    print(f"Top {int(ratio*100)}%: {mean_overlaps[ratio]:.2%} ± {std_overlaps[ratio]:.2%}")

print(f"\n皮尔森相关系数统计结果（N={N}）：")
print(f"平均值: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")
print(f"最小值: {np.min(all_correlations):.3f}")
print(f"最大值: {np.max(all_correlations):.3f}")

