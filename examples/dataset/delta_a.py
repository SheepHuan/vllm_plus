import torch
import math
import numpy as np
import matplotlib.pyplot as plt

def attention_score(q,k):
    return torch.softmax(torch.matmul(q,k.transpose(-2,-1)/math.sqrt(dim)),dim=-1)

def attention(q, k, v):
    attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
    return torch.matmul(attn, v)

def delta_a_approx(q, k, v, delta_k, delta_v):
    # q: (head, seq_q, dim)
    # k, v, delta_k, delta_v: (head, seq_kv, dim)
    head, seq_q, dim = q.shape
    seq_kv = k.shape[1]
    sqrt_d = math.sqrt(dim)
    # 1. 计算 W = softmax(QK^T / sqrt(d_k))
    W = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / sqrt_d, dim=-1)  # (head, seq_q, seq_kv)

    # ΔV对ΔA的影响
    # 对每个token i，W[:,:,i] * ΔV_i，最后对所有head和query位置做范数
    impact_v = []
    for i in range(seq_kv):
        # W[:,:,i]: (head, seq_q)
        # delta_v[:,i,:]: (head, dim)
        # 广播相乘后再sum
        # (head, seq_q, 1) * (head, 1, dim) -> (head, seq_q, dim)
        wv = W[:,:,i].unsqueeze(-1) * delta_v[:,i,:].unsqueeze(1)  # (head, seq_q, dim)
        impact_v.append(wv.norm(dim=(0,2)).sum())  # 对head和dim做范数，再对seq_q求和
    impact_v = torch.stack(impact_v)  # (seq_kv,)

    # ΔK对ΔA的影响
    # 先计算 ∂A/∂K: (head, seq_q, dim)
    # 这里我们用之前的推导
    diag_WV = torch.einsum('hij,hjd->hid', W, v)  # (head, seq_q, dim)
    tmp = torch.matmul(W, v)  # (head, seq_q, dim)
    tmp2 = torch.matmul(W.transpose(-2, -1), q)  # (head, seq_kv, dim)
    SSVTQ = torch.matmul(W, tmp2)  # (head, seq_q, dim)
    dA_dK = (diag_WV - SSVTQ) / sqrt_d  # (head, seq_q, dim)

    # 对每个token i，dA_dK[:,i,:] * ΔK_i，最后对所有head和query位置做范数
    impact_k = []
    for i in range(seq_kv):
        # dA_dK[:,i,:]: (head, dim)
        # delta_k[:,i,:]: (head, dim)
        # 直接相乘
        kk = dA_dK[:,i,:] * delta_k[:,i,:]  # (head, dim)
        impact_k.append(kk.norm(dim=1).sum())  # 对head做范数再sum
    impact_k = torch.stack(impact_k)  # (seq_kv,)

    # 总影响
    total_impact = impact_v + impact_k  # (seq_kv,)

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
    delta_a_norm, delta_v_influence, delta_k_influence = delta_a_approx(q, k, v, delta_k, delta_v)
    delta_v_norm_token = torch.norm(delta_v, dim=-1).sum(dim=0)  # (seq_kv,)
    delta_k_norm_token = torch.norm(delta_k, dim=-1).sum(dim=0)  # (seq_kv,)
    S = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(v.shape[-1]), dim=-1)  # (head, seq_q, seq_kv)
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
    return error_dict

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
methods = ['kv', 'vnorm', 'knorm']
N = 100  # 实验次数
all_errors = {m: [] for m in methods}

for i in range(N):
    # 随机扰动
    mask = np.ones(seq_kv, dtype=bool)  # 所有位置都为True
    mask_tensor = torch.tensor(mask, device=q.device).float()
    delta_k = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None]
    delta_v = torch.randn(head, seq_kv, dim) * mask_tensor[None, :, None]
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
    # 's': 'Attention weight only'
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

