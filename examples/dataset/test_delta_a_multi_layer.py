import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class MultiLayerTransformer:
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 初始化每一层的参数
        self.layers = []
        for _ in range(num_layers):
            layer = {
                'q_proj': torch.randn(hidden_size, hidden_size) * 0.02,
                'k_proj': torch.randn(hidden_size, hidden_size) * 0.02,
                'v_proj': torch.randn(hidden_size, hidden_size) * 0.02,
                'o_proj': torch.randn(hidden_size, hidden_size) * 0.02,
                'mlp_in': torch.randn(hidden_size, hidden_size * 4) * 0.02,
                'mlp_out': torch.randn(hidden_size * 4, hidden_size) * 0.02,
            }
            self.layers.append(layer)
            
    def forward(self, x: torch.Tensor, delta_k: List[torch.Tensor] = None, delta_v: List[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        for layer_idx, layer in enumerate(self.layers):
            # Self attention
            q = torch.matmul(x, layer['q_proj'])
            k = torch.matmul(x, layer['k_proj'])
            v = torch.matmul(x, layer['v_proj'])
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 只在第一层添加扰动
            if layer_idx == 0 and delta_k is not None and delta_v is not None:
                k = k + delta_k[0] * 0.1
                v = v + delta_v[0] * 0.1
            
            # Attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn, v)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            attn_output = torch.matmul(attn_output, layer['o_proj'])
            
            # Residual connection
            x = x + attn_output
            
            # MLP
            mlp_output = torch.matmul(x, layer['mlp_in'])
            mlp_output = torch.relu(mlp_output)
            mlp_output = torch.matmul(mlp_output, layer['mlp_out'])
            
            # Residual connection
            x = x + mlp_output
            
        return x

def compute_errors(model: MultiLayerTransformer, x: torch.Tensor, 
                  delta_k: List[torch.Tensor], delta_v: List[torch.Tensor],
                  top_ratio: float = 0.1) -> Tuple[List[float], List[float]]:
    """
    计算使用ΔV绝对值和ΔA两种方案的误差
    
    Args:
        model: MultiLayerTransformer模型
        x: 输入张量
        delta_k: K扰动列表
        delta_v: V扰动列表
        top_ratio: 选择top多少比例的token
        
    Returns:
        delta_v_errors: 使用ΔV绝对值方案的误差列表
        delta_a_errors: 使用ΔA方案的误差列表
    """
    batch_size, seq_len, hidden_size = x.shape
    delta_v_errors = []
    delta_a_errors = []
    
    # 计算原始输出
    original_output = model.forward(x)
    
    # 计算带扰动的输出
    perturbed_output = model.forward(x, delta_k, delta_v)
    
    # 计算ΔV的绝对值
    delta_v_norm = torch.norm(delta_v[0], dim=-1)  # (num_heads, seq_len)
    delta_v_norm = delta_v_norm.mean(dim=0)  # (seq_len,)
    
    # 选择top tokens
    top_n = max(1, int(seq_len * top_ratio))
    _, top_indices = torch.topk(delta_v_norm, top_n)
    
    # 创建mask
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[top_indices] = True
    
    # 计算每一层的误差
    for layer_idx in range(model.num_layers):
        # 获取当前层的输出
        curr_original = original_output[:, layer_idx:layer_idx+1, :]
        curr_perturbed = perturbed_output[:, layer_idx:layer_idx+1, :]
        
        # 计算ΔA
        diff = curr_perturbed - curr_original
        delta_a = torch.norm(diff, dim=-1)  # (batch_size, 1)
        delta_a = torch.clamp(delta_a, min=1e-8)
        
        # 使用ΔV绝对值方案的误差
        delta_v_error = delta_a.clone()
        delta_v_error[0, mask] = 0  # 将选中的token的误差置为0
        delta_v_errors.append(delta_v_error.mean().item())
        
        # 使用ΔA方案的误差
        delta_a_errors.append(delta_a.mean().item())
    
    return delta_v_errors, delta_a_errors

def test_error_comparison():
    # 设置参数
    num_layers = 6
    hidden_size = 256
    num_heads = 8
    batch_size = 1
    seq_len = 100
    
    # 创建模型
    model = MultiLayerTransformer(num_layers, hidden_size, num_heads)
    
    # 生成输入
    x = torch.randn(batch_size, seq_len, hidden_size) * 0.02
    
    # 为第一层生成扰动
    delta_k = [torch.randn(num_heads, seq_len, model.head_dim) * 0.02]
    delta_v = [torch.randn(num_heads, seq_len, model.head_dim) * 0.02]
    
    # 测试不同的top_ratio
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    delta_v_errors_all = []
    delta_a_errors_all = []
    
    for ratio in ratios:
        delta_v_errors, delta_a_errors = compute_errors(model, x, delta_k, delta_v, ratio)
        delta_v_errors_all.append(delta_v_errors)
        delta_a_errors_all.append(delta_a_errors)
    
    # 绘制结果
    plt.figure(figsize=(15, 5))
    
    # 绘制不同ratio下的误差对比
    plt.subplot(1, 2, 1)
    for i, ratio in enumerate(ratios):
        plt.plot(range(1, num_layers + 1), delta_v_errors_all[i], 
                marker='o', label=f'ΔV (top {int(ratio*100)}%)')
    plt.plot(range(1, num_layers + 1), delta_a_errors_all[0], 
            marker='s', label='ΔA', linestyle='--')
    plt.xlabel('Layer')
    plt.ylabel('Error')
    plt.title('Error Comparison across Layers')
    plt.grid(True)
    plt.legend()
    
    # 绘制不同ratio下的误差减少率
    plt.subplot(1, 2, 2)
    error_reduction = []
    for i, ratio in enumerate(ratios):
        reduction = [(delta_a_errors_all[0][j] - delta_v_errors_all[i][j]) / delta_a_errors_all[0][j] * 100 
                    for j in range(num_layers)]
        error_reduction.append(reduction)
    
    for i, ratio in enumerate(ratios):
        plt.plot(range(1, num_layers + 1), error_reduction[i], 
                marker='o', label=f'top {int(ratio*100)}%')
    plt.xlabel('Layer')
    plt.ylabel('Error Reduction (%)')
    plt.title('Error Reduction Rate across Layers')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('error_comparison.png')
    plt.show()
    
    # 打印具体数值
    print("\n误差对比:")
    for i, ratio in enumerate(ratios):
        print(f"\nTop {int(ratio*100)}% 的token:")
        print("Layer\tΔV Error\tΔA Error\tReduction")
        for layer in range(num_layers):
            reduction = (delta_a_errors_all[0][layer] - delta_v_errors_all[i][layer]) / delta_a_errors_all[0][layer] * 100
            print(f"{layer+1}\t{delta_v_errors_all[i][layer]:.6f}\t{delta_a_errors_all[0][layer]:.6f}\t{reduction:.2f}%")

if __name__ == "__main__":
    test_error_comparison() 