import json
import random
from edit2 import find_text_differences,apply_change
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MKL_THREADING_LAYER"] = "GNU"  # 强制使用 GNU 线程层
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"  # 可选：强制使用 Intel 线程
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

"""
计算相似文本对的REUSE RATIO和EFFICIENCY之间的关系
"""
from vllm import LLM
from vllm.entrypoints.llm import SamplingParams,RequestOutput
from transformers import AutoTokenizer
import numpy as np
from typing import List
import torch

def get_key_value(model:LLM,prompt: str):
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    output:List[RequestOutput] = model.generate(prompt, sampling_params,use_tqdm=False)
    
    llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    
    past_key_values = []
    num_layer = len(llm_layers)
    for j in range(num_layer):
        hack_kv = llm_layers[j].self_attn.hack_kv
        temp_key_cache = hack_kv[0].clone()
        temp_value_cache = hack_kv[1].clone()
        past_key_values.append(torch.stack([temp_key_cache,temp_value_cache],dim=0))
    past_key_values = torch.stack(past_key_values,dim=0).detach().cpu()
    return past_key_values,output[0].prompt_token_ids

def full_compute(llm_model:LLM,text:str,test_num:int=20):
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    avg_ttft_time = []
    for _ in range(test_num):
        output = llm_model.generate(text,sampling_params,use_tqdm=False)
        ttft_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        avg_ttft_time.append(ttft_time)
    tokens = output[0].prompt_token_ids
    return np.mean(avg_ttft_time[5:])*1000,tokens
    
def partial_compute(llm_model:LLM,text:str,reused_map_indices:List[int],unused_map_indices:List[int],reused_kvcache:torch.Tensor,test_num:int=20,device="cuda:0"):
    
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["recomp_ratio"] = 0.0
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = True
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = torch.tensor(unused_map_indices).to(device).to(torch.int64)
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = torch.tensor(reused_map_indices).to(device).to(torch.int64)
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = reused_kvcache.to(device).to(torch.bfloat16)
    
    sampling_params = SamplingParams(max_tokens=1)
    avg_ttft_time = []
    for _ in range(test_num):
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = True
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        output = llm_model.generate(text,sampling_params,use_tqdm=False)
        ttft_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        avg_ttft_time.append(ttft_time)
    tokens = output[0].prompt_token_ids
    return np.mean(avg_ttft_time[20:])*1000,tokens

def compute_speed_up(model:LLM,text1:str,text2:str):
    
    from edit2 import find_text_differences,apply_change
    source_kv,source_tokens = get_key_value(model,text1)
    full_ttft_time,target_tokens = full_compute(model,text2,50)
    diff_report = find_text_differences(source_tokens,target_tokens,window_size=1)
    modified_kv,reused_map_indices,unused_map_indices = apply_change(source_tokens,target_tokens,source_kv,diff_report)
    
    partial_ttft_time,partial_tokens = partial_compute(model,text2,reused_map_indices,unused_map_indices,modified_kv,50)
    
    speed_up_ratio = full_ttft_time/partial_ttft_time
    print(f"speed_up_ratio: {speed_up_ratio}, full_ttft_time: {full_ttft_time}, partial_ttft_time: {partial_ttft_time}, reuse rate: {len(reused_map_indices)/len(target_tokens)} ,reuesd token:{len(reused_map_indices)}, unreused token:{len(unused_map_indices)}")
    return speed_up_ratio


def test_xformers(query,key,value):
    import time
    from xformers.ops import memory_efficient_attention_forward
    from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask
    avg_time = []
    for _ in range(100):
        time1 = time.time()
        out = memory_efficient_attention_forward(
                            query,
                            key,
                            value,
                            attn_bias=LowerTriangularFromBottomRightMask(),
                            p=0.0,
                        )
        time2 = time.time()
        avg_time.append(time2-time1)
    print(f"time: {np.mean(avg_time[50:])*1000}")
    
    


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    device = "cuda:0"
    llm_model = LLM(model=model_name,
                    device=device,
                    # dtype="bfloat16",
                    gpu_memory_utilization=0.8,
                    multi_step_stream_outputs=True,
                    # enforce_eager=True,
                    disable_async_output_proc=True,
                    max_model_len=8192
                    )
    text1 = "Hello, world!" * 100
    text2 = "Hello, world!" * 100
    speed_up_ratio = compute_speed_up(llm_model,text1,text2)
    # seq = 5000
    # head = 16
    # head_dim = 128
    
    # #
    # query = torch.randn(1,seq,head,head_dim).to(device) 
    # key = torch.randn(1,seq,head,head_dim).to(device) 
    # value = torch.randn(1,seq,head,head_dim).to(device) 
    # test_xformers(query,key,value)            
    
    # # 节约内存
    # query = torch.randn(1,3,head,head_dim).to(device) 
    # key = torch.randn(1,seq,head,head_dim).to(device) 
    # value = torch.randn(1,seq,head,head_dim).to(device) 
    # test_xformers(query,key,value)
    
    # 节约内存
    
def fit_poly(x, y, degree=3):
    """多项式拟合
    
    Args:
        x: x轴数据
        y: y轴数据
        degree: 多项式次数
    
    Returns:
        poly: 多项式函数
        mse: 均方误差
    """
    from sklearn.metrics import mean_squared_error
    
    # 对x进行排序，确保曲线平滑
    sort_idx = np.argsort(x)
    x_sorted = np.array(x)[sort_idx]
    y_sorted = np.array(y)[sort_idx]
    
    # 多项式拟合
    coefficients = np.polyfit(x_sorted, y_sorted, degree)
    poly = np.poly1d(coefficients)
    
    # 计算拟合值和均方误差
    y_pred = poly(x_sorted)
    mse = mean_squared_error(y_sorted, y_pred)
    
    return poly, x_sorted, y_sorted, mse

def plot_scatter_with_trend(ax, x, y, scatter_color, line_color, title, xlabel, ylabel):
    """绘制散点图和趋势线的通用函数"""
    import seaborn as sns
    
    # 绘制散点图
    sns.scatterplot(x=x, y=y,
                   alpha=0.6,
                   color=scatter_color,
                   ax=ax)
    
    # 多项式拟合
    poly, x_sorted, y_sorted, mse = fit_poly(x, y)
    
    # 绘制拟合曲线
    ax.plot(x_sorted, poly(x_sorted), 
            color=line_color, 
            linewidth=2,
            label=f'Polynomial fit (MSE: {mse:.3f})')
    
    # 计算相关系数
    correlation = np.corrcoef(x, y)[0,1]
    
    # 添加相关系数文本
    ax.text(0.05, 0.95, 
            f'Correlation: {correlation:.3f}\nMSE: {mse:.3f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            verticalalignment='top')
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

def plot_speed_up(data_path: str, save_path: str = "examples/pipeline/images/speed_up_analysis.png"):
    """绘制速度提升与token长度和复用率的关系图"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置绘图风格
    plt.style.use('seaborn')
    
    # 创建图表和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 加载数据
    data = json.load(open(data_path, "r"))
    
    # 提取数据
    reuse_rates = [item["reuse_ratio"] for item in data]
    speed_ups = [item["speed_up_ratio"] for item in data]
    reused_lengths = [item["target_token_len"] for item in data]
    
    # 绘制 Token Length vs Speed Up
    plot_scatter_with_trend(
        ax=ax1,
        x=reused_lengths,
        y=speed_ups,
        scatter_color='lightblue',
        line_color='darkblue',
        title='Token Length vs Speed Up',
        xlabel='Token Length',
        ylabel='Speed Up Ratio'
    )
    
    # 绘制 Reuse Rate vs Speed Up
    plot_scatter_with_trend(
        ax=ax2,
        x=reuse_rates,
        y=speed_ups,
        scatter_color='bisque',
        line_color='darkorange',
        title='Reuse Rate vs Speed Up',
        xlabel='Reuse Rate',
        ylabel='Speed Up Ratio'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    