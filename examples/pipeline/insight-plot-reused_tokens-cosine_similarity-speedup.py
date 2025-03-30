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
from sentence_transformers import SentenceTransformer

def get_key_value(model:LLM,prompt: str,device:str="cuda:0"):
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
        temp_key_cache = hack_kv[0].clone().to(device)
        temp_value_cache = hack_kv[1].clone().to(device)
        past_key_values.append(torch.stack([temp_key_cache,temp_value_cache],dim=0))
    past_key_values = torch.stack(past_key_values,dim=0)
    return past_key_values,output[0].prompt_token_ids

def full_compute(llm_model:LLM,text:str,test_num:int=20):
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = False
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = None
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = None
    llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["imp_indices"] = None
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    avg_ttft_time = []
    for _ in range(test_num):
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        output = llm_model.generate(text,sampling_params,use_tqdm=False)
        ttft_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        avg_ttft_time.append(ttft_time)
    tokens = output[0].prompt_token_ids
    return np.mean(avg_ttft_time[5:])*1000,tokens
    
def partial_compute(llm_model:LLM,text:str,reused_map_indices:List[int],unused_map_indices:List[int],reused_kvcache,test_num:int=20,device="cuda:0"):
    additional_map_indices = torch.tensor(unused_map_indices).to(device).to(torch.int64)
    old_kv_map_indices = torch.tensor(reused_map_indices).to(device).to(torch.int64)
    sampling_params = SamplingParams(max_tokens=1)
    avg_ttft_time = []
    for _ in range(test_num):
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = True
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = False
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["recomp_ratio"] = 0.0
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["use_additional_indices"] = True
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["additional_map_indices"] = additional_map_indices
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["old_kv_map_indices"] = old_kv_map_indices
        llm_model.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = reused_kvcache
        output = llm_model.generate(text,sampling_params,use_tqdm=False)
        ttft_time = output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time
        avg_ttft_time.append(ttft_time)
    tokens = output[0].prompt_token_ids
    return np.mean(avg_ttft_time[5:])*1000,tokens


def compute_reuse_ratio_with_efficiency(model_name:str,data_path:str,max_num:int=10000,min_token_len:int=50,device:str="cuda:0",save_path:str=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    llm_model = LLM(model=model_name,
                    device=device,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.5,
                    multi_step_stream_outputs=True,
                    enforce_eager=True,
                    disable_async_output_proc=True
                    )
    with open(data_path,"r") as f:
        data = json.load(f)

    data = random.sample(data,min(max_num,len(data)))
    print(f"挑选的数据长度: {len(data)}")
    all_data_size = len(data)
    profiled_data = []
    # FIXME 循环体的第三次一定会报错
    for i,item in tqdm(enumerate(data),total=len(data)):
        try:
            source_text = item["source"]
            target_text = item["target"]
        
            source_kv,source_tokens = get_key_value(llm_model,source_text)
            full_ttft_time,target_tokens = full_compute(llm_model,target_text)
            
            diff_report = find_text_differences(source_tokens,target_tokens,window_size=1)
            modified_kv,reused_map_indices,unused_map_indices = apply_change(source_tokens,target_tokens,source_kv,diff_report)
            # print(f"reused ratio: {len(reused_map_indices)/len(target_tokens)}")
            
            partial_ttft_time,partial_tokens = partial_compute(llm_model,target_text,reused_map_indices,unused_map_indices,modified_kv,device=device)
            
            speed_up_ratio = full_ttft_time/partial_ttft_time
            reuse_ratio = len(reused_map_indices)/len(target_tokens)
            
            # print(f"{i}/{all_data_size}: ttft_time: {full_ttft_time}, partial_ttft_time: {partial_ttft_time}, speed_up_ratio: {speed_up_ratio}, reuse_ratio: {reuse_ratio}")
            # del source_kv,modified_kv,reused_map_indices,unused_map_indices
            profiled_data.append({
                "speed_up_ratio":speed_up_ratio,
                "reuse_ratio":reuse_ratio,
                "full_ttft_time":full_ttft_time,
                "partial_ttft_time":partial_ttft_time,
                "target_token_len":len(target_tokens),
                "source_token_len":len(source_tokens),
                "source_text": item["source"],
                "target_text": item["target"]
            })
        # torch.cuda.empty_cache()
        except Exception as e:
            print(f"error: {e}")
    if save_path is not None:
        json.dump(profiled_data,open(save_path,"w"),indent=4,ensure_ascii=False)

def select_data(data_path:str,save_path:str,min_token_len:int=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    with open(data_path,"r") as f:
        data = json.load(f)
    new_data = [item for item in data if len(tokenizer.encode(item["target"])) >= min_token_len]
    json.dump(new_data,open(save_path,"w"),indent=4,ensure_ascii=False)

    
def fit_poly(x, y, degree: int = 2):
    """多项式拟合
    
    Args:
        x: x轴数据
        y: y轴数据
        degree: 多项式次数
    
    Returns:
        y_fit: 拟合后的y值
    """
    # 对x进行排序，确保曲线平滑
    sort_idx = np.argsort(x)
    x_sorted = np.array(x)[sort_idx]
    y_sorted = np.array(y)[sort_idx]
    
    # 多项式拟合
    coefficients = np.polyfit(x_sorted, y_sorted, degree)
    poly = np.poly1d(coefficients)
    
    # 计算拟合值
    y_fit = poly(x)
    
    return y_fit

def plot_reused_tokens_with_speedup(data_path: str, image_save_path: str = "examples/pipeline/images/speed_up_analysis.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore
    
    # 创建单个图表
    plt.figure(figsize=(8, 6))
    
    # 加载数据
    data = json.load(open(data_path, "r"))
    
    # 提取数据
    reused_tokens = []
    speed_ups = []
    for item in data:
        if item["speed_up_ratio"] < 1.1:
            continue
        # 计算复用的token数量
        reused_token_count = int(item["reuse_ratio"] * item["target_token_len"])
        if reused_token_count>1024:
            continue
        reused_tokens.append(reused_token_count)
        speed_ups.append(item["speed_up_ratio"])
    
    # 转换为numpy数组
    reused_tokens = np.array(reused_tokens)
    speed_ups = np.array(speed_ups)
    
    # 过滤异常值
    z_scores = zscore(speed_ups)
    mask = np.abs(z_scores) < 3  # 保留3个标准差以内的数据
    reused_tokens = reused_tokens[mask]
    speed_ups = speed_ups[mask]
    
    # 绘制散点图
    sns.scatterplot(x=reused_tokens,
                    y=speed_ups,
                    alpha=0.6,
                    color='bisque',
                    label='Data points')
    
    # 计算并绘制中位数点
    # 创建均匀分布的区间
    min_tokens = 0
    max_tokens = max(reused_tokens)
    num_intervals = 50
    unique_tokens = np.linspace(min_tokens, max_tokens, num_intervals)
    median_speeds = []
    
    # 计算每个区间的中位数
    interval_width = (unique_tokens[1] - unique_tokens[0]) / 2  # 区间半宽
    for tokens in unique_tokens:
        # 定义区间范围
        lower_bound = tokens - interval_width
        upper_bound = tokens + interval_width
        # 找出落在这个区间内的所有点
        mask = (reused_tokens >= lower_bound) & (reused_tokens < upper_bound)
        if np.any(mask):  # 如果区间内有点
            median_speeds.append(np.median(speed_ups[mask]))
        else:  # 如果区间内没有点
            median_speeds.append(np.nan)  # 用nan标记空区间
    
    # 移除nan值
    valid_mask = ~np.isnan(median_speeds)
    unique_tokens = unique_tokens[valid_mask]
    median_speeds = np.array(median_speeds)[valid_mask]
    
    # # 绘制中位数点
    # plt.scatter(unique_tokens, median_speeds, 
    #            color='blue', 
    #            s=50,  # 点的大小
    #            alpha=0.8,
    #            label='Median points')
    
    # # 对中位数点进行多项式拟合
    x_smooth = np.linspace(0, max_tokens, 100)  # 从0开始
    coefficients = np.polyfit(unique_tokens, median_speeds, 2)
    poly = np.poly1d(coefficients)
    y_smooth = poly(x_smooth)
    
    # 确保曲线从(0,1.0)开始
    y_smooth = y_smooth - (y_smooth[0] - 1.0)
    
    # # 绘制上边界曲线
    # plt.plot(x_smooth, y_smooth, 
    #          '--', color='blue', alpha=0.7, 
    #          label='Median trend', linewidth=2)
    
    # # 绘制从(0,1.0)到最小值的下边界直线
    # min_speed = min(speed_ups)
    # plt.plot([0, max_tokens], [1.0, min_speed], '--', 
    #          color='green', alpha=0.7, 
    #          label='Lower bound', linewidth=2)
    
    # 添加中间的拟合曲线
    coefficients = np.polyfit(reused_tokens, speed_ups, 2)
    poly = np.poly1d(coefficients)
    y_smooth_middle = poly(x_smooth)
    plt.plot(x_smooth, y_smooth_middle, 
             color='red', 
             label='Trend',
             linewidth=2)
    
    # 设置标签和标题
    plt.xlabel("Number of Reused Tokens")
    plt.ylabel("Speed Up Ratio")
    plt.title("Token Reuse vs Speed Up")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 计算并显示相关系数
    reuse_corr = np.corrcoef(reused_tokens, speed_ups)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {reuse_corr:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(image_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_cosine_similar_with_speedup(similar_path: str, image_save_path: str = "examples/pipeline/images/cosine_similar_analysis.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore, pearsonr, spearmanr
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 加载数据
    data = json.load(open(similar_path, "r"))
    
    # 提取数据
    speed_ups = []
    similarities = []
   
    for item in data:
        if item["speed_up_ratio"] < 1.1:
            continue
        reused_token_count = int(item["reuse_ratio"] * item["target_token_len"])
        # if reused_token_count > 1024:
        #     continue
        speed_ups.append(item["speed_up_ratio"])
        similarities.append(item["cosine_similarity"]["colbert"])

    # 转换为numpy数组
    similarities = np.array(similarities)
    speed_ups = np.array(speed_ups)
    
    # 过滤异常值
    z_scores = zscore(speed_ups)
    mask = np.abs(z_scores) < 3
    similarities = similarities[mask]
    speed_ups = speed_ups[mask]
    
    # 计算相关系数
    pearson_corr, p_value = pearsonr(similarities, speed_ups)
    
    # 创建热力图数据
    x_bins = np.linspace(min(similarities), max(similarities), 40)
    y_bins = np.linspace(min(speed_ups), max(speed_ups), 40)
    density, _, _ = np.histogram2d(similarities, speed_ups, bins=[x_bins, y_bins])
    density = density / density.max()
    
    # 计算每个区间的统计量
    bin_means = []
    bin_stds = []
    for i in range(len(x_bins)-1):
        mask = (similarities >= x_bins[i]) & (similarities < x_bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(speed_ups[mask]))
            bin_stds.append(np.std(speed_ups[mask]))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    # 绘制热力图
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under('white')
    im = plt.imshow(density.T, origin='lower', aspect='auto',
                   extent=[min(similarities), max(similarities), 
                          min(speed_ups), max(speed_ups)],
                   cmap=cmap,
                   vmin=0.0003,
                   vmax=1)
    
    # 添加颜色条
    plt.colorbar(im, label='Normalized Density')
    
    # 绘制误差线
    bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    valid_mask = ~np.isnan(bin_means)
    bin_means_array = np.array(bin_means)[valid_mask]
    bin_stds_array = np.array(bin_stds)[valid_mask]
    centers = bin_centers[valid_mask]
    
    plt.errorbar(centers, 
                bin_means_array,
                yerr=bin_stds_array,
                color='blue', 
                fmt='o-', 
                markersize=4,
                alpha=0.6,
                label='Mean ± Std')
    
    # 添加统计信息文本框
    stats_text = (
        f'Pearson correlation: {pearson_corr:.3f}\n'
        f'p-value: {p_value:.3e}\n'
        f'Mean speed-up: {np.mean(speed_ups):.2f}\n'
        f'Std speed-up: {np.std(speed_ups):.2f}'
    )
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='top',
             fontsize=9)
    
    # 设置标签和标题
    plt.xlabel("Semantic Similarity Score")
    plt.ylabel("Speed Up Ratio")
    plt.title("Semantic Similarity vs Speed Up\n(Distribution Analysis)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(image_save_path, dpi=300, bbox_inches="tight")
    plt.close()

def compute_cosine_similarity(profiled_path:str,save_path:str,device:str="cuda:0"):
    # tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    from FlagEmbedding import BGEM3FlagModel
    llm_model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=True,devices=device)
    
    profiled_data = json.load(open(profiled_path,"r"))
    for idx,item in tqdm(enumerate(profiled_data),total=len(profiled_data)):
        source_text = item["source_text"]
        target_text = item["target_text"]
        score = llm_model.compute_score([source_text,target_text])
        profiled_data[idx]["cosine_similarity"] = score
    json.dump(profiled_data,open(save_path,"w"),indent=4,ensure_ascii=False)

def plot_three_way_analysis(data_path: str, image_save_path: str = "examples/pipeline/images/three_way_analysis.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore, pearsonr

    # 创建图表布局，为颜色条留出空间
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.3, width_ratios=[1, 1, 1, 0.05])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 3])  # 为颜色条添加一个专门的轴

    # 加载和处理数据
    data = json.load(open(data_path, "r"))
    
    # 提取数据
    speed_ups = []
    similarities = []
    reused_tokens = []
   
    for item in data:
        if item["speed_up_ratio"] < 1.1:
            continue
        reused_token_count = int(item["reuse_ratio"] * item["target_token_len"])
        if reused_token_count > 1024:
            continue
        speed_ups.append(item["speed_up_ratio"])
        similarities.append(item["cosine_similarity"]["colbert"])
        reused_tokens.append(reused_token_count)

    # 转换为numpy数组并过滤异常值
    similarities = np.array(similarities)
    speed_ups = np.array(speed_ups)
    reused_tokens = np.array(reused_tokens)
    
    z_scores = zscore(speed_ups)
    mask = np.abs(z_scores) < 3
    similarities = similarities[mask]
    speed_ups = speed_ups[mask]
    reused_tokens = reused_tokens[mask]

    def create_heatmap_with_stats(ax, x_data, y_data, xlabel, ylabel, title):
        # 计算相关系数
        corr, p_value = pearsonr(x_data, y_data)
        
        # 创建热力图数据
        x_bins = np.linspace(min(x_data), max(x_data), 40)
        y_bins = np.linspace(min(y_data), max(y_data), 40)
        density, _, _ = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
        density = density / density.max()
        
        # 计算每个区间的统计量
        bin_means = []
        bin_stds = []
        for i in range(len(x_bins)-1):
            mask = (x_data >= x_bins[i]) & (x_data < x_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(y_data[mask]))
                bin_stds.append(np.std(y_data[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        # 绘制热力图
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under('white')
        im = ax.imshow(density.T, origin='lower', aspect='auto',
                      extent=[min(x_data), max(x_data), 
                             min(y_data), max(y_data)],
                      cmap=cmap,
                      vmin=0.0003,
                      vmax=1)
        
        # 绘制误差线
        bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        valid_mask = ~np.isnan(bin_means)
        bin_means_array = np.array(bin_means)[valid_mask]
        bin_stds_array = np.array(bin_stds)[valid_mask]
        centers = bin_centers[valid_mask]
        
        ax.errorbar(centers, 
                   bin_means_array,
                   yerr=bin_stds_array,
                   color='blue', 
                   fmt='o-', 
                   markersize=2,
                   alpha=0.6,
                   label='Mean ± Std')
        
        # 添加相关系数
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='top',
                fontsize=8)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        return im

    # 绘制三个热力图
    im1 = create_heatmap_with_stats(ax1, similarities, speed_ups, 
                                   "Cosine Similarity", "Speed Up Ratio",
                                   "Cosine Similarity vs Speed Up")
    
    im2 = create_heatmap_with_stats(ax2, reused_tokens, speed_ups,
                                   "Reused Tokens", "Speed Up Ratio",
                                   "Token Reused Num vs Speed Up")
    
    im3 = create_heatmap_with_stats(ax3, similarities, reused_tokens,
                                   "Cosine Similarity", "Reused Tokens",
                                   "Cosine Similarity vs Token Reused Num")

    # 添加共用的颜色条
    plt.colorbar(im1, cax=cax, label='Normalized Samples Density')

    plt.tight_layout()
    plt.savefig(image_save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"
    max_num = 20000
    save_dir = "examples/pipeline/images/insight1"
    os.makedirs(save_dir,exist_ok=True)
    image1_save_path = os.path.join(save_dir,"reused_tokens_with_speed_up.png")
    image2_save_path = os.path.join(save_dir,"cosine_similar_with_speedup.png")
    image_save_path = os.path.join(save_dir,"three_way_analysis.png")
    # instruction_wildv2_data_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # instruction_wildv2_profiled_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_cosine_similarity_profiled.json"
    # instruction_wildv2_long_token_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_cosine_similarity_long_token.json"
    # select_data(instruction_wildv2_data_path,instruction_wildv2_long_token_path,min_token_len=100)
    
    # compute_reuse_ratio_with_efficiency(model_name,instruction_wildv2_long_token_path,device=device,save_path=instruction_wildv2_profiled_path)
    
    
    shargpt_data_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    
    shargpt_long_token_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity_long_token.json"
    shargpt_profiled_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity_profiled.json"
    # select_data(shargpt_data_path,shargpt_long_token_path,min_token_len=100)
    
    # compute_reuse_ratio_with_efficiency(model_name,shargpt_long_token_path,device=device,save_path=shargpt_save_path)
    # compute_cosine_similarity(shargpt_save_path,shargpt_save_path,device=device)
    # plot_reused_tokens_with_speedup(shargpt_long_token_path,image1_save_path)
    plot_three_way_analysis(shargpt_profiled_path,image_save_path)
    