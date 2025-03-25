import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager 
from tqdm import tqdm
import numpy as np
import random
import json
from typing import List
import time
from multiprocessing import Process, Queue, cpu_count
import queue
from multiprocessing import Pool
import pandas as pd

font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

def get_key_value(model:LLM,prompt: List[str]):
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
    template = "<|im_start|>user\n{prompt}\n<|im_end|>"
    prompt = template.format(prompt=prompt)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    output = model.generate(prompt, sampling_params,use_tqdm=False)
    
    llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    
    past_key_values = []
    num_layer = len(llm_layers)
    for j in range(num_layer):
        hack_kv = llm_layers[j].self_attn.hack_kv
        temp_key_cache = hack_kv[0].clone()
        temp_value_cache = hack_kv[1].clone()
        past_key_values.append(torch.stack([temp_key_cache,temp_value_cache],dim=0))
    past_key_values = torch.stack(past_key_values,dim=0)
    return past_key_values

def clean_text(json_path:str,clean_path:str):
    import hashlib
    from edit2 import apply_change,find_text_differences
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    data = json.load(open(json_path))
    new_data = []
    global_id = 0  # 全局ID计数器
    bin_count = [[] for _ in range(10)] # 统计不同token复用率的paire
    
    # 给每个cluster的每一个meber添加一个独特的整数id
    for key,cluster in tqdm(data["clusters"].items()):
        # hash每一个text
        members = cluster["members"]
        hash_set = set()
        # 给第一个member添加global_id
        global_id += 1
        members[0]["global_id"] = global_id
        
        new_members = [members[0]]
        for item in members[1:]:
            # 找到source_item和target_item的差异
            source_token = tokenizer.encode(members[0]["text"])
            target_token = tokenizer.encode(item["text"])
            diff_report = find_text_differences(tokenizer,source_token,target_token,window_size=3)
            token_reuse_rate = diff_report["summary"]["reuse_ratio"]
            
            hash_value = hashlib.md5(item["text"].encode()).hexdigest()
            if hash_value not in hash_set:
                global_id += 1
                hash_set.add(hash_value)
                item["global_id"] = global_id
                new_members.append(item)
                item["token_reuse_rate"] = token_reuse_rate
                
                bin_count[int((token_reuse_rate-0.01)//10)].append(item)
        # new_data.append(new_members)
    json.dump(bin_count,open(clean_path,"w"),indent=4)

def calculate_token_reuse_rate(pair):
    """
    计算token复用率，使用共享的tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    from edit2 import find_text_differences
    
    source_token = tokenizer.encode(pair["source"])
    target_token = tokenizer.encode(pair["target"])
    diff_report = find_text_differences(tokenizer, source_token, target_token, window_size=3)
    return diff_report["summary"]["reuse_ratio"]


def select_similar_pairs(json_path: str, save_path: str):
    """选择相似文本对并计算复用率"""
    # 加载数据
    data = json.load(open(json_path))
    clusters = data["clusters"]
    unique_pairs = []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # 收集文本对
    print("收集文本对...")
    pair_id = 0
    for key, cluster in tqdm(clusters.items()):
        members = cluster["members"]
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                unique_pairs.append({
                    "pair_id": pair_id,
                    "source": members[i]["text"],
                    "target": members[j]["text"],
                    "source_token": tokenizer.encode(members[i]["text"]),
                    "target_token": tokenizer.encode(members[j]["text"]),
                })
                pair_id += 1
    # 保存结果
    print(f"\n保存结果到 {save_path}")
    with open(save_path, "w") as f:
        json.dump(unique_pairs, f, indent=4)
    
       

def compute_similarity(clean_path:str,save_path:str):
    """
    计算相似度
    """
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cuda:1")
    data = json.load(open(clean_path))
    for index,pair in tqdm(enumerate(data)):
        source_embedding = model.encode(pair["source"])
        target_embedding = model.encode(pair["target"])
        similarity = np.dot(source_embedding, target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
        pair["cosine_similarity"] = float(similarity)
    json.dump(data,open(save_path,"w"),indent=4)


            
def calculate_window_sizes(pair_data):
    """
    计算单个pair在不同窗口大小下的复用率
    """
    from edit2 import find_text_differences
    try:
        # 确保token是列表类型
        source_token = pair_data["source_token"] if isinstance(pair_data["source_token"], list) else [pair_data["source_token"]]
        target_token = pair_data["target_token"] if isinstance(pair_data["target_token"], list) else [pair_data["target_token"]]
        pair_id = pair_data["pair_id"]
        window_results = []
        
        for window_size in [1, 2, 3, 4, 5]:
            if window_size > len(source_token) or window_size > len(target_token):
                continue
            diff_report = find_text_differences(source_token, target_token, window_size=window_size)
            window_results.append([window_size, float(diff_report["summary"]["reuse_ratio"])])
        
        return pair_id, window_results
        
    except Exception as e:
        print(f"Error processing pair {pair_data.get('pair_id', 'unknown')}: {str(e)}")
        return pair_id, []

def process_batch(batch_data):
    """处理一批数据的函数"""
    from edit2 import find_text_differences, apply_change
    import torch
    
    results = []
    for pair_data in batch_data:
        try:
            source_token = pair_data["source_token"]
            target_token = pair_data["target_token"]
            window_results = []
            
            for window_size in [1,2,3,4,5]:
                if window_size >= len(source_token) or window_size >= len(target_token):
                    continue
                start_time = time.time()
                diff_report = find_text_differences(source_token, target_token, window_size=window_size)
                source_kvcache = torch.zeros(4,2,len(source_token),1)
                target_kvcache, reused_map_indices, unused_map_indices = apply_change(
                    source_token, target_token, source_kvcache, diff_report
                )
                end_time = time.time()
                reuse_rate = len(reused_map_indices) / len(target_token)
                window_results.append([window_size, reuse_rate, (end_time-start_time)*1000])
            
            results.append({
                "pair_id": pair_data.get("pair_id", -1),
                "window_size": window_results,
                "reuse_rate": reuse_rate if window_results else 0
            })
            
        except Exception as e:
            print(f"Error processing pair {pair_data.get('pair_id', 'unknown')}: {str(e)}")
            continue
            
    return results

def get_windowsize_similarity(data_path: str, save_path: str):
    """使用进程池处理不同窗口大小的计算"""
    # 加载数据
    print("加载数据...")
    data = json.load(open(data_path))
    
    # 设置进程数和批次大小
    num_processes = min(128, cpu_count())
    batch_size = max(1, len(data) // (num_processes * 2))  # 每个进程处理多个批次
    print(f"\n使用{num_processes}个进程进行计算，每批处理{batch_size}个数据对...")
    
    # 将数据分成批次
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    print(f"总共分成{len(batches)}个批次")
    
    # 创建进程池处理批次数据
    all_results = []
    with Pool(processes=num_processes) as pool:
        # 使用imap处理批次并显示进度条
        for batch_results in tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc="处理数据批次"
        ):
            all_results.extend(batch_results)
    
    # 将结果写回原始数据
    print("\n整理结果...")
    result_dict = {r["pair_id"]: r for r in all_results if r is not None}
    for item in data:
        pair_id = item.get("pair_id", -1)
        if pair_id in result_dict:
            item["window_size"] = result_dict[pair_id]["window_size"]
            item["reuse_rate"] = result_dict[pair_id]["reuse_rate"]
    
    # 保存结果
    print(f"保存结果到 {save_path}")
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    
    # 打印统计信息
    print("\n=== 处理完成 ===")
    print(f"总数据量: {len(data)}")
    print(f"成功处理: {len(all_results)}")
    print(f"失败数量: {len(data) - len(all_results)}")
    
    # 计算平均处理时间
    total_time = sum(
        sum(w[2] for w in r["window_size"]) 
        for r in all_results if r["window_size"]
    )
    avg_time = total_time / len(all_results) if all_results else 0
    print(f"平均处理时间: {avg_time:.2f}ms/对")


def plot_similarity_reuse_rate(data):
    # 计算行数和列数
    n_datasets = len(data)
    n_rows = (n_datasets + 1) // 2  # 向上取整，确保能容纳所有图表
    n_cols = 2
    
    # 创建图表，调整大小以适应新布局
    fig = plt.figure(figsize=(10, 4*n_rows))  # 每行高度4英寸，总宽度10英寸
    
    # 创建网格规范，最后一列预留给颜色带
    gs = plt.GridSpec(n_rows, n_cols + 1, width_ratios=[10, 10, 0.5])
    
    axes = []
    # 创建子图
    for i in range(n_datasets):
        row = i // 2  # 确定行号
        col = i % 2   # 确定列号
        axes.append(fig.add_subplot(gs[row, col]))

    def analyze_relationship(points_x, points_y):
        """分析复用率和相似度的关系"""
        # 1. 计算相关系数
        correlation = np.corrcoef(points_x, points_y)[0,1]
        
        # 2. 将数据分成网格进行密度分析
        x_bins = np.linspace(0, 1, 40)
        y_bins = np.linspace(0, 1, 40)
        density, _, _ = np.histogram2d(points_x, points_y, bins=[x_bins, y_bins])
        
        # 归一化密度
        density = density / density.max()
        
        # 3. 计算条件统计量
        bin_means = []
        bin_stds = []
        for i in range(len(x_bins)-1):
            mask = (points_x >= x_bins[i]) & (points_x < x_bins[i+1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(points_y[mask]))
                bin_stds.append(np.std(points_y[mask]))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        return correlation, density, bin_means, bin_stds, x_bins, y_bins

    def plot_data(ax, data_path, tag, color, sim_threshold=0.6, reuse_threshold=0.6):
        points_x = []
        points_y = []
        data = json.load(open(data_path))
        for pair in tqdm(data):
            if len(pair["window_size"]) == 0:
                continue
            window_size, reuse_rate, time = pair["window_size"][0]
            similarity = pair["cosine_similarity"]
            points_x.append(reuse_rate)
            points_y.append(similarity)
        
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        
        # 分析关系并绘制热力图
        correlation, density, bin_means, bin_stds, x_bins, y_bins = analyze_relationship(points_x, points_y)
        
        # 创建自定义颜色映射
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under('white')  # 设置低于vmin的值为白色
        
        # 设置阈值，低于此值的将显示为白色
        threshold = 0.0003
        
        # 绘制热力图
        im = ax.imshow(density.T, origin='lower', aspect='auto',
                      extent=[0, 1, 0, 1], 
                      cmap=cmap,
                      vmin=threshold,  # 设置最小阈值
                      vmax=1)
        
        # 分别绘制阈值上下的误差线
        bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        valid_mask = ~np.isnan(bin_means)
        bin_means_array = np.array(bin_means)[valid_mask]
        bin_stds_array = np.array(bin_stds)[valid_mask]
        centers = bin_centers[valid_mask]
        
        # 计算不同区域的相关性
        # 高质量区域：同时满足两个阈值
        high_quality_mask = (points_x >= reuse_threshold) & (points_y >= sim_threshold)
        high_quality_corr = np.corrcoef(
            points_x[high_quality_mask], 
            points_y[high_quality_mask]
        )[0,1] if np.any(high_quality_mask) else 0
        
        # 低质量区域：不满足任一阈值
        low_quality_mask = ~high_quality_mask
        low_quality_corr = np.corrcoef(
            points_x[low_quality_mask], 
            points_y[low_quality_mask]
        )[0,1] if np.any(low_quality_mask) else 0
        
        # 在右下角添加相关系数信息
        correlation_text = (
            f"Correlation:\n"
            f"High quality: {high_quality_corr:.2f}\n"
            f"Low quality: {low_quality_corr:.2f}"
        )
        
        ax.text(0.67, 0.03, correlation_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(facecolor='white', 
                         edgecolor='gray',
                         alpha=0.8,
                         pad=5),
                fontsize=10,
                linespacing=1.5)
        
        # 修改误差线的英文图例说明
        ax.errorbar(centers, 
                   bin_means_array,
                   yerr=bin_stds_array,
                   color=color, fmt='o-', markersize=4,
                   alpha=0.3,
                   label='Mean ± Std (Low quality)')
        
        # 找到同时满足两个阈值的点
        high_quality_mask = (centers >= reuse_threshold) & (bin_means_array >= sim_threshold)
        
        # 绘制高质量区域的误差线（高透明度）
        if np.any(high_quality_mask):
            ax.errorbar(centers[high_quality_mask],
                       bin_means_array[high_quality_mask],
                       yerr=bin_stds_array[high_quality_mask],
                       color=color, fmt='o-', markersize=4,
                       alpha=1.0,
                       label='Mean ± Std (High quality)')
        
        # 添加阈值线
        # ax.axhline(y=sim_threshold, color='red', linestyle='--', alpha=0.8)
        # ax.axvline(x=reuse_threshold, color='red', linestyle='--', alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(f"{tag}")
        ax.set_xlabel("Token Reuse Ratio")
        ax.set_ylabel("Cosine Similarity")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        
        return im

    # 为每个数据集创建图表，并保存最后一个热力图对象用于创建颜色带
    last_im = None
    for idx, item in enumerate(data):
        last_im = plot_data(axes[idx], item["path"], item["tag"], item["color"], 
                           item["sim_threshold"], item["reuse_threshold"])
    
    # 添加共享的颜色带，跨越所有行
    cbar_ax = fig.add_subplot(gs[:, -1])
    plt.colorbar(last_im, cax=cbar_ax, label='Normalized Density')
    
    plt.tight_layout()
    plt.savefig("examples/pipeline/images/similarity_reuse_rate_analysis.png",
                dpi=300, bbox_inches="tight")
    plt.close()

def plot_window_size_reuse_rate(data_path:str):
    """
    绘制不同窗口大小的token复用率范围
    """
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 收集数据
    window_reuse_rates = [[] for _ in range(5)]
    data = json.load(open(data_path))
    
    for pair in tqdm(data):
        if len(pair["window_size"]) == 0:
            continue
        for window_data in pair["window_size"]:
            window_size, reuse_rate, _ = window_data
            window_reuse_rates[window_size-1].append(reuse_rate)
    # 绘制箱线图
    box_data = [rates for rates in window_reuse_rates if rates]  # 过滤掉空列表
    ax1.boxplot(box_data, labels=[f'w={i+1}' for i in range(len(box_data))])
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('Reuse Rate (%)')
    ax1.set_title('Distribution of Reuse Rates by Window Size (Box Plot)')
    ax1.grid(True, alpha=0.3)
    
    # 绘制小提琴图
    import seaborn as sns
    # 准备数据为长格式
    violin_data = []
    for window_size, rates in enumerate(window_reuse_rates, 1):
        violin_data.extend([(window_size, rate) for rate in rates])
    
    violin_df = pd.DataFrame(violin_data, columns=['Window Size', 'Reuse Rate'])
    sns.violinplot(data=violin_df, x='Window Size', y='Reuse Rate', ax=ax2)
    ax2.set_title('Distribution of Reuse Rates by Window Size (Violin Plot)')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    # stats_text = "Statistical Summary:\n"
    # for window_size, rates in enumerate(window_reuse_rates, 1):
    #     if rates:
    #         stats_text += f"\nWindow Size {window_size}:\n"
    #         stats_text += f"Mean: {np.mean(rates):.2f}%\n"
    #         stats_text += f"Median: {np.median(rates):.2f}%\n"
    #         stats_text += f"Std: {np.std(rates):.2f}%\n"
    #         stats_text += f"25th: {np.percentile(rates, 25):.2f}%\n"
    #         stats_text += f"75th: {np.percentile(rates, 75):.2f}%\n"
    
    # 在图形下方添加统计信息
    # plt.figtext(0.1, -0.2, stats_text, fontsize=10, ha='left', va='top')
    
    plt.tight_layout()
    plt.savefig("examples/pipeline/images/window_size_reuse_rate.png", 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()

def plot_window_size_time(data_path:str):
    """
    绘制不同窗口大小的KV编辑时间和token长度的关系
    """
    color = [
        "blue",
        "red",
        "green",
        "yellow",
        "purple",
        "orange",
        "brown"
    ]
    plt.figure(figsize=(15, 10))
    # data = json.load(open(data_path))
    # for pair in tqdm(data):
    #     if len(pair["window_size"]) == 0:
    #         continue
    #     points_x = []
    #     points_y = []
    #     for window_data in pair["window_size"]:
    #         window_size, reuse_rate, time = window_data 
    #         points_x.append(window_size)
    #         points_y.append(time)
    #     plt.scatter(points_x,points_y,s=10,c=color,alpha=0.7,label=f"window_size={pair['window_size']}")
    


if __name__ == "__main__":
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # raw_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,save_path)
    # get_windowsize_similarity(sim_reuse_path,sim_reuse_path)
    # plot_similarity_reuse_rate(sim_reuse_path)
    # plot_window_size_reuse_rate(sim_reuse_path)
    # raw_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(sim_reuse_path,sim_reuse_path)
    
    # raw_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    
    
    # raw_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    
    # raw_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    
    
    data = [
        {
            "tag": "InstructionWild v2",
            "path": "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
            "color": "blue",
            "sim_threshold": 0.49,
            "reuse_threshold": 0.52
        },
        {
            "tag": "ShareGPT-90k",
            "path": "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
            "color": "blue",
            "sim_threshold": 0.4,
            "reuse_threshold": 0.3
        },
        {
            "tag": "LMSysChat-1M",
            "path": "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
            "color": "blue",
            "sim_threshold": 0.4,
            "reuse_threshold": 0.44
        }
    ]
    plot_similarity_reuse_rate(data)