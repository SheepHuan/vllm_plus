import os
from vllm import LLM, SamplingParams, RequestOutput
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

import random
def process_cluster(args):
    """处理单个聚类的文本对生成
    
    Args:
        args: (cluster_id, members, start_pair_id) 元组
        
    Returns:
        pairs: 该聚类生成的文本对列表
    """
    cluster_id, members, start_pair_id,max_size = args
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    pairs = []
    pair_id = start_pair_id
    if len(members) > max_size:
        members = random.sample(members,max_size)
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                pairs.append({
                        "pair_id": pair_id,
                        "source": members[i]["text"],
                        "target": members[j]["text"],
                        "source_token": tokenizer.encode(members[i]["text"]),
                        "target_token": tokenizer.encode(members[j]["text"]),
                    "cluster_id": cluster_id
                    })
                pair_id += 1
            
    return pairs

def select_similar_pairs(json_path: str, save_path: str,max_size:int=100):
    """从聚类结果中选择相似文本对，使用多进程优化
    
    Args:
        json_path: 聚类结果文件路径
        save_path: 相似对保存路径
    """
    # 加载数据
    print("加载数据...")
    data = json.load(open(json_path))
    clusters = data["clusters"]
    
    # 准备多进程参数
    num_processes = min(64, len(clusters))
    print(f"\n使用{num_processes}个进程处理{len(clusters)}个聚类...")
    
    # 计算每个聚类的pair_id起始值
    start_pair_ids = []
    current_id = 0
    for key, cluster in clusters.items():
        members = cluster["members"]
        size = min(len(members),max_size)
        num_pairs = (size * (size - 1)) // 2
        start_pair_ids.append(current_id)
        current_id += num_pairs
    
    # 准备进程池参数
    process_args = [
        (key, cluster["members"], start_id,max_size) 
        for (key, cluster), start_id in zip(clusters.items(), start_pair_ids)
    ]
    
    # 使用进程池处理
    all_pairs = []
    with Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        for cluster_pairs in tqdm(
            pool.imap(process_cluster, process_args),
            total=len(clusters),
            desc="处理聚类"
        ):
            all_pairs.extend(cluster_pairs)
    
    # 按pair_id排序
    all_pairs.sort(key=lambda x: x["pair_id"])
    
    # 输出统计信息
    print(f"\n处理完成:")
    print(f"- 总聚类数: {len(clusters)}")
    print(f"- 总文本对数: {len(all_pairs)}")
    
    # 保存结果
    print(f"\n保存结果到 {save_path}")
    with open(save_path, "w") as f:
        json.dump(all_pairs, f, indent=4)
    
def compute_similarity(clean_path:str, save_path:str, batch_size:int=64, model_name:str="sentence-transformers/all-MiniLM-L6-v2", device:str="cuda:1"):
    """计算文本对之间的余弦相似度
    
    Args:
        clean_path: 输入的清洗后数据路径
        save_path: 结果保存路径
        batch_size: 批处理大小，默认64
    """
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    data = json.load(open(clean_path))
    # data = data[:1000]
    # 批量处理数据
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]
        
        # 收集当前批次的source和target文本
        batch_sources = [pair["source"] for pair in batch_data]
        batch_targets = [pair["target"] for pair in batch_data]
        
        # 批量计算嵌入向量
        embeddings = model.encode(batch_sources+batch_targets)
        source_embeddings = embeddings[:len(batch_sources)]
        target_embeddings = embeddings[len(batch_sources):]
        
        # all_similarity = np.dot(source_embeddings,target_embeddings.T) / (np.linalg.norm(source_embeddings,axis=1)[:,None] * np.linalg.norm(target_embeddings,axis=1)[None,:])
        
        for j, pair in enumerate(batch_data):
            # 计算单个pair的余弦相似度
            similarity = np.dot(source_embeddings[j], target_embeddings[j]) / (
                np.linalg.norm(source_embeddings[j]) * np.linalg.norm(target_embeddings[j])
            )
        pair["cosine_similarity"] = float(similarity)
    
    
    print(f"\n保存结果到 {save_path}")
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


            
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
    """处理单个数据批次
    
    Args:
        batch_data: 包含多个文本对的批次数据
    
    Returns:
        results: 包含处理结果的列表，每个元素包含:
            - pair_id: 文本对ID
            - window_size: 不同窗口大小的结果
            - reuse_rate: token复用率
    """
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

def get_windowsize_diff_and_reuse(data_path: str, save_path: str):
    """使用多进程计算不同窗口大小下的相似度和复用率
    
    Args:
        data_path: 输入数据文件路径
        save_path: 结果保存路径
    
    处理流程:
        1. 将数据分成多个批次
        2. 使用进程池并行处理
        3. 合并结果并保存
    """
    # 加载数据
    print("加载数据...")
    import ijson
    with open(data_path, "r") as f:
        for item in ijson.items(f, "high_quality_pairs"):
            pass
    
    data = json.load(open(data_path))
    # data = data["high_quality_pairs"]
    
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


def plot_similarity_reuse_rate(data,save_path:str):
    """创建多子图布局，展示多个数据集的复用率-相似度关系
    
    Args:
        data: 包含多个数据集信息的列表，每个元素需包含:
            - tag: 数据集标签
            - path: 数据文件路径
            - color: 绘图颜色
            - sim_threshold: 相似度阈值
            - reuse_threshold: 复用率阈值
    """
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
        """分析复用率和相似度之间的关系
        
        Args:
            points_x: 复用率数据点数组
            points_y: 相似度数据点数组
        
        Returns:
            correlation: 相关系数
            density: 二维密度矩阵
            bin_means: 每个bin的平均值
            bin_stds: 每个bin的标准差
            x_bins: x轴的bin边界
            y_bins: y轴的bin边界
        """
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
        """在指定子图上绘制热力图和统计信息
        
        Args:
            ax: matplotlib子图对象
            data_path: 数据文件路径
            tag: 数据集标签
            color: 误差线颜色
            sim_threshold: 相似度阈值
            reuse_threshold: 复用率阈值
        
        Returns:
            im: 热力图对象，用于创建颜色条
        """
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
    plt.savefig(save_path,
                dpi=300, bbox_inches="tight")
    plt.close()

def plot_window_size_reuse_rate(data_path:str):
    """分析不同窗口大小下的token复用率分布
    
    Args:
        data_path: 数据文件路径
    
    绘制两种可视化:
        1. 箱线图：显示各窗口大小下复用率的统计分布
        2. 小提琴图：显示复用率的概率密度分布
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
    
def get_key_value(model:LLM,prompt: List[str]):
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    model.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
    template = "<|im_start|>user\n{prompt}\n<|im_end|>"
    prompt = template.format(prompt=prompt)
    
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


    
def select_high_correlation_pairs(data_path: str, high_correlation_path: str = None, low_correlation_path: str = None, reuse_threshold: float = None, sim_threshold: float = None):
    """选择高相关性和低相关性文本对
    
    Args:
        data_path: 包含相似文本对的数据文件路径
        save_path: 高相关性文本对保存路径
        low_correlation_path: 低相关性文本对保存路径，如果为None则不保存
        reuse_threshold: 复用率阈值，如果为None则自动计算
        sim_threshold: 相似度阈值，如果为None则自动计算
    """
    # 加载数据
    all_pairs = json.load(open(data_path))
    reuse_ratios = []
    cosine_similarities = []
    valid_pairs = []
    
    for pair in all_pairs:
        if len(pair["window_size"]) == 0:
            continue
        reuse_ratios.append(pair["window_size"][0][1])
        cosine_similarities.append(pair["cosine_similarity"])
        valid_pairs.append(pair)
        
    reuse_ratios = np.array(reuse_ratios)
    cosine_similarities = np.array(cosine_similarities)
    
    # 如果没有提供阈值，则计算最优阈值
    if reuse_threshold is None or sim_threshold is None:
        def calculate_score(reuse_th, sim_th):
            """计算阈值组合的得分"""
            high_quality_mask = (reuse_ratios >= reuse_th) & (cosine_similarities >= sim_th)
            high_quality_count = np.sum(high_quality_mask)
            
            if high_quality_count < 100:  # 确保有足够的样本
                return float('-inf')
                
            high_quality_corr = np.corrcoef(
                reuse_ratios[high_quality_mask], 
                cosine_similarities[high_quality_mask]
            )[0, 1]
            
            # 计算低质量区域的相关性
            low_quality_mask = ~high_quality_mask
            low_quality_count = np.sum(low_quality_mask)
            
            if low_quality_count < 100:  # 确保有足够的样本
                return float('-inf')
                
            low_quality_corr = np.corrcoef(
                reuse_ratios[low_quality_mask], 
                cosine_similarities[low_quality_mask]
            )[0, 1]
            
            # 设置目标要求
            if high_quality_corr < 0.6:  # 基本要求
                return float('-inf')
            
            # 主要关注高质量区域的相关性
            score = (5 * high_quality_corr) - abs(low_quality_corr)
            return score
        
        # 网格搜索最优阈值
        best_score = float('-inf')
        best_reuse_th = 0
        best_sim_th = 0
        
        reuse_thresholds = np.linspace(0.3, 0.8, 50)  # 增加搜索精度
        sim_thresholds = np.linspace(0.3, 0.8, 50)    # 增加搜索精度
        
        print("搜索最优阈值组合...")
        for reuse_th in tqdm(reuse_thresholds):
            for sim_th in sim_thresholds:
                score = calculate_score(reuse_th, sim_th)
                if score > best_score:
                    best_score = score
                    best_reuse_th = reuse_th
                    best_sim_th = sim_th
                    
        reuse_threshold = best_reuse_th
        sim_threshold = best_sim_th
        print(f"\n找到的最优阈值:")
        print(f"- Reuse threshold: {reuse_threshold:.3f}")
        print(f"- Similarity threshold: {sim_threshold:.3f}")
    else:
        print(f"\n使用提供的阈值:")
        print(f"- Reuse threshold: {reuse_threshold:.3f}")
        print(f"- Similarity threshold: {sim_threshold:.3f}")
    
    # 使用阈值筛选数据对
    final_mask = (reuse_ratios >= reuse_threshold) & (cosine_similarities >= sim_threshold)
    high_quality_pairs = []
    low_quality_pairs = []
    
    # 为每个数据对添加额外信息
    for pair, mask in zip(valid_pairs, final_mask):
        pair_info = {
            "source": pair["source"],
            "target": pair["target"],
            "source_token": pair["source_token"],
            "target_token": pair["target_token"],
            "cosine_similarity": pair["cosine_similarity"],
            "reuse_rate": pair["window_size"][0][1],  # 使用window_size=1的复用率
            "pair_id": pair["pair_id"]
        }
        if mask:
            high_quality_pairs.append(pair_info)
        else:
            low_quality_pairs.append(pair_info)
    
    # 计算最终的相关性指标
    high_quality_mask = (reuse_ratios >= reuse_threshold) & (cosine_similarities >= sim_threshold)
    low_quality_mask = ~high_quality_mask
    
    high_quality_corr = np.corrcoef(
        reuse_ratios[high_quality_mask], 
        cosine_similarities[high_quality_mask]
    )[0, 1]
    
    low_quality_corr = np.corrcoef(
        reuse_ratios[low_quality_mask], 
        cosine_similarities[low_quality_mask]
    )[0, 1]
    
    # 输出统计信息
    print(f"\n找到的最优阈值:")
    print(f"- Reuse threshold: {reuse_threshold:.3f}")
    print(f"- Similarity threshold: {sim_threshold:.3f}")
    print(f"\n相关性分析:")
    print(f"- 高质量区域相关性: {high_quality_corr:.3f}")
    print(f"- 低质量区域相关性: {low_quality_corr:.3f}")
    print(f"- 高质量数据对数量: {len(high_quality_pairs)}")
    print(f"- 总数据对数量: {len(valid_pairs)}")
    print(f"- 高质量数据比例: {len(high_quality_pairs)/len(valid_pairs):.2%}")
    
   
    if high_correlation_path:
        # 保存高质量数据对结果
        high_quality_result = {
            "metadata": {
                "thresholds": {
                    "reuse_threshold": float(reuse_threshold),
                    "sim_threshold": float(sim_threshold)
                },
                "statistics": {
                    "high_quality_correlation": float(high_quality_corr),
                    "low_quality_correlation": float(low_quality_corr),
                    "high_quality_pairs_count": len(high_quality_pairs),
                    "total_pairs_count": len(valid_pairs),
                    "high_quality_ratio": float(len(high_quality_pairs)/len(valid_pairs))
                }
            },
            "high_quality_pairs": high_quality_pairs
        }
        
        print(f"\n保存高质量数据对到: {high_correlation_path}")
        with open(high_correlation_path, "w") as f:
            json.dump(high_quality_result, f, indent=4)
    
    # 如果指定了低质量数据保存路径，则保存低质量数据对
    if low_correlation_path:
        low_quality_result = {
            "metadata": {
                "thresholds": {
                    "reuse_threshold": float(reuse_threshold),
                    "sim_threshold": float(sim_threshold)
                },
                "statistics": {
                    "high_quality_correlation": float(high_quality_corr),
                    "low_quality_correlation": float(low_quality_corr),
                    "low_quality_pairs_count": len(low_quality_pairs),
                    "total_pairs_count": len(valid_pairs),
                    "low_quality_ratio": float(len(low_quality_pairs)/len(valid_pairs))
                }
            },
            "low_quality_pairs": low_quality_pairs
        }
        
        print(f"保存低质量数据对到: {low_correlation_path}")
        with open(low_correlation_path, "w") as f:
            json.dump(low_quality_result, f, indent=4)

def compute_kverr_between_high_correlation(data_path:str,save_path:str,device:str="cuda:0",model_name:str="Qwen/Qwen2.5-7B-Instruct",use_modelscope:bool=False):
    """计算高相关性文本对之间的KV编辑错误率
    
    Args:
        data_path: 包含相似文本对的数据文件路径
    
    """
    if use_modelscope:
        os.environ["VLLM_USE_MODELSCOPE"] = True
    from edit2 import apply_change,find_text_differences
    model = LLM(model=model_name,
                device=device,
                dtype="bfloat16"
                )
    template = "<|im_start|>user\n{prompt}\n<|im_end|>"
    paris = json.load(open(data_path))["high_quality_pairs"]
    paris = random.sample(paris,min(len(paris),10000))
    kverr_data = []
    for pair in tqdm(paris,desc="计算KV误差"):
        try:
            source_prompt = template.format(prompt=pair["source"])
            target_prompt = template.format(prompt=pair["target"])
            source_kv,source_token = get_key_value(model,source_prompt)
            target_kv,target_token = get_key_value(model,target_prompt)
        # 计算编辑KV，然后组合得到KV
            diff_report = find_text_differences(source_token,target_token,window_size=1)
            modified_kv,reused_map_indices,_ = apply_change(source_token,target_token,source_kv,diff_report)
            
            target_kv = target_kv[:,:,reused_map_indices,:]
            modified_kv = modified_kv[:,:,reused_map_indices,:]
            
            # 计算不同层的KV误差值
            kv_err = torch.abs(target_kv - modified_kv)
            # 区分key，value
            key_err = kv_err[:,0,:,:]
            value_err= kv_err[:,1,:,:]
            # 计算每层KV误差值的平均值
            key_err_layer_mean = key_err.mean(dim=[1,2])
            value_err_layer_mean = value_err.mean(dim=[1,2])
            # 计算每层KV误差值的方差
            key_err_layer_std = key_err.std(dim=[1,2])
            value_err_layer_std = value_err.std(dim=[1,2])
            # 计算每层KV误差值的L2范数
            key_err_layer_l2 = key_err.norm(dim=[1,2])
            value_err_layer_l2 = value_err.norm(dim=[1,2])
            
            kverr_data.append({
                "source":pair["source"],
                "target":pair["target"],
                "source_token":source_token,
                "target_token":target_token,
                "key_err_layer_mean":key_err_layer_mean.tolist(),
                "value_err_layer_mean":value_err_layer_mean.tolist(),
                "key_err_layer_std":key_err_layer_std.tolist(),
                "value_err_layer_std":value_err_layer_std.tolist(),
                "key_err_layer_l2":key_err_layer_l2.tolist(),
                "value_err_layer_l2":value_err_layer_l2.tolist(),
                "cosine_similarity":pair["cosine_similarity"],
                "reuse_rate":pair["reuse_rate"],
            })
        except Exception as e:
            print(f"计算KV误差时出错: {e}")
            continue
    with open(save_path,"w") as f:
        json.dump(kverr_data,f,indent=4)

def plot_kverr_distribution(data_list: List[dict], show_full_reuse: bool = True):
    """绘制多个数据集的KV缓存误差分布与token重用率的关系
    
    Args:
        data_list: 包含多个数据集信息的列表，每个元素需包含:
            - tag: 数据集标签
            - path: 数据文件路径
            - color: 绘图颜色
        show_full_reuse: 是否显示重用率为100%的数据点
    """
    import seaborn as sns
    
    # 计算子图布局
    n_datasets = len(data_list)
    n_rows = (n_datasets + 1) // 2  # 向上取整
    n_cols = min(2, n_datasets)  # 最多2列
    
    # 创建图表
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 8*n_rows))
    if n_datasets == 1:
        axes = np.array([axes])  # 确保axes是数组
    
    # 处理每个数据集
    for idx, item in enumerate(data_list):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # 加载数据
        data = json.load(open(item["path"]))
        all_data = []
        
        # 收集数据
        for entry in data:
            reuse_rate = entry["reuse_rate"]
            if not show_full_reuse and reuse_rate >= 0.99:
                continue
                
            for layer_id in range(28):  # 假设28层
                # 添加key误差数据
                all_data.append({
                    "layer_id": layer_id,
                    "reuse_rate": reuse_rate,
                    "error_value": abs(entry["key_err_layer_mean"][layer_id]),
                    "error_type": "Key Error"
                })
                
                # 添加value误差数据
                all_data.append({
                    "layer_id": layer_id,
                    "reuse_rate": reuse_rate,
                    "error_value": abs(entry["value_err_layer_mean"][layer_id]),
                    "error_type": "Value Error"
                })
        
        df = pd.DataFrame(all_data)
        
        # 计算每个重用率下的平均误差
        mean_errors = df.groupby(["reuse_rate", "error_type"])["error_value"].mean().reset_index()
        
        # 设置颜色
        colors = {
            "Key Error": {
                "scatter": "blue",  # 淡蓝色散点
                "line": "darkblue"       # 深蓝色趋势线
            },
            "Value Error": {
                "scatter": "orange",     # 淡橙色散点
                "line": "darkorange"     # 深橙色趋势线
            }
        }
        
        # 绘制散点图和趋势线
        for err_type in ["Key Error", "Value Error"]:
            err_data = mean_errors[mean_errors["error_type"] == err_type]
            # 绘制散点图
            sns.scatterplot(data=err_data,
                          x="reuse_rate", y="error_value",
                          alpha=0.5, label=err_type,
                          color=colors[err_type]["scatter"],
                          ax=ax)
            # 绘制趋势线
            sns.regplot(data=err_data,
                      x="reuse_rate", y="error_value",
                      scatter=False,
                      color=colors[err_type]["line"],
                      ax=ax)
        
        # 设置标题和标签
        title = f"{item['tag']}\nAverage KV Error vs Token Reuse Rate"
        if not show_full_reuse:
            title += " (Excluding 100% Reuse)"
        ax.set_title(title)
        ax.set_xlabel("Token Reuse Rate")
        ax.set_ylabel("Average Error")
        ax.grid(True, alpha=0.3)
        
        # 设置对数刻度
        ax.set_yscale('log')
        
    # 如果数据集数量为奇数，删除多余的子图
    if n_datasets % 2 == 1 and n_datasets > 1:
        fig.delaxes(axes[n_rows-1, 1])
    
    plt.tight_layout()
    plt.savefig(f"examples/pipeline/images/kv_error_analysis{'_no_full_reuse' if not show_full_reuse else ''}.png", 
                dpi=300, bbox_inches="tight")
    plt.close()

def analyze_low_correlation(data_path:str, save_path:str,device = "cuda:1", test_1:bool=False, test_2:bool=False, test_3:bool=False):
    from sentence_transformers import SentenceTransformer
    """
    分析低相关性文本对的问题。
    1. 高相似度，低重用率是否是因为，相似度计算错误，导致相似度很高。
    2. 高相似度，低重用率是否是因为，文本本身存在大小写问题，例如近义词较多、大小写问题，导致相似度高？
    
    """
    
    data = json.load(open(data_path))
    low_correlation_data = data["low_quality_pairs"]
    new_data = []
    if test_1:
        # 判断相似度误差问题
        model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct",local_files_only=True,device=device).to(torch.bfloat16)
        for pair in tqdm(low_correlation_data,desc="计算GTE-Qwen2-7B相似度"):
            source_embedding = model.encode(pair["source"])
            target_embedding = model.encode(pair["target"])
            cosine_similarity = np.dot(source_embedding, target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
            pair["cosine_similarity_by_gte_qwen2_7b"] =  float(cosine_similarity)
            new_data.append(pair)
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    if test_2:
        from edit2 import find_text_differences,apply_change
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True,device=device).to(torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        # 判断存在大小写的问题
        for pair in tqdm(low_correlation_data,desc="大小写判断"):
            source_text = pair["source"].lower()
            target_text = pair["target"].lower()
            source_embedding = model.encode(source_text)
            target_embedding = model.encode(target_text)
            cosine_similarity = np.dot(source_embedding, target_embedding) / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
            pair["cosine_similarity_after_lowercase"] =  float(cosine_similarity)
            source_token = tokenizer.encode(source_text)
            target_token = tokenizer.encode(target_text)
            diff_report = find_text_differences(source_token,target_token,window_size=1)
            source_kv = torch.zeros(28,2,len(source_token),1,dtype=torch.bfloat16)
            modified_token,reused_map_indices,_ = apply_change(source_token,target_token,source_kv,diff_report)
            pair["reuse_rate_after_lowercase"] = len(reused_map_indices)/len(source_token)
            new_data.append(pair)
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    if test_3:
        # 判断近义词问题
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True,device=device)
        tokenizer =  model.tokenizer
        
        for pair in tqdm(low_correlation_data,desc="近义词判断"):
            source_text = pair["source"].lower()
            target_text = pair["target"].lower()
            source_token = tokenizer(source_text).input_ids
            target_token = tokenizer(target_text).input_ids
            # 获取token embeddings (n×384和m×384)
            source_token_embedding = model.encode(source_text,output_value="token_embeddings").cpu().numpy()  # n x 384
            target_token_embedding = model.encode(target_text,output_value="token_embeddings").cpu().numpy()  # m x 384
            
            # 计算token之间的余弦相似度矩阵 (n×m)
            similarity_matrix = np.dot(source_token_embedding, target_token_embedding.T) / (
                np.linalg.norm(source_token_embedding, axis=1)[:, np.newaxis] * 
                np.linalg.norm(target_token_embedding, axis=1)[np.newaxis, :]
            )
            
            # 设定相似度阈值（可以根据需要调整）
            similarity_threshold = 0.7 
            
            # 找到相似度高于阈值的token对
            similar_token_pairs = []
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    if similarity_matrix[i, j] >= similarity_threshold and source_token[i] != target_token[j]:
                        similar_token_pairs.append({
                            # "source_token_id": source_token[i],
                            # "target_token_id": target_token[j],
                            "source_token": tokenizer.decode(source_token[i]),
                            "target_token": tokenizer.decode(target_token[j]),
                            "similarity": float(similarity_matrix[i, j])
                        })
            
            # 计算token级别的复用率
           
            
            # 添加到pair信息中
            pair["similar_token_pairs"] =  similar_token_pairs
            
            new_data.append(pair)
            
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

    if test_1 or test_2 or test_3:
        json.dump(new_data,open(save_path,"w"),indent=4,ensure_ascii=False)

def plot_kverr_with_resue_segment_length(data_path:str,save_path:str):
    """
    统计token复用片段长度和KV缓存误差的关系
    
    """
    data = json.load(open(data_path))
    for pair in data:
        reuse_segment_length = pair["reuse_segment_length"]
        kverr = pair["kverr"]
        plt.scatter(reuse_segment_length,kverr)
    plt.savefig(save_path)
    plt.close()

def plot_analyze_low_correlation(test1_path, test2_path):
    """分析低相关性文本对在不同条件下的相关性变化
    
    Args:
        test1_path: 使用GTE-Qwen2-7B模型的测试结果路径
        test2_path: 大小写处理后的测试结果路径
    """
    import seaborn as sns
    
    # 加载数据
    test1_data = json.load(open(test1_path))
    test2_data = json.load(open(test2_path))
    
    # 创建图表布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    def create_heatmap(data, ax, title, x_label, y_label, original=True, color="blue"):
        points_x = []
        points_y = []
        
        for pair in data:
            reuse_rate = pair["reuse_rate"]
            if original:
                similarity = pair["cosine_similarity"]
            elif "cosine_similarity_by_gte_qwen2_7b" in pair:
                similarity = pair["cosine_similarity_by_gte_qwen2_7b"]
            elif "cosine_similarity_after_lowercase" in pair:
                similarity = pair["cosine_similarity_after_lowercase"]
                reuse_rate = pair["reuse_rate_after_lowercase"]
            
            points_x.append(reuse_rate)
            points_y.append(similarity)
        
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        
        # 计算相关系数
        correlation = np.corrcoef(points_x, points_y)[0,1]
        
        # 创建热力图数据
        x_bins = np.linspace(0, 1, 40)
        y_bins = np.linspace(0, 1, 40)
        density, _, _ = np.histogram2d(points_x, points_y, bins=[x_bins, y_bins])
        density = density / density.max()
        
        # 计算条件统计量
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
        
        # 绘制热力图
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_under('white')
        im = ax.imshow(density.T, origin='lower', aspect='auto',
                      extent=[0, 1, 0, 1], 
                      cmap=cmap,
                      vmin=0.0003,
                      vmax=1)
        
        # 添加相关系数信息
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                verticalalignment='top')
        
        # 绘制误差线
        bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        valid_mask = ~np.isnan(bin_means)
        bin_means_array = np.array(bin_means)[valid_mask]
        bin_stds_array = np.array(bin_stds)[valid_mask]
        centers = bin_centers[valid_mask]
        
        # 绘制误差线
        ax.errorbar(centers, 
                   bin_means_array,
                   yerr=bin_stds_array,
                   color=color, fmt='o-', markersize=4,
                   alpha=0.6,
                   label=f'Mean ± Std ({title})')
        
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return im
    
    # 原始MiniLM相似度与复用率关系
    im1 = create_heatmap(test1_data, ax1, 
                        "Original MiniLM",
                        "Token Reuse Rate", "MiniLM Cosine Similarity",
                        original=True, color="blue")
    
    # GTE-Qwen2-7B相似度与复用率关系
    im2 = create_heatmap(test1_data, ax2,
                        "GTE-Qwen2-7B",
                        "Token Reuse Rate", "GTE-Qwen2-7B Cosine Similarity",
                        original=False, color="green")
    
    # 原始文本相似度与复用率关系
    im3 = create_heatmap(test2_data, ax3,
                        "Original Text",
                        "Token Reuse Rate", "Original Cosine Similarity",
                        original=True, color="blue")
    
    # 小写处理后相似度与复用率关系
    im4 = create_heatmap(test2_data, ax4,
                        "Lowercase Text",
                        "Token Reuse Rate", "Lowercase Cosine Similarity",
                        original=False, color="green")
    
    # 添加颜色条
    plt.colorbar(im1, ax=ax1, label='Normalized Density')
    plt.colorbar(im2, ax=ax2, label='Normalized Density')
    plt.colorbar(im3, ax=ax3, label='Normalized Density')
    plt.colorbar(im4, ax=ax4, label='Normalized Density')
    
    plt.tight_layout()
    plt.savefig("examples/pipeline/images/low_correlation_analysis.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    
def find_high_low_sim_reuse(data_path:str,high_sim_low_reuse_path:str,low_sim_high_reuse_path:str):
    data = json.load(open(data_path))["low_quality_pairs"]
    high_sim_low_reuse = []
    low_sim_high_reuse = []
    for pair in data:
        del pair["source_token"]
        del pair["target_token"]
        if pair["cosine_similarity"] > pair["reuse_rate"]:
            high_sim_low_reuse.append(pair)
        else:
            low_sim_high_reuse.append(pair)
    json.dump(high_sim_low_reuse,open(high_sim_low_reuse_path,"w"),indent=4,ensure_ascii=False)
    json.dump(low_sim_high_reuse,open(low_sim_high_reuse_path,"w"),indent=4,ensure_ascii=False)

def slim_clean_data(data_path:str,save_path:str,threshold=0.7):
    data = json.load(open(data_path))
    new_data = []
    high_sim_index = []
    low_sim_index = []
    for index,pair in tqdm(enumerate(data),desc="挑选"):
        if pair["cosine_similarity"] >= threshold:
            high_sim_index.append(index)
        else:
            low_sim_index.append(index)

    max_remaind_num = 10000
    
    remained_high_sim_index = random.sample(high_sim_index,min(len(high_sim_index),max_remaind_num))   
    remained_low_sim_index = random.sample(low_sim_index,min(len(low_sim_index),max_remaind_num))
    for index in remained_high_sim_index:
        new_data.append(data[index])
    for index in remained_low_sim_index:
        new_data.append(data[index])
    
    print(f"保留{len(new_data)}条数据")
    # print(f"删除高相似度{len(remove_high_sim_index)}条数据")
    
    json.dump(new_data,open(save_path,"w"),indent=4,ensure_ascii=False)
            

# def remove_same_pairs(data_path:str,save_path:str,threshold=0.95):
#     data = json.load(open(data_path))
#     new_data = {
#         "high_quality_pairs":[],
#     }
#     # 先挑选出相似度超过0.95的超过,低于0.95的分成两组,低于了.95的全要,高于0.95的随机采样和低于0.95的组一样多
#     data = data["high_quality_pairs"]
#     high_sim_data = []
#     low_sim_data = []
#     for pair in tqdm(data,desc="挑选"):
#         if pair["cosine_similarity"] >= threshold:
#             high_sim_data.append(pair)
#         else:
#             low_sim_data.append(pair)
#     # 低于0.95的全部要
#     new_data["high_quality_pairs"].extend(low_sim_data)
#     # 高于0.95的随机采样和低于0.95的组一样多
#     high_sim_data = random.sample(high_sim_data,min(len(high_sim_data),len(low_sim_data)))
#     new_data["high_quality_pairs"].extend(high_sim_data)
#     print(f"保留{len(new_data['high_quality_pairs'])}条数据")
#     print(f"保留高相似度数据{len(high_sim_data)}条")
#     print(f"保留低相似度数据{len(low_sim_data)}条")
#     json.dump(new_data,open(save_path,"w"),indent=4,ensure_ascii=False)
    
if __name__ == "__main__":
    pass
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MKL_THREADING_LAYER"] = "GNU"  # 强制使用 GNU 线程层
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"  # 可选：强制使用 Intel 线程
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # raw_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # high_correlation_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_high_correlation.json"
    # low_correlation_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_low_correlation.json"
    # kverr_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_high_correlation_kverr.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,sim_reuse_path)
    # get_windowsize_similarity(sim_reuse_path,sim_reuse_path)
    # plot_similarity_reuse_rate(sim_reuse_path)
    # plot_window_size_reuse_rate(sim_reuse_path)
    # select_high_correlation_pairs(sim_reuse_path,high_correlation_path,low_correlation_path)
    # compute_kverr_between_high_correlation(high_correlation_path,kverr_path)
   
    # high_sim_low_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_high_sim_low_reuse.json"
    # low_sim_high_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_low_sim_high_reuse.json"
    # find_high_low_sim_reuse(low_correlation_path,high_sim_low_reuse_path,low_sim_high_reuse_path)
    
    # low_correlation_ana_test1_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_low_correlation_analysis_test1.json"
    # low_correlation_ana_test2_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_low_correlation_analysis_test2.json"
    # low_correlation_ana_test3_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_low_correlation_analysis_test3.json"
    # analyze_low_correlation(low_correlation_path,low_correlation_ana_path,device="cuda:1",test_1=True,test_2=False,test_3=False)
    # analyze_low_correlation(low_correlation_path,low_correlation_ana_test2_path,device="cuda:1",test_1=False,test_2=True,test_3=False)
    # analyze_low_correlation(low_correlation_path,low_correlation_ana_test3_path,device="cuda:1",test_1=False,test_2=False,test_3=True)
    # plot_analyze_low_correlation(low_correlation_ana_test1_path,low_correlation_ana_test2_path)
    
    # sim_reuse_path_gte_qwen2_7b = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate_gte_qwen2_7b.json"
    # compute_similarity(sim_reuse_path,
    #                    sim_reuse_path_gte_qwen2_7b,
    #                    model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
    #                    device="cuda:1",
    #                    batch_size=1)
    # plot_similarity_reuse_rate(sim_reuse_path_gte_qwen2_7b)
    # high_sim_low_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_high_sim_low_reuse_gte_qwen2_7b.json"
    # low_sim_high_reuse_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_low_sim_high_reuse_gte_qwen2_7b.json"
    # select_high_correlation_pairs(sim_reuse_path_gte_qwen2_7b)
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # raw_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # high_correlation_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_high_correlation.json"
    # high_correlation_slim_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_high_correlation_slim.json"
    # low_correlation_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_low_correlation.json"
    # kverr_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_high_correlation_kverr.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(sim_reuse_path,sim_reuse_path)
    # select_high_correlation_pairs(sim_reuse_path,high_correlation_path,low_correlation_path)
    # remove_same_pairs(high_correlation_path,high_correlation_remove_same_path)
    # compute_kverr_between_high_correlation(high_correlation_slim_path,kverr_path,device="cuda:0")
    
    
    # raw_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/belle/belle_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # raw_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # sim_reuse_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # high_correlation_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_high_correlation.json"
    # low_correlation_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_low_correlation.json"
    # high_correlation_slim_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_high_correlation_slim.json"
    # kverr_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_high_correlation_kverr.json"
    # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path)
    # get_windowsize_similarity(similar_path,sim_reuse_path)
    # select_high_correlation_pairs(sim_reuse_path,high_correlation_path,low_correlation_path,reuse_threshold=0.44,sim_threshold=0.4)
    # remove_same_pairs(high_correlation_path,high_correlation_slim_path)
    
    # compute_kverr_between_high_correlation(high_correlation_slim_path,kverr_path,device="cuda:0")
    # raw_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters.json"
    # clean_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs.json"
    # similar_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_cosine_similarity.json"
    # silm_similarity_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_cosine_similarity_slim.json"
    # sim_reuse_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_reuse_rate.json"
    # high_correlation_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_high_correlation.json"
    # low_correlation_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_low_correlation.json"
    # # select_similar_pairs(raw_path, clean_path)
    # compute_similarity(clean_path,similar_path,device="cuda:0")
    # slim_clean_data(similar_path,silm_similarity_path,threshold=0.8)
    # get_windowsize_diff_and_reuse(silm_similarity_path,sim_reuse_path)
    # select_high_correlation_pairs(sim_reuse_path,high_correlation_path,low_correlation_path)
    
    
    # data = [
    #     {
    #         "tag": "InstructionWild v2",
    #         "path": "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
    #         "color": "blue",
    #         "sim_threshold": 0.320,
    #         "reuse_threshold": 0.606
    #     },
    #     # {
    #     #     "tag": "InstructionWild v2 GTE-Qwen2-7B",
    #     #     "path": "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_reuse_rate_gte_qwen2_7b.json",
    #     #     "color": "green",
    #     #     "sim_threshold": 0.780,
    #     #     "reuse_threshold": 0.3
    #     # },
    #     {
    #         "tag": "WildChat-1M",
    #         "path": "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
    #         "color": "blue",
    #         "sim_threshold": 0.667,
    #         "reuse_threshold": 0.443
    #     },
        
    #     {
    #         "tag": "ShareGPT-90k",
    #         "path": "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
    #         "color": "blue",
    #         # "sim_threshold": 0.4,
    #         # "reuse_threshold": 0.3,
    #         "sim_threshold": 0.3,
    #         "reuse_threshold": 0.535
    #     },
    #     {
    #         "tag": "LMSysChat-1M",
    #         "path": "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters_similar_pairs_reuse_rate.json",
    #         "color": "blue",
    #         "sim_threshold": 0.4,
    #         "reuse_threshold": 0.44
    #     }
    # ]
    # plot_similarity_reuse_rate(data,save_path="examples/pipeline/images/similarity_reuse_rate.png")
    
    
    # kverr_data = [
    #     # {
    #     #     "tag": "InstructionWild v2",
    #     #     "path": "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings_clusters_similar_pairs_high_correlation_kverr.json",
    #     #     # "color": "blue"
    #     # },
    #     {
    #         "tag": "ShareGPT-90k",
    #         "path": "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings_clusters_similar_pairs_high_correlation_kverr.json",
    #         # "color": "blue"
    #     },
        
    # ]
    
    # plot_kverr_distribution(kverr_data,show_full_reuse=False)