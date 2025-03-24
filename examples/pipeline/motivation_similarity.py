import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import torch
import tqdm
import numpy as np
import pandas as pd
import hashlib
import uuid
import os
from multiprocessing import Pool
import math

def generate_embeddings(data,save_path,batch_size=1024):
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct",device="cuda:1").to(torch.bfloat16)
    print(f"Generating embeddings for {len(data)} requests")
    # data = data[:1024]
    # 批量计算data中text的embdings，然后保存为torch张量字典，字典的key时item的hashid,值时embedding张量
    saved_data = {}
    for i in tqdm.tqdm(range(0,len(data),batch_size),desc="Generating embeddings"):
        batch_data = data[i:i+batch_size]
        text = [item["text"] for item in batch_data]
        
        try:
            embeddings = model.encode(text)
            ids = [item["id"] for item in batch_data]
            # 将embeddings保存为torch张量字典，字典的key时item的hashid,值时embedding张量
            batch_data = {id:embedding for id,embedding in zip(ids,embeddings)}
            saved_data.update(batch_data)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error generating embeddings for batch {i}: {e}")
            continue
        
    np.savez(save_path,**saved_data)



def cluster_embeddings(embeddings_path,low_dim_save_path=None):
    import cupy as cp
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN  
    from cuml.metrics import pairwise_distances
    import os
    # 使用DBSCAN聚类算法对embeddings进行聚类
    embeddings = np.load(embeddings_path)
    gpu_embeddings = cp.asarray(embeddings)  # 转换到CuPy数组（显存占用约4.3GB）
    
    # ---------------------------------------------------
    # 步骤2: GPU降维（可选但推荐）
    # ---------------------------------------------------
    umap = UMAP(
        n_components=50, 
        n_neighbors=30, 
        init="random",
        random_state=42
    )
    gpu_lowdim = umap.fit_transform(gpu_embeddings)  # 输出CuPy数组
    if low_dim_save_path is not None:
        np.save(low_dim_save_path,cp.asnumpy(gpu_lowdim))
    # ---------------------------------------------------
    # 步骤3: GPU快速聚类
    # ---------------------------------------------------
    clusterer = HDBSCAN(
        min_samples=15, 
        cluster_selection_epsilon=0.5,
        metric="euclidean", 
        prediction_data=True
    )
    labels_gpu = clusterer.fit_predict(gpu_lowdim)  # 返回CuPy数组
    # 将结果转回CPU（按需）
    labels = cp.asnumpy(labels_gpu)
    
    # ---------------------------------------------------
    # 步骤4: 相似性分析（直接在GPU计算）
    # ---------------------------------------------------
    def get_representative_samples(cluster_id):
        """获取每个簇的代表样本索引"""
        mask = labels_gpu == cluster_id
        cluster_points = gpu_lowdim[mask]
        
        # 计算簇中心
        centroid = cluster_points.mean(axis=0)
        
        # 找距离中心最近的样本
        distances = pairwise_distances(cluster_points, centroid.reshape(1, -1))
        nearest_idx = cp.argmin(distances)
        
        return {
            "centroid": centroid,
            "nearest_sample": cluster_points[nearest_idx],
            "sample_indices": cp.where(mask)[0]
        }

    # 获取所有有效簇的代表信息
    valid_clusters = [c for c in cp.unique(labels_gpu) if c != -1]
    cluster_info = {c: get_representative_samples(c) for c in valid_clusters}
    pass

def count_duplicate_ids(data):
    # 检查有多少id时相同的
    id_counts = {}
    for item in data:
        if item["id"] in id_counts:
            id_counts[item["id"]] += 1
        else:
            id_counts[item["id"]] = 1
    for id,count in id_counts.items():
        if count > 1:
            print(f"id: {id} 出现了 {count} 次")

def process_chunk(args):
    npz_path, start_idx, end_idx, keys = args
    # 在每个进程中单独加载数据
    data = np.load(npz_path)
    chunk_embeddings = np.stack([data[key] for key in keys[start_idx:end_idx]])
    chunk_keys = keys[start_idx:end_idx]
    return chunk_embeddings, chunk_keys

def convert_npz_to_npy(npz_path, npy_path, n_processes=8):
    # 只获取keys
    with np.load(npz_path) as data:
        n_samples = len(data.files)
        keys = list(data.files)
    
    # 计算每个进程处理的数据量
    chunk_size = math.ceil(n_samples / n_processes)
    chunks = []
    
    # 准备每个进程的参数
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        chunks.append((npz_path, i, end_idx, keys))
    
    # 使用进程池处理数据
    with Pool(n_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    # 分离embeddings和keys
    embeddings_chunks, keys_chunks = zip(*results)
    
    # 合并所有结果
    embeddings = np.concatenate(embeddings_chunks, axis=0)
    ordered_keys = sum(keys_chunks, [])  # 扁平化keys列表
    
    print(f"Saving embeddings with shape {embeddings.shape} to {npy_path}")
    print(f"Saving {len(ordered_keys)} keys to {npy_path}.keys.json")
    
    # 保存embeddings和对应的keys
    np.save(npy_path, embeddings)
    # 将keys保存为JSON文件
    with open(f"{npy_path}.keys.json", 'w', encoding='utf-8') as f:
        json.dump(ordered_keys, f, ensure_ascii=False, indent=2)

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x



def faiss_cluster(embeddings_path, json_keys_path, original_json_path, threshold=0.8):
    import numpy as np
    import faiss
    import torch
    from collections import defaultdict
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 加载原始数据，建立ID到文本的映射
    print("Loading original data...")
    with open(original_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    id_to_text = {item['id']: item['text'] for item in original_data}
    
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    keys = json.load(open(json_keys_path))
    
    # 转换为GPU资源
    print("Creating GPU index...")
    res = faiss.StandardGpuResources()
    dim = embeddings_normalized.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(embeddings_normalized)
    print("全部请求数量:",embeddings_normalized.shape)
    # 分批处理knn search
    batch_size = 50000  # 可以根据GPU内存调整
    k = 100  # 每个查询返回的邻居数
    threshold = 0.95  # 相似度阈值
    n_batches = (len(embeddings_normalized) + batch_size - 1) // batch_size
    
    print("Processing knn search in batches...")
    all_pairs = set()
    for i in tqdm.tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(embeddings_normalized))
        batch = embeddings_normalized[start_idx:end_idx]
        
        # 执行knn search
        distances, indices = gpu_index.search(batch, k)
        
        # 处理结果
        for j in range(end_idx - start_idx):
            current_idx = start_idx + j
            for neighbor_idx, dist in zip(indices[j], distances[j]):
                if dist > threshold and neighbor_idx > current_idx:  # 确保相似度大于阈值且避免重复
                    all_pairs.add((current_idx, neighbor_idx))
    
    # 后续处理保持不变
    similar_pairs_count = len(all_pairs)
    print(f"找到 {similar_pairs_count} 对相似文本")
    
    unique_texts = set()
    for pair in all_pairs:
        unique_texts.update(pair)
    duplicate_ratio = len(unique_texts) / len(embeddings)
    print(f"相似对数量: {similar_pairs_count}, 涉及{duplicate_ratio:.1%}个用户对话")
    
    similar_texts = set()
    for i, j in all_pairs:
        similar_texts.add(i)
        similar_texts.add(j)
    similar_texts_count = len(similar_texts)
    print(f"相似文本数量: {similar_texts_count}")
    
    # 保存相似对的结果，增加文本内容
    similar_pairs_with_keys = []
    for idx1, idx2 in all_pairs:
        similar_pairs_with_keys.append({
            "id1": keys[idx1],
            "id2": keys[idx2],
            "text1": id_to_text[keys[idx1]],
            "text2": id_to_text[keys[idx2]],
            "similarity": float(np.dot(embeddings_normalized[idx1], embeddings_normalized[idx2]))
        })
    
    # 保存相似对结果
    output_pairs_path = embeddings_path.replace('.npy', '_similar_pairs.json')
    with open(output_pairs_path, 'w', encoding='utf-8') as f:
        json.dump(similar_pairs_with_keys, f, ensure_ascii=False, indent=2)
    print(f"相似对结果已保存到: {output_pairs_path}")
    
    # 构建聚类
    uf = UnionFind(len(embeddings_normalized))
    for i, j in all_pairs:
        uf.union(i, j)
    
    clusters = defaultdict(list)
    for i in range(len(embeddings_normalized)):
        clusters[uf.find(i)].append(i)
    
    # 将聚类结果转换为包含原始ID和文本的格式
    clusters_with_keys = {}
    for cluster_id, indices in clusters.items():
        if len(indices) >= 2:  # 只保存大小≥2的群体
            members = []
            for idx in indices:
                key = keys[idx] 
                members.append({
                    "id": key,
                    "text": id_to_text[key]
                })
            clusters_with_keys[str(cluster_id)] = {
                "size": len(indices),
                "members": members
            }
    
    # 保存聚类结果
    output_clusters_path = embeddings_path.replace('.npy', '_clusters.json')
    with open(output_clusters_path, 'w', encoding='utf-8') as f:
        json.dump({
            "statistics": {
                "total_texts": len(embeddings),
                "similar_pairs": len(all_pairs),
                "duplicate_ratio": duplicate_ratio,
                "similar_texts": similar_texts_count,
                "cluster_count": len(clusters_with_keys)
            },
            "clusters": clusters_with_keys
        }, f, ensure_ascii=False, indent=2)
    print(f"聚类结果已保存到: {output_clusters_path}")
    
    # 计算并打印聚类统计信息
    cluster_sizes = [len(c) for c in clusters.values() if len(c) >= 2]
    if cluster_sizes:
        max_cluster_size = max(cluster_sizes)
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        print(f"\n聚类统计信息:")
        print(f"最大群体规模: {max_cluster_size} 个文本")
        print(f"平均群体规模: {avg_cluster_size:.2f} 个文本")
        print(f"群体数量: {len(cluster_sizes)} 个")
        print(f"聚类中的总文本数: {sum(cluster_sizes)} 个")
    else:
        print("没有找到任何聚类群体")
        
        
def count_cluster_size(cluster_path):
    # 统计有多少不同的组大小，计算每个大小占的租占比
    cluster_sizes = []
    with open(cluster_path, 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    clusters = clusters["clusters"]
    for cluster_id, cluster_info in clusters.items():
        cluster_sizes.append(cluster_info["size"])
    cluster_sizes_count = {}
    for size in cluster_sizes:
        if size in cluster_sizes_count:
            cluster_sizes_count[size] += 1
        else:
            cluster_sizes_count[size] = 1
    print(cluster_sizes_count)
    # return cluster_sizes_count
    

if __name__ == "__main__":
    pass
    # data_path = "examples/dataset/data/lmsys_chat_1m_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/lmsys_chat_1m_batch_embeddings.npz"
    # generate_embeddings(data,save_path,2048)


    # data_path = "examples/dataset/data/lmsys_chat_1m_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/lmsys_chat_1m_batch_embeddings.npy",
    #     n_processes=64  # 可以根据CPU核心数调整
    # )
    # data_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)

    # cluster_path = "examples/dataset/data/similar/lmsys/lmsys_chat_1m_batch_embeddings_clusters.json"
    # count_cluster_size(cluster_path)
    
    # data_path = "examples/dataset/data/sharegpt90k_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/sharegpt90k_batch_embeddings.npz"
    # generate_embeddings(data,save_path,4)
    
    # data_path = "examples/dataset/data/sharegpt90k_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/sharegpt90k_batch_embeddings.npy",
    #     n_processes=128  # 可以根据CPU核心数调整
    # )
    
    # data_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/sharegpt/sharegpt90k_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)
    
    # data_path = "examples/dataset/data/wild_chat_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/wild_chat_batch_embeddings.npz"
    # generate_embeddings(data,save_path,4)
    
    # data_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings.npy",
    #     n_processes=96  # 可以根据CPU核心数调整
    # ) 
    # data_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/wildchat/wild_chat_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/wildchat/wild_chat_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)
    
    # data_path = "examples/dataset/data/similar/belle/belle_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/similar/belle/belle_batch_embeddings.npz"
    # generate_embeddings(data,save_path,64)
    
    # data_path = "examples/dataset/data/similar/belle/belle_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/similar/belle/belle_batch_embeddings.npy",
    #     n_processes=64  # 可以根据CPU核心数调整
    # )
    # data_path = "examples/dataset/data/similar/belle/belle_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/belle/belle_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/belle/belle_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)
    
    
    
    # data_path = "examples/dataset/data/similar/chatbot_arena/chatbot_arena_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/similar/chatbot_arena/chatbot_arena_batch_embeddings.npz"
    # generate_embeddings(data,save_path,64)
    
    # data_path = "examples/dataset/data/similar/chatbot_arena/chatbot_arena_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/similar/chatbot_arena/chatbot_arena_batch_embeddings.npy",
    #     n_processes=64  # 可以根据CPU核心数调整
    # )
    # data_path = "examples/dataset/data/similar/chatbot_arena/chatbot_arena_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/chatbot_arena/chatbot_arena_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/chatbot_arena/chatbot_arena_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)
    
    # data_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings.npz"
    # generate_embeddings(data,save_path,64)
    
    # data_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings.npy",
    #     n_processes=64  # 可以根据CPU核心数调整
    # )
    
    # data_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/instruction_wildv2/instruction_wildv2_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)
    
    
    
    # data_path = "examples/dataset/data/similar/moss/moss_ppl.json"
    # data = json.load(open(data_path))
    # save_path = "examples/dataset/data/similar/moss/moss_batch_embeddings.npz"
    # generate_embeddings(data,save_path,128)
    
    # data_path = "examples/dataset/data/similar/moss/moss_batch_embeddings.npz"
    # convert_npz_to_npy(
    #     data_path,
    #     "examples/dataset/data/similar/moss/moss_batch_embeddings.npy",
    #     n_processes=64  # 可以根据CPU核心数调整
    # )
    
    # data_path = "examples/dataset/data/similar/moss/moss_batch_embeddings.npy"
    # json_keys_path = "examples/dataset/data/similar/moss/moss_batch_embeddings.npy.keys.json"
    # original_json_path = "examples/dataset/data/similar/moss/moss_ppl.json"  # 原始数据文件
    # faiss_cluster(data_path, json_keys_path, original_json_path)