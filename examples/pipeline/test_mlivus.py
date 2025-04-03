import json
from pymilvus import connections, MilvusClient
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np

def test_time_cost(input_path,output_path):
    model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda:0')
    data = json.load(open(input_path))
    profile_datas = []
    connection_name = "sharegpt"
    client = MilvusClient("examples/dataset/data/database/milvus_sharegpt.db")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    data = random.sample(data,5000)
    
    for item in tqdm(data):
        profile_data =[ ]
        text = item["text"]
        token_len = len(tokenizer.encode(text))
        if token_len < 10 or token_len > 4096:
            continue
        profile_data.append(token_len)
        # 计算embedding时间
        
        s_time = time.time()
        embedding = model.encode(text)
        e_time = time.time()
        profile_data.append(e_time - s_time)
        
        s_time = time.time()
        results = client.search(
            collection_name=connection_name,
            data=[embedding],
            limit=50,
            output_fields=["text"]
        )
        e_time = time.time()
        profile_data.append(e_time - s_time)
    
        profile_datas.append(profile_data)
    json.dump(profile_datas,open(output_path,"w"),ensure_ascii=False,indent=4)

def plot_profile_data(profile_data, output_path):
    # Load performance data
    with open(profile_data, "r") as f:
        profile_data = json.load(f)

    # Parse data and convert time to milliseconds
    token_lengths = np.array([data[0] for data in profile_data])
    embedding_times = np.array([data[1] * 1000 for data in profile_data])  # convert to ms
    search_times = np.array([data[2] * 1000 for data in profile_data])     # convert to ms

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot embedding times
    plt.scatter(token_lengths, embedding_times, color='tab:blue', alpha=0.5, label='Embedding Time')

    # Plot search times
    plt.scatter(token_lengths, search_times, color='tab:red', alpha=0.5, label='Search Time')

    # Set labels and title
    plt.xlabel('Token Length')
    plt.ylabel('Processing Time (ms)')
    plt.title('Relationship between Token Length and Processing Time (Log Scale)')
    plt.yscale('log')
    
    # Add legend
    plt.legend(loc='upper left', fontsize='small')

    # Optimize layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    sharegpt_path = "examples/dataset/data/sharegpt/sharegpt90k_ppl.json"
    output_path = "examples/dataset/data/sharegpt/sharegpt90k_milvus_profile.json"
    # test_time_cost(sharegpt_path,output_path)
    plot_profile_data(output_path,"examples/pipeline/images/sharegpt90k_milvus_profile.png")
