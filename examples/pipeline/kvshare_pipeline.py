from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import os
from kvshare_lib import KVSharePlugin
from log import setup_logger
from tqdm import tqdm
import hashlib
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
import random
from edit import edit_distance_with_operations

def gen_with_kvcache(json_file_path,db_name,device="cuda:0"):
    logger = setup_logger("kvshare_generate")
    json_file = open(json_file_path,"r")
    data = json.load(json_file)

    kvshare = KVSharePlugin(llm_model_name="Qwen/Qwen2.5-7B-Instruct",
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,
                            chunk_separator="\n\n--\n\n--\n\n",
                            milvus_connection_name=db_name,
                            milvus_database_path=f"/root/code/vllm_plus/examples/pipeline/data/dataset_db/{db_name}.db",
                            milvus_dimension=1536,
                            device=device,
                            )

    kvshare.use_semantic_search = False
    kvshare.save_kvcache = True
    kvshare.enable_show_log = False
    
    for item in tqdm(data, desc="Generating with KV cache", total=len(data)):
        output = kvshare.generate_with_kvcache(item["text1"])

def count_token_saved(json_file_path,db_name,device="cuda:0"):
    logger = setup_logger("count_token_saved")
    json_file = open(json_file_path,"r")
    data = json.load(json_file)
    
    kvshare = KVSharePlugin(llm_model_name="Qwen/Qwen2.5-7B-Instruct",
                        sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                        gpu_memory_utilization=0.6,
                        max_model_len=8192,
                        multi_step_stream_outputs=True,
                        disable_async_output_proc=True,
                        chunk_separator="\n\n--\n\n--\n\n",
                        milvus_connection_name=db_name,
                        milvus_database_path=f"/root/code/vllm_plus/examples/pipeline/data/dataset_db/{db_name}.db",
                        milvus_dimension=1536,
                        device=device,
                        )
    count_token_data = []
    for item in tqdm(data, desc="Counting tokens saved", total=len(data)):
        text1 = item["text2"]
        chunks = kvshare.text_split(text1)
        chunks = [chunk for chunk in chunks if chunk!="\n\n" and chunk!="\n"]
        num_token_hit = 0
        num_token_miss = 0
        chunks_indices = []
        chunks_ids = []
        for chunk in chunks:
            chunk_ids = kvshare.tokenizer.encode(chunk)
            chunks_ids.append(chunk_ids)
            if len(chunks_indices) !=0:
                chunks_indices.append([chunks_indices[-1][1],chunks_indices[-1][1]+len(chunk_ids)])
            else:
                chunks_indices.append([0,len(chunk_ids)])
        similar_text = []
        for i in range(len(chunks)):
            embeddings =  kvshare.sentence_model.encode(chunks[i])  # 批量计算embedding
            res = kvshare.client.search(
                collection_name=db_name,  # target collection
                data=[embeddings],  # query vectors
                limit=1,  # number of returned entities
                output_fields=["text", "key_value_cache","embeddings","hash_key"],  # specifies fields to be returned
            )

            if len(res[0]) != 0:
                if res[0][0]["distance"] >= 0.8:      
                    hash_srouce = hashlib.md5(chunks[i].encode()).hexdigest()
                    hash_target = hashlib.md5( res[0][0]["entity"]["text"].encode()).hexdigest()
                    if hash_srouce == hash_target:
                        num_token_hit += len(chunks_ids[i])
                        similar_text.append(res[0][0]["entity"]["text"])
                    else:
                        old_token_ids = kvshare.tokenizer.encode(res[0][0]["entity"]["text"])
                        new_token_ids = chunks_ids[i]
                        _,ops = edit_distance_with_operations(old_token_ids,new_token_ids,kvshare.tokenizer)
                        additional_indices = [op[-1] for op in ops if op[0] == "Insert" or op[0] == "Replace"]
                        num_token_hit += len(new_token_ids) - len(additional_indices)
                        num_token_miss += len(additional_indices)
                        similar_text.append(res[0][0]["entity"]["text"])
                else:
                    num_token_miss += len(kvshare.tokenizer.encode(chunks[i]))
                    similar_text.append("")
            else:
                num_token_miss += len(kvshare.tokenizer.encode(chunks[i]))
        num_all_token = 0
        for i in range(len(chunks_indices)):
            num_all_token += chunks_indices[i][1] - chunks_indices[i][0]
        rate = num_token_hit/num_all_token
        count_token_data.append(
            {"num_token_hit":num_token_hit,
            "num_token_miss":num_token_miss,
            "num_token_total":num_all_token,
            "rate":rate,
            "real_text":item["text2"],
            "similar_text": '\n'.join(similar_text)
            })
        # logger.info(f"num_token_hit: {num_token_hit}, num_token_miss: {num_token_miss}")
    # 统计rate > 0.5,>0.7,>0.9的item的个数，同时计算这些在整个数据集上的占比
    num_rate_01 = len([item for item in count_token_data if item["rate"] > 0.1])
    num_rate_03 = len([item for item in count_token_data if item["rate"] > 0.3])
    num_rate_05 = len([item for item in count_token_data if item["rate"] > 0.5])
    num_rate_07 = len([item for item in count_token_data if item["rate"] > 0.7])
    num_rate_09 = len([item for item in count_token_data if item["rate"] > 0.9])
    
    logger.info(f"num_rate_01: {num_rate_01}, num_rate_03: {num_rate_03}, num_rate_05: {num_rate_05}, num_rate_07: {num_rate_07}, num_rate_09: {num_rate_09}")
    logger.info(f"num_rate_01: {num_rate_01/len(count_token_data)}, num_rate_03: {num_rate_03/len(count_token_data)}, num_rate_05: {num_rate_05/len(count_token_data)}, num_rate_07: {num_rate_07/len(count_token_data)}, num_rate_09: {num_rate_09/len(count_token_data)}")
    
    all_saved_token = 0
    all_num_token_dataset = 0
    for item in count_token_data:
        all_saved_token += item["num_token_hit"]
        all_num_token_dataset += item["num_token_total"]    
    logger.info(f"all_saved_token: {all_saved_token}, all_num_token_dataset: {all_num_token_dataset}")
    logger.info(f"rate: {all_saved_token/all_num_token_dataset}")
    
    with open(f"examples/pipeline/data/count_token_data/{db_name}.json","w") as f:
        json.dump(count_token_data,f,indent=4)

def parse_json_file(json_file_path):
    json_file = open(json_file_path,"r")
    data = json.load(json_file)
    all_saved_token = 0
    all_num_token_dataset = 0
    for item in data:
        all_saved_token += item["num_token_hit"]
        all_num_token_dataset += item["num_token_total"]    
    print(f"all_saved_token: {all_saved_token}, all_num_token_dataset: {all_num_token_dataset}")
    print(f"rate: {all_saved_token/all_num_token_dataset}")

if __name__ == "__main__":
    pass
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # json_file = "examples/dataset/data/sharegpt90k_sim_only_similarity.json"
    # gen_with_kvcache(json_file,"sharegpt90k_qwen2_5_7b",device="cuda:0")
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # json_file = "examples/dataset/data/lmsys_chat_1m_sim_only_similarity.json"
    # gen_with_kvcache(json_file,"lmsys_chat_1m_qwen2_5_7b",device="cuda:0")
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # json_file = "examples/dataset/data/wild_chat_sim_only_similarity.json"
    # gen_with_kvcache(json_file,"wild_chat_qwen2_5_7b",device="cuda:0")
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # json_file = "examples/dataset/data/sharegpt90k_sim_only_similarity.json"
    # count_token_saved(json_file,"sharegpt90k_qwen2_5_7b",device="cuda:0")
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # json_file = "examples/pipeline/data/count_token_data/sharegpt90k_qwen2_5_7b.json"
    # parse_json_file(json_file)
    
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # json_file = "examples/dataset/data/lmsys_chat_1m_sim_only_similarity.json"
    # count_token_saved(json_file,"lmsys_chat_1m_qwen2_5_7b",device="cuda:0")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    json_file = "examples/dataset/data/wild_chat_sim_only_similarity.json"
    count_token_saved(json_file,"wild_chat_qwen2_5_7b",device="cuda:0")