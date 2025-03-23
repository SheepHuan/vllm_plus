import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
import json
from typing import List
def get_key_value(model:LLM,prompt: List[str],save_dir:str):
    # template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # prompt = template.format(prompt=prompt)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=1)
    output = model.generate(prompt, sampling_params)
    print(output[0].outputs[0].text)
    llm_layers = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
    
    past_key_values = []
    num_layer = len(llm_layers)
    for j in range(num_layer):
        hack_kv = llm_layers[j].self_attn.hack_kv
        temp_key_cache = hack_kv[0].clone()
        temp_value_cache = hack_kv[1].clone()
        # print(temp_key_cache.shape)
        past_key_values.append([temp_key_cache,temp_value_cache])
    os.makedirs(save_dir,exist_ok=True)
    kv_save_path = os.path.join(save_dir,"kv.pth")
    token_save_path = os.path.join(save_dir,"token.json")
    prompt_token_ids = output[0].prompt_token_ids
    torch.save(past_key_values,kv_save_path)
    json.dump(prompt_token_ids,open(token_save_path,"w"))

    return past_key_values


def gen_kv():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    llm = LLM(model=model_name, gpu_memory_utilization=0.6,         max_model_len=8192,
          multi_step_stream_outputs=True,enforce_eager=True,enable_prefix_caching=False,
          disable_async_output_proc=True,dtype="bfloat16")
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata["check"] = False
    llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata['collect'] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test1 = "apple,banana,orange"
    # tokens1 = tokenizer.encode(test1)
    get_key_value(llm,test1,"examples/pipeline/data/kv/test1")
    
    test2 = "apple"
    # tokens2 = tokenizer.encode(test2)
    get_key_value(llm,test2,"examples/pipeline/data/kv/test2")
    
    test3 = "banana"
    # tokens3 = tokenizer.encode(test3)
    get_key_value(llm,test3,"examples/pipeline/data/kv/test3")
    
    test4 = "orange"
    # tokens4 = tokenizer.encode(test4)
    get_key_value(llm,test4,"examples/pipeline/data/kv/test4")

    
    
def get_kv(kv_save_dir:str,token_save_dir:str,tag:str,split=None):
    kvs = torch.load(os.path.join(kv_save_dir,tag,"kv.pth"))
    token = json.load(open(os.path.join(token_save_dir,tag,"token.json"),"r"))
    
    
    num_layer = len(kvs)
    for i in range(num_layer):
        if split != None:
            start_idx,end_idx = split
            kvs[i][0] = kvs[i][0][start_idx:end_idx,:]
            kvs[i][1] = kvs[i][1][start_idx:end_idx,:]
            token = token[start_idx:end_idx]
        kvs[i] = torch.stack(kvs[i],dim=0)
    
    kvs = torch.stack(kvs,dim=0).permute(2,1,0,3)
    return kvs,token

def custom_hash(int_list):
    import hashlib
    # 将列表转换为字符串
    list_str = ','.join(map(str, int_list))

    # 使用 SHA-256 算法进行哈希
    hasher = hashlib.sha256()
    hasher.update(list_str.encode('utf-8'))
    hashed_value = hasher.hexdigest()
    return hashed_value

def plot_kv_acc(save_dir:str):
    import matplotlib.pyplot as plt
    
    # 设置图表风格
    # plt.style.use('seaborn')
    plt.figure(figsize=(10, 6))
    
    real_kvs, real_tokens = get_kv(save_dir, save_dir, "test1")
    
    chunk_set = dict()
    chunk_tag = ["test2", "test3", "test4"]
    for i in range(len(chunk_tag)):
        chunk_kv, chunk_token = get_kv(save_dir, save_dir, chunk_tag[i])
        chunk_set[chunk_token[0]] = (chunk_kv[0], chunk_token)
        print("save chunk", chunk_token[0])
    
    key_errs = []
    value_errs = []
    num_layer = 28
    for i in range(num_layer):
        layer_ids = list(range(0, i))
        avg_key_err = []
        avg_value_err = []
        for pos, token in enumerate(real_tokens):
            if token in chunk_set:
                real_token_kv = real_kvs[pos]
                chunk_pos = chunk_set[token][0]
                
                key_err = torch.sum(torch.abs(real_token_kv[0,layer_ids,:] - chunk_pos[0,layer_ids,:]))
                value_err = torch.sum(torch.abs(real_token_kv[1,layer_ids,:] - chunk_pos[1,layer_ids,:]))
                avg_key_err.append(key_err.item())
                avg_value_err.append(value_err.item())
        key_errs.append(sum(avg_key_err)/len(avg_key_err) if avg_key_err else 0)
        value_errs.append(sum(avg_value_err)/len(avg_value_err) if avg_value_err else 0)
    
    # 绘制曲线
    plt.plot(list(range(num_layer)), key_errs, 
             label='Key Error', 
             color='#1f77b4', 
             linewidth=2, 
             marker='o',
             markersize=6)
    plt.plot(list(range(num_layer)), value_errs, 
             label='Value Error', 
             color='#2ca02c', 
             linewidth=2, 
             marker='s',
             markersize=6)
    
    # 设置x轴刻度
    plt.xticks(range(num_layer))  # 显示所有整数刻度
    
    # 设置图表属性
    plt.xlabel('Layer Number', fontsize=15)
    plt.ylabel('Error', fontsize=15)
    # plt.title('Key-Value Error across Layers', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='upper left')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig("examples/pipeline/images/kv_acc.png", dpi=300, bbox_inches='tight')
        
if __name__ == "__main__":
    
    gen_kv()
    # plot_kv_acc("examples/pipeline/data/kv")

    