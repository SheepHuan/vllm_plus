from sentence_transformers import SentenceTransformer
import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
from plot_sim_reuse_correlation import get_key_value
from edit2 import apply_change,find_text_differences
from vllm import LLM
from transformers import AutoTokenizer  

# def compute_similarity(model,prompt1,prompt2):
#     embedding1 = model.encode(prompt1,convert_to_tensor=True)
#     embedding2 = model.encode(prompt2,convert_to_tensor=True)
#     similarity = torch.nn.functional.cosine_similarity(embedding1,embedding2,dim=0)
#     return similarity

def compute_kverr(tokenizer,prompt1:str,prompt2:str,template:str):
    # source_kv,source_token = get_key_value(model,template.format(prompt=prompt1))
    # target_kv,target_token = get_key_value(model,template.format(prompt=prompt2))
    token1 = tokenizer.encode(prompt1)
    token2 = tokenizer.encode(prompt2)
    diff_report = find_text_differences(token1,token2,window_size=1)
    # moves = diff_report["moves"]
    # for move in moves:
    for move in diff_report["moves"]:
        print(move["text"],move["from_position"],move["to_position"])
    print(diff_report["summary"]["reuse_ratio"])        
    # modified_kv,reused_map_indices,_ = apply_change(source_token,target_token,source_kv,diff_report)
    # kverr = torch.abs(target_kv-modified_kv)
    # kverr = kverr.mean(dim=[0,1,3])
    # # target_kv = target_kv[:,:,reused_map_indices,:]
    # # modified_kv = modified_kv[:,:,reused_map_indices,:]

    # # kverr = torch.nn.functional.mse_loss(target_kv,modified_kv)
    # for i in range(len(target_token)):
    #     if i in reused_map_indices:
    #         print("reused:",tokenizer.decode(target_token[i]),"->",kverr[i])
    #     else:
    #         print("unused:",tokenizer.decode(target_token[i]),"->","inf")
    # pass
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MKL_THREADING_LAYER"] = "GNU"  # 强制使用 GNU 线程层
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"  # 可选：强制使用 Intel 线程
    template = "<|im_start|>user\n{prompt}\n<|im_end|>"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
    # model = LLM(model=model_name,
    #             device="cuda:0",
    #             dtype="bfloat16"
    #             )
    template = "<|im_start|>user\n{prompt}\n<|im_end|>"
    prompt1 = "A,B,C,D,E,F,G."
    prompt2 = "A\tB\tC\tD\tE\tF\tG."
    prompt3 = "A,B,C,D\tE\tF\tG."
    
    compute_kverr(tokenizer,prompt2,prompt3,template)
    
    # similarity1 = compute_similarity(model,prompt1,prompt2)
    # similarity2 = compute_similarity(model,prompt1,prompt3)
    # print(similarity1)
    # print(similarity2)

