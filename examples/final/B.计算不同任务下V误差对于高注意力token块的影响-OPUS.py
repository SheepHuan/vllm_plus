from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List
import evaluate
import random
from tqdm import tqdm
import sys

from libs.pipeline import KVShareNewPipeline
from vllm import LLM,SamplingParams

def benchmark_opus_full_compute(pipeline:KVShareNewPipeline):
    template_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease translate the following text from Chinese to English: {prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    meteor = evaluate.load("meteor")
    data = json.load(open("/root/code/kvshare/bench_data/benchmark_opus.json","r"))
    # data = data[:32]
    fc_res = "results/opus_result_normal.json"
    # 计算正常推理的meteor得分
    normal_score = []
    batch_size = 32  # 设置批处理大小
    sampling_params = SamplingParams(temperature=0.0,max_tokens=512)
    tokenizer = pipeline.model.get_tokenizer()
    for i in tqdm(range(0, len(data), batch_size), desc="计算正常推理得分"):
        batch_data = data[i:i + batch_size]
        batch_target_docs = [item["target_doc"] for item in batch_data]
        batch_answers = [item["answer"] for item in batch_data]
        
        # 批量处理输入
        batch_prompts = [template_text.format(prompt=doc) for doc in batch_target_docs]

        # 批量生成
        batch_outputs = pipeline.full_compute(pipeline.model,sampling_params,batch_prompts)
        
        # # 解码输出
        # batch_decoded_outputs = []
        for batch_idx, output in enumerate(batch_outputs):
            decoded_output = output.outputs[0].text
            score = meteor.compute(references=[batch_answers[batch_idx]],predictions=[decoded_output])["meteor"]
            normal_score.append(score)
        

    
    
    print(f"正常推理的meteor得分: {sum(normal_score)/len(normal_score)}")
    json.dump(normal_score, open("examples/dataset/data/opus_result_normal.json", "w"), indent=4, ensure_ascii=False)


def benchmark_opus_partial_compute(pipeline:KVShareNewPipeline,
                                   enbale_has_token_error:bool=False,
                                   enbale_las_token_error:bool=False):
    template_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease translate the following text from Chinese to English: {prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    meteor = evaluate.load("meteor")
    data = json.load(open("/root/code/kvshare/bench_data/benchmark_opus.json","r"))
    data = data[:8]
    normal_score = []
    batch_size = 4 # 设置批处理大小
    sampling_params = SamplingParams(temperature=0.0,max_tokens=512)
    tokenizer = pipeline.model.get_tokenizer()
    for i in tqdm(range(0, len(data), batch_size), desc="计算正常推理得分"):
        batch_data = data[i:i + batch_size]
        batch_target_docs = [item["target_doc"] for item in batch_data]
        batch_answers = [item["answer"] for item in batch_data]
        
        # 批量处理输入
        batch_prompts = [template_text.format(prompt=doc) for doc in batch_target_docs]

        batch_kvcache,batch_outputs,keys = pipeline.get_kvcache_by_full_compute(
            pipeline.model,
            sampling_params,
            batch_prompts,
            device="cuda:0"
        )
        # kv shape [num_head,k+v,seq_len,head_size]
        # batch_prompts = [template_text.format(prompt=doc) for doc in batch_target_docs]
        max_request_id = max([int(output.request_id) for output in batch_outputs])+1
       
        compute_len = 1
        batch_sample_selected_token_indices = [[compute_len-1] for _ in range(len(batch_prompts))]
        batch_unreused_map_indices = [list(range(len(tokenizer.encode(prompt))-compute_len,len(tokenizer.encode(prompt)))) for prompt in batch_prompts]
        batch_reused_map_indices = [list(range(len(tokenizer.encode(prompt))-compute_len)) for prompt in batch_prompts]
        next_batch_request_ids = [max_request_id+ii for ii in range(len(batch_prompts))]
        pass
        batch_pc_outputs = pipeline.partial_compute(
            pipeline.model,
            sampling_params,
            batch_prompts,
            batch_kvcache,
            batch_reused_map_indices,
            batch_unreused_map_indices,
            batch_sample_selected_token_indices,
            next_batch_request_ids,
            enable_kvshare=False,
            enable_cacheblend=False,
            enable_only_compute_unreused=True,
            has_additional_value_error = enbale_has_token_error,
            las_additional_value_error = enbale_las_token_error,
            enable_compute_as=True
        )
        for batch_idx, output in enumerate(batch_pc_outputs):
            decoded_output = output.outputs[0].text
            score = meteor.compute(references=[batch_answers[batch_idx]],predictions=[decoded_output])["meteor"]
            normal_score.append(score)
            print("===========================")
            print(batch_outputs[batch_idx].outputs[0].text)
            print(decoded_output)

    
    print(f"V存在误差得分: {sum(normal_score)/len(normal_score)}")
    # json.dump(normal_score, open("examples/dataset/data/opus_result_normal.json", "w"), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    # os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    pipeline = KVShareNewPipeline(model_name,device="cuda:0")
    # model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # model:CustomQwen2ForCausalLM = CustomQwen2ForCausalLM.from_pretrained(model_name,device_map="cuda",torch_dtype=torch.bfloat16).eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Translate the following text from Chinese to English:\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    

    # benchmark_opus_full_compute(pipeline)
    benchmark_opus_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=False)
    # benchmark_opus_partial_compute(pipeline,enbale_has_token_error=True,enbale_las_token_error=False)
    # benchmark_gsm8k()