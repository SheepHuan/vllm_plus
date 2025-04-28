from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List
import evaluate
import random
from tqdm import tqdm
import sys
import math
import re

from libs.pipeline import KVShareNewPipeline
from vllm import LLM,SamplingParams



def benchmark_opus_partial_compute(pipeline:KVShareNewPipeline,
                                   enbale_has_token_error:bool=False,
                                   enbale_las_token_error:bool=False):
    # template_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease translate the following text from Chinese to English: {prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nPlease translate the following text from Chinese to English: {prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    meteor = evaluate.load("meteor")
    data = json.load(open("/root/code/kvshare/bench_data/benchmark_opus.json","r"))
    # data = data[:48]
    normal_score = []
    batch_size = 24 # 设置批处理大小
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
            # print("===========================")
            # print(batch_outputs[batch_idx].outputs[0].text)
            # print("***")
            # print(decoded_output)

    print("================="*10)
    print(f"opus meteor: {sum(normal_score)/len(normal_score)}, has: {enbale_has_token_error}, las: {enbale_las_token_error}")
    # print(f"V存在误差得分: {sum(normal_score)/len(normal_score)}")


def benchmark_xsum_partial_compute(pipeline:KVShareNewPipeline,
                                   enbale_has_token_error:bool=False,
                                   enbale_las_token_error:bool=False):
    # template_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease summarize the following text into a sentence: {prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nPlease summarize the following text into a sentence: {prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    rouge = evaluate.load("rouge")
    data = json.load(open("/root/code/kvshare/bench_data/benchmark_xsum.json","r"))
    # data = data[:48]
    normal_score = []
    batch_size = 8 # 设置批处理大小
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
            score = rouge.compute(references=[batch_answers[batch_idx]],predictions=[decoded_output])["rougeL"]
            normal_score.append(score)
            # print("===========================")
            # print(batch_outputs[batch_idx].outputs[0].text)
            # print("***")
            # print(decoded_output)

    print("================="*10)
    print(f"xsm rougel: {sum(normal_score)/len(normal_score)}, has: {enbale_has_token_error}, las: {enbale_las_token_error}")
    # print(f"V存在误差得分: {sum(normal_score)/len(normal_score)}")

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
    return last_digit

def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))

def benchmark_gsm8k_partial_compute(pipeline:KVShareNewPipeline,
                                   enbale_has_token_error:bool=False,
                                   enbale_las_token_error:bool=False):
    # template_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nPlease tell the answer of  following question: {prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Please tell the answer of  following question: \n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # rouge = evaluate.load("rouge")
    data = json.load(open("/root/code/kvshare/bench_data/benchmark_gsm8k.json","r"))
    # data = data[:48]
    normal_score = []
    batch_size = 16 # 设置批处理大小
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
            score = is_correct(decoded_output,batch_answers[batch_idx])
            # score = rouge.compute(references=[batch_answers[batch_idx]],predictions=[decoded_output])["rougeL"]
            normal_score.append(score)
            # print("===========================")
            # print(batch_outputs[batch_idx].outputs[0].text)
            # print("***")
            # print(decoded_output)

    
    print("================="*10)
    print(f"gsm8k acc: {sum(normal_score)/len(normal_score)}, has: {enbale_has_token_error}, las: {enbale_las_token_error}")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    # os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    pipeline = KVShareNewPipeline(model_name,device="cuda:0")
    # model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # model:CustomQwen2ForCausalLM = CustomQwen2ForCausalLM.from_pretrained(model_name,device_map="cuda",torch_dtype=torch.bfloat16).eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Translate the following text from Chinese to English:\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    

    # benchmark_opus_full_compute(pipeline)
    benchmark_opus_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=False)
    benchmark_opus_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=True)
    benchmark_opus_partial_compute(pipeline,enbale_has_token_error=True,enbale_las_token_error=False)
    print("==========="*10)
    benchmark_xsum_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=False)
    benchmark_xsum_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=True)
    benchmark_xsum_partial_compute(pipeline,enbale_has_token_error=True,enbale_las_token_error=False)
    print("==========="*10)
    benchmark_gsm8k_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=False) 
    benchmark_gsm8k_partial_compute(pipeline,enbale_has_token_error=False,enbale_las_token_error=True)  
    benchmark_gsm8k_partial_compute(pipeline,enbale_has_token_error=True,enbale_las_token_error=False)  