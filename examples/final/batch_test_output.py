import json
import torch
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import traceback
import time
import json
from vllm import LLM,SamplingParams
from tqdm import tqdm
import os
from libs.pipeline import KVShareNewPipeline
from libs.edit import KVEditor
from transformers import AutoTokenizer
import random
import langid
import json
import requests
from openai import OpenAI

qwen_template_text = "<|im_start|>You are Qwen, created by Alibaba. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"
llama3_template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def process_single_item(item, client):
    try:
        text = item["target_text"]["text"]
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="chatgpt-4o-latest",
        )
        output = chat_completion.choices[0].message.content
        item["target_text"]["chatgpt_output"] = output
        return item
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        return None

def process_batch(batch_items):
    client = OpenAI(
        api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",
        base_url="https://www.dmxapi.cn/v1",
    )
    results = []
    for item in batch_items:
        result = process_single_item(item, client)
        if result is not None:
            results.append(result)
    return results

def test_gpt(data_path, save_path, num_processes=4):
    data = json.load(open(data_path, "r"))
    save_data = []
    # data = data[:10]
    # 将数据分成多个批次
    batch_size = len(data) // num_processes
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    print(f"开始使用{num_processes}个进程处理数据...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        for future in tqdm(futures, desc="处理批次"):
            try:
                batch_results = future.result()
                save_data.extend(batch_results)
            except Exception as e:
                print(f"批次处理出错: {str(e)}")
                print(f"错误详情: {traceback.format_exc()}")
                continue
    
    print(f"成功处理 {len(save_data)}/{len(data)} 条数据")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

def partial_compute_qwen2(data_path, save_path, model_name="Qwen/Qwen2.5-7B-Instruct", batch_size=2):
    pipeline = KVShareNewPipeline(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    save_data = []
    data = json.load(open(data_path, "r"))
    data = random.sample(data,4000)
    
    if os.path.exists(save_path):
        has_run_data = json.load(open(save_path, "r"))
        has_run_key = set()
        for item in has_run_data:
            has_run_key.add(item["target_text"]["id"])
    else:
        has_run_key = set()
        has_run_data = []
    
    data = [item for item in data if item["target_text"]["id"] not in has_run_key]
  
    
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        try:
            batch_items = data[i:i + batch_size]
            # print(i)
            # 批量准备prompt
            all_source_prompts = []
            all_target_prompts = []
            for item in batch_items:
                source_prompts = [item["sim_top1"]["text"], item["reused_top1"]["text"]]
                target_prompts = [item["target_text"]["text"] for _ in range(len(source_prompts))]
                
                all_source_prompts.extend([template_text.format(user_text=prompt) for prompt in source_prompts])
                all_target_prompts.extend([template_text.format(user_text=prompt) for prompt in target_prompts])

            # 批量计算full compute
            full_compute_target_outputs = KVShareNewPipeline.batch_full_compute(
                pipeline.model,
                SamplingParams(temperature=0.0, max_tokens=512),
                all_target_prompts
            )
            batch_target_token_ids = [output.prompt_token_ids for output in full_compute_target_outputs]
            
            # 批量获取kv cache
            batch_source_key_values, batch_source_outputs = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0.0, max_tokens=1),
                all_source_prompts
            )
            batch_source_token_ids = [source_output.prompt_token_ids for source_output in batch_source_outputs]
            
            # 批量编辑kv cache
            target_kvcache, reused_map_indices, unreused_map_indices, sample_selected_token_indices = KVEditor.batch_kvedit(
                batch_target_token_ids,
                batch_source_token_ids,
                batch_source_key_values
            )
            
            # 批量partial compute
            partial_batch_target_outputs = KVShareNewPipeline.partial_compute(
                pipeline.model,
                SamplingParams(temperature=0.0, max_tokens=512),
                all_target_prompts,
                reused_map_indices,
                unreused_map_indices,
                sample_selected_token_indices,
                target_kvcache
            )

            # 保存结果
            for idx, item in enumerate(batch_items):
                base_idx = idx * 2  # 因为每个item有2个输出
                item["sim_top1"]["partial_output"] = partial_batch_target_outputs[base_idx].outputs[0].text
                item["reused_top1"]["partial_output"] = partial_batch_target_outputs[base_idx + 1].outputs[0].text
                item["target_text"]["full_output"] = full_compute_target_outputs[base_idx].outputs[0].text
                save_data.append(item)
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")
            with open(save_path,"w") as f:
                json.dump(save_data+has_run_data,f,indent=4,ensure_ascii=False)
            continue
        
    with open(save_path,"w") as f:
        json.dump(save_data+has_run_data,f,indent=4,ensure_ascii=False)
        
def chech_move(data_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = json.load(open(data_path,"r"))
    for idx,item in tqdm(enumerate(data),desc="Checking moves"):
        print(idx)
        sim_top1_text = item["sim_top1"]["text"] 
        reused_top1_text = item["reused_top1"]["text"]
       
        target_text = item["target_text"]["text"]
        
        target_tokens = tokenizer.encode(target_text)
        sim_tokens = tokenizer.encode(sim_top1_text)
        reused_tokens = tokenizer.encode(reused_top1_text)
        
        diff_report = KVEditor.find_text_differences(sim_tokens,target_tokens)
        for move in diff_report["moves"]:
            move_from = move["from_position"]
            move_to = move["to_position"]
            if move_from[1]-move_from[0]+1 ==0 or move_to[1]-move_to[0]+1 ==0:
                print(move_from,move_to)
                
        diff_report = KVEditor.find_text_differences(reused_tokens,target_tokens)
        for move in diff_report["moves"]:
            move_from = move["from_position"]
            move_to = move["to_position"]
            if move_from[1]-move_from[0]+1 ==0 or move_to[1]-move_to[0]+1 ==0:
                print(move_from,move_to)



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"]="true"
    # os.environ["VLLM_USE_MODELSCOPE"]="true"
    # export VLLM_USE_MODELSCOPE=True
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"、
    # model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-Int4"
    template_text = llama3_template_text
    # data_path = "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331_clean.json"
    # gpt_path = "examples/final/data/instruction_wildv2_similar_250331_answer_by_chatgpt.json"
    # save_path = f"examples/final/data/instruction_wildv2_similar_250331_answer_by_{model_name.split('/')[-1]}_partial_output.json"
    # partial_compute_qwen2(gpt_path,save_path,model_name,batch_size=4)
    # data = json.load(open("examples/dataset/data/sharegpt/sharegpt90k_similar_250331_clean.json","r"))
    # print(len(data))

    # data_path = "examples/dataset/data/sharegpt/sharegpt90k_similar_250331_clean.json"
    gpt_path = "examples/pipeline/data/data/sharegpt90k_similar_250331_answer_by_chatgpt.json"
    save_path = f"examples/pipeline/data/data/sharegpt90k_similar_250331_answer_by_{model_name.split('/')[-1]}_partial_output.json"
    
    # test_gpt(data_path,gpt_path)
    partial_compute_qwen2(gpt_path,save_path,model_name,batch_size=1)
    # chech_move(data_path)
    
