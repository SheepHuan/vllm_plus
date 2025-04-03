import json
import random
import os
import tqdm
import json
from libs.pipeline import KVShareNewPipeline
from libs.edit import KVEditor
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import os
import random
from sentence_transformers import SentenceTransformer
import numpy as np
from matplotlib import pyplot as plt
from evaluate import load
import seaborn as sns
from scipy.stats import zscore
import math
import re
import evaluate
import matplotlib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

qwen_template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nTranslate the following text from Chinese to English:\n{text}\n<|im_end|>\n<|im_start|>assistant\n"""
llama3_template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Translate the following text from Chinese to English:\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


def generate_output_data(input_path: str, output_path: str, model_name = "Qwen/Qwen2.5-7B-Instruct", batch_size=4):
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    
    all_data = data["all_data"]
    similar_pairs = data["data"]
    similar_pairs = random.sample(similar_pairs, min(len(similar_pairs), 2000))
    save_data = []
    if os.path.exists(output_path):
        has_run_data = json.load(open(output_path,"r"))
        has_run_ids = set()
        # for item in has_run_data:
        #     has_run_ids.add(item["id"])
    else:
        has_run_ids = set()
        has_run_data =[]
        
    BLEU = evaluate.load('bleu')
    
    # 按batch_size分批处理数据
    for i in tqdm(range(0, len(similar_pairs), batch_size), desc="Processing batches"):
        try:
            batch_items = similar_pairs[i:i + batch_size]
            real_need_run_items = []
            for item in batch_items:
                if item["id"] in has_run_ids:
                    continue
                real_need_run_items.append(item)
            batch_items = real_need_run_items
            if len(batch_items)==0:
                continue
            
            # 准备所有prompt
            all_target_prompts = []
            all_source_prompts = []
            batch_answers = []
            for item in batch_items:
                question = item["translation"]["zh"]
                answer = item["translation"]["en"]
                batch_answers.append(answer)
                
                # 添加目标文本
                target_text = template.format(text=question)
                all_target_prompts.append(target_text)
                
                # 添加相似度top1的源文本
                
                source_doc = all_data[str(item["sim_top1"])]
                source_text = template.format(text=source_doc["translation"]["zh"])
                all_source_prompts.append(source_text)
                    
                # 添加重用token top1的源文本
                source_doc = all_data[str(item["resued_top1"])]
                source_text = template.format(text=source_doc["translation"]["zh"])
                all_source_prompts.append(source_text)
                
                # batch_concat_len.append(len(item["concat_items"]))
                # for concat_item in item["concat_items"]:
                #     source_doc = all_data[str(concat_item)]
                #     source_text = template.format(text=source_doc["translation"]["zh"])
                #     all_source_prompts.append(source_text)
        
            # 批量计算full compute
            full_compute_outputs = KVShareNewPipeline.batch_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=512),
                all_target_prompts
            )
            batch_target_token_ids = []
            batch_target_prompts = []
            for idx,item in enumerate(batch_items):
                #
                batch_target_prompts.append(all_target_prompts[idx])
                batch_target_token_ids.append(full_compute_outputs[idx].prompt_token_ids)
                    
                # 添加重用token top1的源文本
                batch_target_prompts.append(all_target_prompts[idx])
                batch_target_token_ids.append(full_compute_outputs[idx].prompt_token_ids)
                
                # batch_target_prompts.append(all_target_prompts[idx])
                # batch_target_token_ids.append(full_compute_outputs[idx].prompt_token_ids)
            
            # 批量获取kv cache
            batch_source_key_values, batch_source_outputs = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=1),
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
            partial_batch_outputs = KVShareNewPipeline.partial_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=256),
                batch_target_prompts,
                reused_map_indices,
                unreused_map_indices,
                sample_selected_token_indices,
                target_kvcache
            )

            # 处理每个batch item的结果
            for idx, item in enumerate(batch_items):
                try:
                    # 获取full compute结果
                    full_compute_output = full_compute_outputs[0].outputs[0].text
                    item["output"] = full_compute_output
                    item["bleu"] = BLEU.compute(predictions=[full_compute_output], references=[batch_answers[idx]])
                    
                    sim_top1_output = partial_batch_outputs[idx*2].outputs[0].text
                    item["sim_top1_output"] = sim_top1_output
                    item["sim_top1_bleu"] = BLEU.compute(predictions=[sim_top1_output], references=[batch_answers[idx]])
                    
                    reused_top1_output = partial_batch_outputs[idx*2+1].outputs[0].text
                    item["reused_top1_output"] = reused_top1_output
                    item["reused_top1_bleu"] = BLEU.compute(predictions=[reused_top1_output], references=[batch_answers[idx]])
                    
                    # item["concat_output"] = partial_batch_outputs[idx*3+2].outputs[0].text
                    # item["concat_bleu"] = BLEU.compute(predictions=[item["concat_output"]], references=[batch_answers[idx]])
                    
                   
                    save_data.append(item)
                except Exception as e:
                    json.dump(save_data+has_run_data, open(output_path, "w"), indent=4, ensure_ascii=False)
                    print(f"处理item {idx}时出错: {str(e)}")
                    continue

        except Exception as e:
            json.dump(save_data+has_run_data, open(output_path, "w"), indent=4, ensure_ascii=False)
            print(f"处理批次时出错: {str(e)}")
            # print(f"错误详情: {traceback.format_exc()}")
            continue
    json.dump(save_data+has_run_data, open(output_path, "w"), indent=4, ensure_ascii=False)

def generate_output_data_by_concat(input_path,profiled_path,output_path,model_name,batch_size=4):
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    profiled_data = json.load(open(profiled_path,"r"))
    save_data = []
    
    all_data = data["all_data"]

    BLEU = evaluate.load('bleu')
    
    # 按batch_size分批处理数据
    for i in tqdm(range(0, len(profiled_data), batch_size), desc="Processing batches"):
        try:
            batch_items = profiled_data[i:i + batch_size]
            # real_need_run_items = []
           
            # batch_items = real_need_run_items
            # if len(batch_items)==0:
            #     continue
            
            # 准备所有prompt
            all_target_prompts = []
            all_source_prompts = []
            batch_answers = []
            batch_concat_len = []
            for item in batch_items:
                question = item["translation"]["zh"]
                answer = item["translation"]["en"]
                batch_answers.append(answer)
                
                # 添加目标文本
                target_text = template.format(text=question)
                all_target_prompts.append(target_text)
                

                batch_concat_len.append(len(item["concat_items"]))
                for concat_item in item["concat_items"]:
                    source_doc = all_data[str(concat_item)]
                    source_text = template.format(text=source_doc["translation"]["zh"])
                    all_source_prompts.append(source_text)
        
            # 批量计算full compute
            full_compute_outputs = KVShareNewPipeline.batch_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=512),
                all_target_prompts
            )
            batch_target_token_ids = []
            batch_target_prompts = []
            for idx,item in enumerate(batch_items):
                batch_target_prompts.append(all_target_prompts[idx])
                batch_target_token_ids.append(full_compute_outputs[idx].prompt_token_ids)
            
            # 批量获取kv cache
            batch_source_key_values, batch_source_outputs = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=1),
                all_source_prompts
            )
            
            batch_source_token_ids = []
            concat_prefix =0
            for concat_len in batch_concat_len:
                batch_source_token_ids.append(sum([batch_source_outputs[i].prompt_token_ids for i in range(concat_prefix,concat_prefix+concat_len)],[]))
                concat_prefix += concat_len
            
            # 批量编辑kv cache
            target_kvcache, reused_map_indices, unreused_map_indices, sample_selected_token_indices = KVEditor.batch_kvedit(
                batch_target_token_ids,
                batch_source_token_ids,
                batch_source_key_values
            )
            
            # 批量partial compute
            partial_batch_outputs = KVShareNewPipeline.partial_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=256),
                batch_target_prompts,
                reused_map_indices,
                unreused_map_indices,
                sample_selected_token_indices,
                target_kvcache
            )

            # 处理每个batch item的结果
            for idx, item in enumerate(batch_items):
                try:
                    # 获取full compute结果
                    # full_compute_output = full_compute_outputs[0].outputs[0].text
                    # item["output"] = full_compute_output
                    # item["bleu"] = BLEU.compute(predictions=[full_compute_output], references=[batch_answers[idx]])
                    
                    
                    item["concat_output"] = partial_batch_outputs[idx].outputs[0].text
                    item["concat_bleu"] = BLEU.compute(predictions=[item["concat_output"]], references=[batch_answers[idx]])
                    
                   
                    save_data.append(item)
                except Exception as e:
                    json.dump(save_data, open(output_path, "w"), indent=4, ensure_ascii=False)
                    print(f"处理item {idx}时出错: {str(e)}")
                    continue

        except Exception as e:
            json.dump(save_data, open(output_path, "w"), indent=4, ensure_ascii=False)
            print(f"处理批次时出错: {str(e)}")
            continue
    json.dump(save_data, open(output_path, "w"), indent=4, ensure_ascii=False)
    



def random_concat_opus(input_path,output_path,max_len=2000):
    model = SentenceTransformer('all-MiniLM-L6-v2',device='cuda:0')
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    data = json.load(open(input_path))
    global_id = 0
    save_data = {
        "all_data":{},
        "data":[]
    }
    
    for i in tqdm(range(max_len)):
        random_len = random.randint(2,5)
        
        items = random.sample(data,random_len)
        
        for item in items:
            if item["id"] not in save_data["all_data"]:
                save_data["all_data"][item["id"]] = item
        
        en = " ".join([item["translation"]["en"] for item in items])
        zh = " ".join([item["translation"]["zh"] for item in items])
        
        # 计算top1的相似度
        batch_zh = [item["translation"]["zh"] for item in items]
        current_embedding = model.encode(zh)
        batch_embedding = model.encode(batch_zh)
        sim_matrix = np.dot(batch_embedding,current_embedding)
        sim_top1 = items[np.argmax(sim_matrix)]

        resued_nums = [len(tokenizer.encode(item["translation"]["zh"])) for item in items]
        reused_top1 = items[np.argmax(resued_nums)]
        
        save_data["data"].append({
            "id":global_id,
            "translation": {
                "en":en,
                "zh":zh
            },
            "sim_top1":sim_top1["id"],
            "resued_top1": reused_top1["id"],
            "concat_items": [item["id"] for item in items]
        })
        global_id += 1
    json.dump(save_data,open(output_path,"w"),ensure_ascii=False,indent=4)

def plot_bleu_comparison(input_path: str, save_path: str = "examples/pipeline/images/opus_bleu_comparison.png"):
    data = json.load(open(input_path,"r"))
    
    # 提取不同方法的BLEU分数
    full_compute_bleu = []
    sim_top1_bleu = []
    reused_top1_bleu = []
    concat_bleu = []
    for item in data:
        if "bleu" in item and "sim_top1_bleu" in item and "reused_top1_bleu" in item and "concat_bleu" in item:
            full_compute_bleu.append(item["bleu"]["bleu"])
            sim_top1_bleu.append(item["sim_top1_bleu"]["bleu"])
            reused_top1_bleu.append(item["reused_top1_bleu"]["bleu"])
            concat_bleu.append(item["concat_bleu"]["bleu"])
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 计算并绘制CDF
    def plot_cdf(data, label, color):
        sorted_data = np.sort(data)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, p, label=label, color=color)
    
    # 绘制各个方法的CDF曲线
    plot_cdf(full_compute_bleu, 'Full Compute', 'blue')
    plot_cdf(sim_top1_bleu, 'Sim Top1', 'red')
    plot_cdf(reused_top1_bleu, 'Reused Top1', 'green')
    plot_cdf(concat_bleu, 'Concat', 'purple')
    
    # 添加理想参考线
    plt.axvline(x=1.0, color='black', linestyle='--', label='Ideal (BLEU=1.0)')
    
    # 设置图形属性
    plt.xlabel('BLEU Score')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of BLEU Scores for Different Methods')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴范围，确保能看到1.0
    plt.xlim(0, 1.1)
    
    # 保存图形
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh.json"
    output_path = "examples/dataset/data/opus/opus_dataset_en-zh_multi_concat.json"
    # random_concat_opus(input_path,output_path,max_len=1000)
    
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    template = qwen_template
    # template = llama3_template_text
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    
    profile_path = f"examples/dataset/data/opus/opus_dataset_en-zh_multi_concat_output_by_{model_name.split('/')[-1]}.json"
    profile_path_by_concat = f"examples/dataset/data/opus/opus_dataset_en-zh_multi_concat_output_by_concat_{model_name.split('/')[-1]}.json"
    # generate_output_data(output_path,profile_path,model_name,batch_size=8)
    # generate_output_data_by_concat(output_path,profile_path,profile_path_by_concat,model_name,batch_size=24)
    
    # 绘制BLEU分数比较图
    plot_bleu_comparison(profile_path_by_concat)
    