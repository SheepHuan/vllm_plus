import json

from log import setup_logger
import os
import evaluate
import time
import numpy as np

def samsum_full_compute_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin
    kvshare = KVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,)
    
    output_data = []
    data = json.load(open(input_path, "r"))
    
    rouge_evaluator = evaluate.load('rouge')
    
    for item in data:
        input_text = item["dialogue"]
        gt_text = item["summary"]
        prompts = "Summarize the following dialogue. Do not change the original meaning. Do not add any other information. And do not exceed 500 words.\n" + input_text
        output_text,metric = kvshare.generate(prompts,max_tokens=2048)
        score = rouge_evaluator.compute(predictions=[output_text],references=[gt_text])
        item["full_compute"] = {
            "output": output_text,
            "metric": metric,
            "rouge": score
        }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
def samsum_full_compute_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin
    kvshare = KVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,)
    
    output_data = []
    data = json.load(open(input_path, "r"))
    
    rouge_evaluator = evaluate.load('rouge')
    
    for item in data:
        input_text = item["dialogue"]
        gt_text = item["summary"]
        prompts = "Summarize the following dialogue. Do not change the original meaning. Do not add any other information. And do not exceed 500 words.\n" + input_text
        output_text,metric = kvshare.generate(prompts,max_tokens=2048)
        score = rouge_evaluator.compute(predictions=[output_text],references=[gt_text])
        item["full_compute"] = {
            "output": output_text,
            "metric": metric,
            "rouge": score
        }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        

def samsum_prefix_caching_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin
    kvshare = KVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,
                            enable_prefix_caching=True,
                            milvus_connection_name="prefix_caching"
                            )
    
    output_data = []
    data = json.load(open(input_path, "r"))
    
    rouge_evaluator = evaluate.load('rouge')
    
    for item in data:
        input_text = item["dialogue"]
        gt_text = item["summary"]
        prompts = "Summarize the following dialogue. Do not change the original meaning. Do not add any other information. And do not exceed 500 words.\n" + input_text
        output_text,metric = kvshare.generate(prompts,max_tokens=2048)
        score = rouge_evaluator.compute(predictions=[output_text],references=[gt_text])
        item["prefix_caching"] = {
            "output": output_text,
            "metric": metric,
            "rouge": score
        }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def samsum_kvshare_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin,LLamaKVSharePlugin
    kvshare = LLamaKVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,
                            milvus_connection_name=f"kvshare_samsum_llama3",
                            milvus_database_path="examples/pipeline/data/milvus_samsum.db",
                            )
    output_data = []
    data = json.load(open(input_path, "r"))
    
    rouge_evaluator = evaluate.load('rouge')
    
    for item in data:
        input_text = item["dialogue"]
        gt_text = item["summary"]
        prompts = "Summarize the following dialogue. Do not change the original meaning. Do not add any other information. And do not exceed 500 words.\n" + input_text
        try:
            kvshare._set_confuse_metadata_raw("recomp_ratio",0.15)
            output_text,metric = kvshare.generate_with_kvshare(prompts,max_tokens=2048)
            score = rouge_evaluator.compute(predictions=[output_text],references=[gt_text])
            item["kvshare"] = {
                "output": output_text,
                "metric": metric,
                "rouge": score
            }
        except Exception as e:
            print(e)
            item["kvshare"] = {
                "output": "",
                "metric": {},
                "rouge": {}
            }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        
def samsum_gptcache_bench(model_name,input_path,save_path):
    def response_text(openai_resp):
        return openai_resp['choices'][0]['message']['content']

    from gptcache import cache
    from gptcache.adapter import openai
    from gptcache.embedding import Onnx,SBERT
    from gptcache.manager import CacheBase, VectorBase, get_data_manager
    from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
    model = SBERT(model="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=model.dimension))
    cache.init(
        embedding_func=model.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        )

    output_data = []
    data = json.load(open(input_path, "r"))
    
    rouge_evaluator = evaluate.load('rouge')
    
    for item in data:
        input_text = item["dialogue"]
        gt_text = item["summary"]
        prompts = "Summarize the following dialogue. Do not change the original meaning. Do not add any other information. And do not exceed 500 words.\n" + input_text
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompts
                    }
                ],
            )
            output_text = response_text(response)
            score = rouge_evaluator.compute(predictions=[output_text],references=[gt_text])
            item["gptcache"] = {
                "output": output_text,
                "metric": {"ttft": time.time() - start_time},
                "rouge": score
            }
        except Exception as e:
            print(e)
            item["gptcache"] = {
                "output": "",
                "metric": {},
                "rouge": {}
            }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        
def count_data(input_path,tag):
    data = json.load(open(input_path, "r"))
    ttft = []
    rougeL = []
    for item in data:
        if item[tag]["metric"] != {} and item[tag]["rouge"] != {}:
            rougeL.append(item[tag]["rouge"]["rougeL"])
            ttft.append(item[tag]["metric"]["ttft"])
    print(f"ttft: {np.mean(ttft)}")
    print(f"rougeL: {np.mean(rougeL)}")
        
if __name__ == "__main__":
    """
    export VLLM_USE_MODELSCOPE=true
    export CUDA_VISIBLE_DEVICES=1
    export VLLM_ATTENTION_BACKEND="XFORMERS"
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --trust-remote-code --gpu-memory-utilization 0.6
    vllm serve LLM-Research/Meta-Llama-3.1-8B-Instruct --trust-remote-code --gpu-memory-utilization 0.6
    
    
    
    """
    
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    # samsum_full_compute_bench(model_name,"examples/dataset/data/samsum_group.json",f"examples/pipeline/data/samsum_full_compute_{model_name.split('/')[-1]}.json")
    samsum_kvshare_bench(model_name,"examples/dataset/data/samsum_group.json",f"examples/pipeline/data/samsum_kvshare_{model_name.split('/')[-1]}.json")
    # samsum_gptcache_bench(model_name,"examples/dataset/data/samsum_group.json",f"examples/pipeline/data/samsum_gptcache_{model_name.split('/')[-1]}.json")
    # samsum_prefix_caching_bench(model_name,"examples/dataset/data/samsum_group.json",f"examples/pipeline/data/samsum_prefix_caching_{model_name.split('/')[-1]}.json")
    # count_data("examples/pipeline/data/samsum_full_compute_Qwen2.5-7B-Instruct.json","full_compute")
    # count_data("examples/pipeline/data/samsum_full_compute_Meta-Llama-3.1-8B-Instruct.json","full_compute")
    # count_data("examples/pipeline/data/samsum_kvshare_Qwen2.5-7B-Instruct.json","kvshare")
    # count_data("examples/pipeline/data/samsum_kvshare_Meta-Llama-3.1-8B-Instruct.json","kvshare")
    # count_data("examples/pipeline/data/samsum_gptcache_Qwen2.5-7B-Instruct.json","gptcache")
    # count_data("examples/pipeline/data/samsum_gptcache_Meta-Llama-3.1-8B-Instruct.json","gptcache")
    # count_data("examples/pipeline/data/samsum_prefix_caching_Qwen2.5-7B-Instruct.json","prefix_caching")
    # count_data("examples/pipeline/data/samsum_prefix_caching_Meta-Llama-3.1-8B-Instruct.json","prefix_caching")
    