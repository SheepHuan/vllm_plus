import datasets
import json

from log import setup_logger
import os
import evaluate
import time
def wmt_kvshare_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin,LLamaKVSharePlugin
    kvshare =  LLamaKVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.4,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,
                            milvus_connection_name="kvshare_wmt",
                            milvus_database_path="examples/pipeline/data/milvus_wmt.db",
                            spacy_pipeline="zh_core_web_sm"
                            )
    
    output_data = []
    bleu_evaluator = evaluate.load("bleu")
    data = json.load(open(input_path, "r"))
    tokenizer = kvshare.tokenizer
    
    for item in data:
        input_text = item["input"]
        gt_text = item["gt"]
    
        # chunks = kvshare.text_split(input_text)
        # chunks = [f"{chunk}\n" for chunk in chunks]
        prompts = "将下面的中文文本翻译成英文。" + input_text
        # kvshare.use_semantic_search = False
        # kvshare.save_kvcache = True
        # output_text,metric = kvshare.generate_with_kvshare(prompts,max_tokens=1)
        # pass
        
        kvshare.use_semantic_search = True
        kvshare.save_kvcache = False
        kvshare.enable_show_log =True
        try:
            kvshare._set_confuse_metadata_raw("recomp_ratio",0.15)
            output_text,metric = kvshare.generate_with_kvshare(prompts,max_tokens=2048) 
            score = bleu_evaluator.compute(predictions=[output_text],references=[gt_text])
            
        except Exception as e:
            print(e)
            output_text=""
            metric = {}
            score={}
        
        
        
        item["kvshare"] = {
            "output": output_text,
            "metric": metric,
            "bleu": score
        }
        
        output_data.append(item)

        with open(save_path, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)


def wmt_full_compute_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin
    kvshare = KVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True
                            )
    
    output_data = []
    bleu_evaluator = evaluate.load("bleu")
    data = json.load(open(input_path, "r"))
    # tokenizer = kvshare.tokenizer
    
    for item in data:
        input_text = item["input"]
        gt_text = item["gt"]
        prompts = "将下面的中文文本翻译成英文。" + input_text
        output_text,metric = kvshare.generate(prompts,max_tokens=2048)
        score = bleu_evaluator.compute(predictions=[output_text],references=[gt_text])
        item["full_compute"] = {
            "output": output_text,
            "metric": metric,
            "bleu": score
        }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def wmt_gptcache_bench(model_name,input_path,save_path):
    """
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
    
    """
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
    
    bleu_evaluator = evaluate.load("bleu")
    
    for item in data:
        input_text = item["input"]
        gt_text = item["gt"]
        prompts = "将下面的中文文本翻译成英文。" + input_text
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
            score = bleu_evaluator.compute(predictions=[output_text],references=[gt_text])
            item["gptcache"] = {
                "output": output_text,
                "metric": {"ttft": time.time() - start_time},
                "bleu": score
            }
        except Exception as e:
            print(e)
            item["gptcache"] = {
                "output": "",
                "metric": {},
                "bleu": {}
            }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
def wmt_prefix_caching_bench(model_name,input_path,save_path):
    from kvshare_lib import KVSharePlugin
    kvshare = KVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
                            max_model_len=8192,
                            multi_step_stream_outputs=True,
                            disable_async_output_proc=True,
                            enable_prefix_caching=True,
                            milvus_connection_name="prefix_caching_wmt",
                            milvus_database_path="examples/pipeline/data/milvus_wmt.db",
                            )
    
    output_data = []
    bleu_evaluator = evaluate.load("bleu")
    data = json.load(open(input_path, "r"))
    # tokenizer = kvshare.tokenizer
    
    for item in data:
        input_text = item["input"]
        gt_text = item["gt"]
        prompts = "将下面的中文文本翻译成英文。" + input_text
        output_text,metric = kvshare.generate(prompts,max_tokens=2048)
        score = bleu_evaluator.compute(predictions=[output_text],references=[gt_text])
        item["prefix_caching"] = {
            "output": output_text,
            "metric": metric,
            "bleu": score
        }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    


def benchmark_metric(json_file_path,tag):
    json_file = open(json_file_path,"r")
    data = json.load(json_file)
    avg_bleu = []
    avg_ttft = []
    for item in data:
        metric = item[tag]["metric"]
        if "bleu" not in item[tag]["bleu"]:
            continue
        bleu = item[tag]["bleu"]["bleu"]
        ttft = metric["ttft"]
        avg_bleu.append(bleu)
        avg_ttft.append(ttft)
    print(f"avg_bleu: {sum(avg_bleu)/len(avg_bleu)}")
    print(f"avg_ttft: {sum(avg_ttft)/len(avg_ttft)}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["VLLM_USE_MODELSCOPE"] = "true"
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
    model_name = "/root/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
    input_path = "/root/code/vllm_plus/examples/dataset/data/wmt_dataset_benchmark.json"
    # save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_cacheblend_{model_name.split('/')[-1]}.json"
    # wmt_cacheblend_bench(model_name,input_path,save_path)
    
    # save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_kvshare_{model_name.split('/')[-1]}.json"
    # wmt_kvshare_bench(model_name,input_path,save_path)
    
    # save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_full_compute_{model_name.split('/')[-1]}.json"
    # wmt_full_compute_bench(model_name,input_path,save_path)
    
    # save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_gptcache_{model_name.split('/')[-1]}.json"
    # wmt_gptcache_bench(model_name,input_path,save_path)
    
    # save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_prefix_caching_{model_name.split('/')[-1]}.json"
    # wmt_prefix_caching_bench(model_name,input_path,save_path)
    

    # kvshare_path= "examples/pipeline/data/wmt_bench_kvshare_Qwen2.5-7B-Instruct.json"
    # kvshare_path = "examples/pipeline/data/wmt_bench_kvshare_Meta-Llama-3.1-8B-Instruct.json"
    # benchmark_metric(kvshare_path,"kvshare")
    
    
    # cacheblend_path= "examples/pipeline/data/wmt_bench_cacheblend_Qwen2.5-7B-Instruct.json"
    # benchmark_metric(cacheblend_path,"cacheblend")
    
    # full_compute_path= "examples/pipeline/data/wmt_bench_full_compute_Qwen2.5-7B-Instruct.json"
    # full_compute_path= "examples/pipeline/data/wmt_bench_full_compute_Meta-Llama-3.1-8B-Instruct.json"
    # benchmark_metric(full_compute_path,"full_compute")
    
    # gptcache_path= "examples/pipeline/data/wmt_bench_gptcache_Qwen2.5-7B-Instruct.json"
    gptcache_path = "examples/pipeline/data/wmt_bench_gptcache_Meta-Llama-3.1-8B-Instruct.json"
    benchmark_metric(gptcache_path,"gptcache")
    
    # prefix_caching_path= "examples/pipeline/data/wmt_bench_prefix_caching_Qwen2.5-7B-Instruct.json"
    # prefix_caching_path= "examples/pipeline/data/wmt_bench_prefix_caching_Meta-Llama-3.1-8B-Instruct.json"
    # benchmark_metric(prefix_caching_path,"prefix_caching")
    