import datasets
import json
from kvshare_lib import KVSharePlugin
from log import setup_logger
import os
import evaluate

def wmt_kvshare_bench(model_name,input_path,save_path):
    # logger = setup_logger("wmt_cacheblend_bench")
    kvshare = KVSharePlugin(llm_model_name=model_name,
                            sentence_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                            gpu_memory_utilization=0.6,
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
        prompts = "Translate the following text from Chinese to English.\n" + input_text
        # kvshare.use_semantic_search = False
        # kvshare.save_kvcache = True
        # output_text,metric = kvshare.generate_with_kvshare(prompts,max_tokens=1)
        # pass
        
        kvshare.use_semantic_search = True
        kvshare.save_kvcache = False
        kvshare.enable_show_log =True
        try:
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

def wmt_cacheblend_bench(model_name,input_path,save_path):
    # logger = setup_logger("wmt_cacheblend_bench")
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
    tokenizer = kvshare.tokenizer
    
    for item in data:
        input_text = item["input"]
        gt_text = item["gt"]
    
        chunks = kvshare.text_split(input_text)
        chunks = [f"{chunk}\n" for chunk in chunks]
        # chunks = [f"Translate the following text from Chinese to English.\n"] + chunks
        # input_ids =[]
        # for chunk in chunks:
        #     input_ids.extend(tokenizer.encode(chunk))
        # prompt = tokenizer.decode(input_ids)
        # real_input_ids = tokenizer.encode(prompt)
        # print(len(real_input_ids),len(input_ids),len(real_input_ids)==len(input_ids))
        
        output_text,metric = kvshare.generate_with_cacheblend(chunks,"Translate the following text from Chinese to English.\n")
        score = bleu_evaluator.compute(predictions=[output_text],references=[gt_text])
        item["cacheblend"] = {
            "output": output_text,
            "metric": metric,
            "bleu": score
        }
        
        output_data.append(item)

    with open(save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def wmt_gptcache_bench(input_path,save_path):
    from gptcache.manager import get_data_manager
    from gptcache.core import cache, Cache
    from gptcache.adapter import openai

    cache.init(data_manager=get_data_manager())
    os.environ["OPENAI_API_KEY"] = "API KEY"
    cache.set_openai_key()

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': "What's 1+1? Answer in one word."}
        ],
        temperature=0,
        stream=True  # this time, we set stream=True
    )

    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message

    full_reply_content = ''.join([m.get('content', '') for m in collected_messages])

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    input_path = "/root/code/vllm_plus/examples/dataset/data/wmt_dataset_benchmark.json"
    # save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_cacheblend_{model_name.split('/')[-1]}.json"
    # wmt_cacheblend_bench(model_name,input_path,save_path)
    
    save_path = f"/root/code/vllm_plus/examples/pipeline/data/wmt_bench_kvshare_{model_name.split('/')[-1]}.json"
    wmt_kvshare_bench(model_name,input_path,save_path)
