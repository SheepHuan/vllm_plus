from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
import os

def generate_with_kvshare(doc_prompts,q_prompt):
    for sample_idx in range(1,2):
        f = open(f"/root/code/vllm/examples/inputs/{sample_idx}.json")
        ex = json.load(f)
        chunk_num = ex['chunk_num']
        doc_prompts = [ex[f'{i}'] for i in range(chunk_num)]
        q_prompt = ex['query']
        doc_chunk_ids = [tokenizer.encode(doc) for doc in doc_prompts]
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        sys_ids = tokenizer.encode(system_prompt)
        question_prompt = "<|im_start|>user\n" + q_prompt + "<|im_end|>\n<|im_start|>assistant\n"
        question_ids = tokenizer.encode(question_prompt)


        # Create a sampling params object.
        sampling_params = SamplingParams(temperature=0, max_tokens=1)

        # Create an tokenizer and LLM.
        cache_fuse_metadata = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['check'] = False

        s_start = [151644]
        s_end = [151645]
        # s_end_len = len(s_end)
        old_kvs = []

        doc_chunk_ids = [sys_ids]+doc_chunk_ids
        doc_chunk_ids = [s_start + chunk_ids + s_end for chunk_ids in doc_chunk_ids]
        
        doc_chunk_ids = doc_chunk_ids + [question_ids]

        # last_len = len([q_ids])

        cache_fuse_metadata['collect'] = True
        cache_fuse_metadata["check"] = False

        num_layer = len(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers)
        chunk_past_key_values = []
        
        # Concatenate old KVs
        for i in range(len(doc_chunk_ids)):
            prompts = [tokenizer.decode(doc_chunk_ids[i])]
            llm.generate(prompts, sampling_params)
            
            llm_layers = llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
            for j in range(num_layer):
                past_key_values = llm_layers[j].self_attn.hack_kv
                temp_k = past_key_values[0].clone()
                temp_v = past_key_values[1].clone()
                # if i in [0,len(doc_chunk_ids)-1]:
                #     temp_k = past_key_values[0].clone() # do not chage with s_start_1
                #     temp_v = past_key_values[1].clone()
                # else:
                #     temp_k = past_key_values[0][len(s_start):len(doc_chunk_ids[i])-len(s_end)].clone()
                #     temp_v = past_key_values[1][len(s_start):len(doc_chunk_ids[i])-len(s_end)].clone()    

                if i == 0:
                    chunk_past_key_values.append([temp_k, temp_v])
                else:
                    #pdb.set_trace()
                    chunk_past_key_values[j][0] = torch.cat((chunk_past_key_values[j][0],temp_k), dim=0)
                    chunk_past_key_values[j][1] = torch.cat((chunk_past_key_values[j][1],temp_v), dim=0)
            #print(temp_k.shape[0])
            llm.llm_engine.model_executor.driver_worker.model_runner.model.model.old_kvs = chunk_past_key_values
            
        input_ids = []

        for i in range(len(doc_chunk_ids)):
            temp_ids = doc_chunk_ids[i]
            # if i in [0,len(doc_chunk_ids)-1]:
            #     temp_ids = doc_chunk_ids[i]
            # else:
            #     temp_ids = doc_chunk_ids[i][len(s_start):len(doc_chunk_ids[i])-len(s_end)]
            input_ids += temp_ids
            
        input_prompt = tokenizer.decode(input_ids)
        # print(input_prompt)
        new_input_ids = tokenizer.encode(input_prompt)
        if len(input_ids) < len(new_input_ids):
            b = [input_ids[i]==new_input_ids[i] for i in range(len(input_ids))]
            index = b.index(False)
            print("--------------------------------")
            print(tokenizer.decode(input_ids[index-5:index+5]))
            print(tokenizer.decode(new_input_ids[index-5:index+5]))
            print("--------------------------------")
            
        # exit(0)
        sampling_params = SamplingParams(temperature=0, max_tokens=10)
        cache_fuse_metadata["check"] = True
        cache_fuse_metadata['collect'] = False
        cache_fuse_metadata['suffix_len'] = 0
        output = llm.generate([input_prompt], sampling_params)
        print(f"Cached generation: {output[0].outputs[0].text}")
        print(f"TTFT with cache: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
        
    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    cache_fuse_metadata["check"] = False
    cache_fuse_metadata['collect'] = False
    output = llm.generate([input_prompt], sampling_params)
    print(f"Normal generation: {output[0].outputs[0].text}")
    print(f"TTFT with full prefill: {output[0].metrics.first_token_time-output[0].metrics.first_scheduled_time}")
    print("------------")


if __name__ == "__main__":
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"


    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm = LLM(model=model_name, gpu_memory_utilization=0.8,max_model_len=8192,
          multi_step_stream_outputs=True,
          disable_async_output_proc=True)