import json
from vllm import LLM,SamplingParams
from tqdm import tqdm
import os
from kvshare_new_pipeline import KVShareNewPipeline
from transformers import AutoTokenizer
import random

from edit2 import find_text_differences,apply_change
def test_user_ins_ppl(dataset_path,model_name,save_path):
    pipeline = KVShareNewPipeline(model_name)
    dataset = json.load(open(dataset_path, "r"))
    similar_pairs = dataset["similar_pairs"]
    all_texts = dataset["all_texts"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    save_data = []
    similar_pairs = random.sample(similar_pairs,min(len(similar_pairs),150))
    template_text = "<|im_start|>You are Qwen, created by Alibaba. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"
    for item in tqdm(similar_pairs):
        target_id = item["id"]
        target_text = all_texts[str(target_id)]
        if len(tokenizer.encode(target_text)) < 50:
            continue
        targert_full_compute_output,target_token_ids,_ = KVShareNewPipeline.full_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),target_text)
        item["full_compute_output"] = targert_full_compute_output
        for similar_doc in item["high_similarity_top5"]:
            similar_doc_id = similar_doc["id"]
            similar_doc_text = all_texts[str(similar_doc_id)]
            
            past_key_values,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=1),similar_doc_text)
            
            diff_report = find_text_differences(source_token_ids,target_token_ids)
            target_kvcache,reused_map_indices,unreused_map_indices = apply_change(source_token_ids,target_token_ids,past_key_values,diff_report)
            
            partial_compute_output,partial_token_ids,_ = KVShareNewPipeline.partial_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),target_text,reused_map_indices,unreused_map_indices,target_kvcache)
            similar_doc["partial_compute_output"] = partial_compute_output
            
        for similar_doc in item["high_token_reused_top5"]:
            
            similar_doc_id = similar_doc["id"]
            similar_doc_text = all_texts[str(similar_doc_id)]
            past_key_values,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=1),similar_doc_text)
            
            diff_report = find_text_differences(source_token_ids,target_token_ids)
            target_kvcache,reused_map_indices,unreused_map_indices = apply_change(source_token_ids,target_token_ids,past_key_values,diff_report)
            
            partial_compute_output,partial_token_ids,_ = KVShareNewPipeline.partial_compute(pipeline.model,SamplingParams(temperature=0.0,max_tokens=512),target_text,reused_map_indices,unreused_map_indices,target_kvcache)
            similar_doc["partial_compute_output"] = partial_compute_output


        save_data.append(item)
    json.dump(save_data, open(save_path, "w"), ensure_ascii=False, indent=4)
    # return response

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset_path = "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331.json"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    save_path = "examples/dataset/data/insturctionv2/instruction_wildv2_similar_250331_output.json"
    test_user_ins_ppl(dataset_path,model_name,save_path)
