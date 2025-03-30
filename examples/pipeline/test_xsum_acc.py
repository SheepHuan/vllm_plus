import json
from kvshare_new_pipeline import KVShareNewPipeline
from nll_demo import calculate_nll
from vllm.sampling_params import SamplingParams
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
import os
import random

def generate_output_data(input_path: str, output_path: str, test_high_sim: bool = False, test_max_resued: bool = False):
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name,device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    # data = random.sample(json.load(open(input_path,"r")),100)
    all_documents = data["all_documents"]
    similar_docs = data["similar_docs"]
    save_data = []
    similar_docs = random.sample(similar_docs,100)
    for idx,item in tqdm(enumerate(similar_docs),total=len(similar_docs)):
        try:
            if test_high_sim:
                source_text = all_documents[str(item["high_similarity_doc"]["id"])]["document"]
                target_text = item["document"]
            elif test_max_resued:
                source_text = all_documents[str(item["max_resued_doc"])]["document"]
                target_text = item["document"]
            template="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nYou need to summerize bellow document to one sentence. Document:\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            source_text = template.format(prompt=source_text)
            target_text = template.format(prompt=target_text)
            
            sampling_params_only_one = SamplingParams(temperature=0, max_tokens=1)
            source_kvcache,source_token_ids = KVShareNewPipeline.get_kvcache_by_full_compute(pipeline.model,sampling_params_only_one,source_text)
            sampling_params = SamplingParams(temperature=0, max_tokens=512)
            target_gt_output,target_token_ids,ttft_time = KVShareNewPipeline.full_compute(pipeline.model,sampling_params,target_text)
            
            diff_report = KVShareNewPipeline.find_texts_differences(source_token_ids,target_token_ids)
            modified_kvcache,reused_map_indices,unused_map_indices = KVShareNewPipeline.apply_changes2kvcache(source_token_ids,
                                                                                                            target_token_ids,
                                                                                                            source_kvcache,
                                                                                                            diff_report)
            sampling_params = SamplingParams(temperature=0, max_tokens=512)
            modified_output,modified_token_ids,ttft_time = KVShareNewPipeline.partial_compute(pipeline.model,
                                                                                            sampling_params,
                                                                                            target_text,
                                                                                            reused_map_indices,
                                                                                            unused_map_indices,
                                                                                            modified_kvcache)

            save_data.append({
                "source_text":source_text,
                "target_text":target_text,
                "target_output_full_compute":target_gt_output,
                "target_output_partial_compute":modified_output,      
                "summary":item["summary"]
            })
            
        except Exception as e:
            print(e)
            continue
    json.dump(save_data,open(output_path,"w"),indent=4,ensure_ascii=False)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    input_path = "examples/dataset/data/xsum/all-MiniLM-L6-v2_train_similar_docs_topk50_high_similarity.json"
    output_path = "examples/dataset/data/xsum/all-MiniLM-L6-v2_train_similar_docs_topk50_high_similarity_not_high_reused_output.json"
    generate_output_data(input_path,output_path,test_high_sim=True,test_max_resued=False)
