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
from matplotlib import font_manager 
import traceback
# download the font files and save in this fold
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
 
font_files = font_manager.findSystemFonts(fontpaths=font_path)
 
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

qwen_template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\nTranslate the following text from Chinese to English:\n{text}\n<|im_end|>\n<|im_start|>assistant\n"""
llama3_template_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Translate the following text from Chinese to English:\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


def generate_output_data(input_path: str, output_path: str, model_name = "Qwen/Qwen2.5-7B-Instruct", batch_size=4):
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    
    all_data = data["all_translations"]
    similar_pairs = data["similar_pairs"]
    similar_pairs = random.sample(similar_pairs, 2000)
    save_data = []
    if os.path.exists(output_path):
        has_run_data = json.load(open(output_path,"r"))
        has_run_ids = set()
        for item in has_run_data:
            has_run_ids.add(item["id"])
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
                question = all_data[str(item["id"])]["zh"]
                answer = all_data[str(item["id"])]["en"]
                batch_answers.append(answer)
                
                # 添加目标文本
                target_text = template.format(text=question)
                all_target_prompts.append(target_text)
                
                # 添加相似度top5的源文本
                for sim_item in item["cosine_similarity_top5"]:
                    source_doc = all_data[str(sim_item["id"])]
                    source_text = template.format(text=source_doc["zh"])
                    all_source_prompts.append(source_text)
                    
                # 添加重用token top5的源文本
                for reused_item in item["reused_token_num_top5"]:
                    source_doc = all_data[str(reused_item["id"])]
                    source_text = template.format(text=source_doc["zh"])
                    all_source_prompts.append(source_text)
                    
            # 批量计算full compute
            full_compute_outputs = KVShareNewPipeline.batch_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=256),
                all_target_prompts
            )
            batch_target_token_ids = []
            batch_target_prompts = []
            for idx,item in enumerate(batch_items):
                for sim_item in item["cosine_similarity_top5"]:
                    batch_target_prompts.append(all_target_prompts[idx])
                    batch_target_token_ids.append(full_compute_outputs[idx].prompt_token_ids)
                    
                # 添加重用token top5的源文本
                for reused_item in item["reused_token_num_top5"]:
                    batch_target_prompts.append(all_target_prompts[idx])
                    batch_target_token_ids.append(full_compute_outputs[idx].prompt_token_ids)
            
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
                    
                    # 处理相似度top5的结果
                    profile_similar_top5_docs = []
                    sim_start_idx = idx * (len(item["cosine_similarity_top5"]) + len(item["reused_token_num_top5"]))
                    for j, sim_item in enumerate(item["cosine_similarity_top5"]):
                        partial_output = partial_batch_outputs[sim_start_idx + j].outputs[0].text
                        profile_similar_top5_docs.append({
                            "id": sim_item["id"],
                            "output": partial_output,
                            "bleu": BLEU.compute(predictions=[partial_output], references=[batch_answers[idx]]),
                            "cosine_similarity": sim_item["similarity"]
                        })
                    item["cosine_similarity_top5"] = profile_similar_top5_docs
                    
                    # 处理重用token top5的结果
                    profile_reused_token_num_top5_docs = []
                    reused_start_idx = sim_start_idx + len(item["cosine_similarity_top5"])
                    for j, reused_item in enumerate(item["reused_token_num_top5"]):
                        partial_output = partial_batch_outputs[reused_start_idx + j].outputs[0].text
                        profile_reused_token_num_top5_docs.append({
                            "id": reused_item["id"],
                            "output": partial_output,
                            "bleu": BLEU.compute(predictions=[partial_output], references=[batch_answers[idx]]),
                            "reused_token_num": reused_item["reused_token_num"]
                        })
                    item["reused_token_num_top5"] = profile_reused_token_num_top5_docs
            
                   
                    save_data.append(item)
                except Exception as e:
                    print(f"处理item {idx}时出错: {str(e)}")
                    continue
                    
            # # 定期保存结果
            # if i % 200 ==0:
            #     json.dump(save_data, open(output_path, "w"), indent=4, ensure_ascii=False)
            
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")
            continue
    json.dump(save_data+has_run_data, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
def plot_bleu_acc(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)
    bleu_acc = []
    for item in data:
        bleu_acc.append(item["bleu"])
    plt.plot(bleu_acc)
    plt.show()

def plot_bleu_comparison(input_path: str, save_path: str = "examples/pipeline/images/opus_bleu_comparison.png"):
    """对比全量计算、相似度top1和重用token top1的BLEU分数分布"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 收集三种方法的BLEU分数
    full_compute_bleu = []
    similarity_top1_bleu = []
    reused_token_top1_bleu = []
    
    for item in data:
        try:
            if item["bleu"]["bleu"] >= 0.001:
                full_compute_bleu.append(item["bleu"]["bleu"])
            
            # 相似度top1的BLEU
            if item["cosine_similarity_top5"] and len(item["cosine_similarity_top5"]) > 0:
                similarity_top1_bleu.append(item["cosine_similarity_top5"][0]["bleu"]["bleu"])
            
            # 重用token top1的BLEU
            if item["reused_token_num_top5"] and len(item["reused_token_num_top5"]) > 0:
                reused_token_top1_bleu.append(item["reused_token_num_top5"][0]["bleu"]["bleu"])
        except Exception as e:
            continue
    
    # 定义统一的颜色方案
    colors = {
        'Full Compute': 'red',      # 红色
        'Similarity Top1': 'green',    # 绿色
        'Reused Token Top1': 'blue'   # 蓝色
    }
    labels = list(colors.keys())
    
    # 创建单个图
    plt.figure(figsize=(6, 4))
    
    # 添加ChatGPT参考线
    plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Ground Truth')
    
    # 绘制CDF曲线
    data_to_plot = [full_compute_bleu, similarity_top1_bleu, reused_token_top1_bleu]
    
    # 收集所有均值用于统一显示
    means = []
    for scores, label in zip(data_to_plot, labels):
        # 计算CDF
        sorted_scores = np.sort(scores)
        p = np.arange(1, len(scores) + 1) / len(scores)
        
        # 绘制CDF曲线
        plt.plot(sorted_scores, p, label=label, color=colors[label], alpha=0.7)
        
        # 收集均值
        mean_value = np.mean(scores)
        means.append((label, mean_value))
    
    # 统一显示所有均值
    mean_text = "Means:\n" + "\n".join([f"{label}: {mean:.3f}" for label, mean in means])
    plt.text(0.5, 0.5, mean_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('BLEU Score')
    plt.ylabel('Cumulative Probability')
    # plt.title('BLEU Score Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # 将图例放在图表右上角
    plt.legend(loc='lower center')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # 打印详细统计信息
    print("\nBLEU分数统计:")
    print("-" * 50)
    print(f"全量计算:")
    print(f"  样本数量: {len(full_compute_bleu)}")
    print(f"  平均BLEU: {np.mean(full_compute_bleu):.4f} ± {np.std(full_compute_bleu):.4f}")
    print(f"  中位数: {np.median(full_compute_bleu):.4f}")
    print(f"  范围: [{np.min(full_compute_bleu):.4f}, {np.max(full_compute_bleu):.4f}]")
    
    print(f"\n相似度Top1:")
    print(f"  样本数量: {len(similarity_top1_bleu)}")
    print(f"  平均BLEU: {np.mean(similarity_top1_bleu):.4f} ± {np.std(similarity_top1_bleu):.4f}")
    print(f"  中位数: {np.median(similarity_top1_bleu):.4f}")
    print(f"  范围: [{np.min(similarity_top1_bleu):.4f}, {np.max(similarity_top1_bleu):.4f}]")
    
    print(f"\n重用Token Top1:")
    print(f"  样本数量: {len(reused_token_top1_bleu)}")
    print(f"  平均BLEU: {np.mean(reused_token_top1_bleu):.4f} ± {np.std(reused_token_top1_bleu):.4f}")
    print(f"  中位数: {np.median(reused_token_top1_bleu):.4f}")
    print(f"  范围: [{np.min(reused_token_top1_bleu):.4f}, {np.max(reused_token_top1_bleu):.4f}]")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["VLLM_USE_MODELSCOPE"]="true"
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1.json"
    
    # model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    template = llama3_template_text
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    # output_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json"
    # generate_output_data(input_path,output_path,batch_size=2,model_name=model_name)
    # plot_bleu_comparison(output_path)
    
    
    plot_bleu_comparison("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json","examples/pipeline/images/opus_bleu_comparison_llama3.1-8b.png")
    plot_bleu_comparison("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1_output_qwen2.5-7b.json","examples/pipeline/images/opus_bleu_comparison_qwen2.5-7b.png")
    # outputs = [
    #     ("Llama3.1-8B","examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json"),
    #     ("Qwen2.5-32B","examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_qwen2.5-7b.json"),
    # ]
    # for model_name,output_path in outputs:
    #     generate_output_data(input_path,output_path,batch_size=2,model_name=model_name)
    #     plot_bleu_comparison(output_path)