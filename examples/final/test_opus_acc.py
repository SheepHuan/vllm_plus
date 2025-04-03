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
import multiprocessing as mp
from functools import partial

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

OPUS_KVCACHE_DIR="examples/pipeline/kvcache/opus"

def generate_output_data(input_path: str, output_path: str, model_name = "Qwen/Qwen2.5-7B-Instruct", batch_size=4,window_size=3):
    device = "cuda:0"
    pipeline = KVShareNewPipeline(model_name, device)
    
    with open(input_path, "r") as f:
        data = json.load(f)
    save_data = []
    
    all_data = data["all_translations"]
    similar_pairs = data["similar_pairs"]
    similar_pairs = [pair for pair in similar_pairs if pair["reused_top1_w31"]["similarity"] < 0.9]
    if os.path.exists(output_path):
        profile_data = json.load(open(output_path,"r"))["similar_pairs"]
        similar_pairs = [pair for pair in similar_pairs if pair["id"] not in profile_data]
    else:
        profile_data = []
    
    print(f"处理后样本数量: {len(similar_pairs)}")
    similar_pairs = random.sample(similar_pairs, min(len(similar_pairs),3000))
    
    
    
    
    save_data = []

    meteor = evaluate.load('meteor')
    tokenizer = pipeline.model.get_tokenizer()
    
    # 逐个处理数据，不再使用批量处理
    for item in tqdm(similar_pairs, desc="Processing items"):
        try:
            # if item["similarity"] > 0.95:
            #     continue
            # 准备prompt
            question = all_data[str(item["id"])]["zh"]
            answer = all_data[str(item["id"])]["en"]
            
            # 添加目标文本
            target_prompt = template.format(text=question)
            source_prompt = template.format(text=all_data[str(item["reused_top1_w31"]["id"])]["zh"])
            
            # 编码token
            target_token_ids = tokenizer.encode(target_prompt)
            
            source_cache_path = os.path.join(OPUS_KVCACHE_DIR,f"opus_kvcache_id-{item['reused_top1_w31']['id']}.pt")
            # 获取kv cache
            if os.path.exists(source_cache_path):
                source_key_values = torch.load(source_cache_path)
                source_token_ids = [tokenizer.encode(source_prompt)]
            else:
                source_key_values, source_outputs = KVShareNewPipeline.get_kvcache_by_full_compute(
                    pipeline.model,
                    SamplingParams(temperature=0, max_tokens=1),
                    [source_prompt]
                )
                torch.save(source_key_values, source_cache_path)
            
                source_token_ids = source_outputs[0].prompt_token_ids
            
            for window_size in [6,12,24]:
                # 单个样本的kvedit
                target_kvcache, reused_map_indices, unreused_map_indices, sample_selected_token_indices = KVEditor.batch_kvedit(
                    [target_token_ids],
                    [source_token_ids],
                    source_key_values,
                    window_size=window_size
                )
                
                # 单个样本的partial compute
                partial_outputs = KVShareNewPipeline.partial_compute(
                    pipeline.model,
                    SamplingParams(temperature=0, max_tokens=512),
                    [target_prompt],
                    reused_map_indices,
                    unreused_map_indices,
                    sample_selected_token_indices,
                    target_kvcache
                )

                try:
                    partial_output = partial_outputs[0].outputs[0].text
                    item["reused_top1_w31"][f"output_w{window_size}"] = partial_output
                    item["reused_top1_w31"][f"meteor_w{window_size}"] = meteor.compute(predictions=[partial_output], references=[answer])
                except Exception as e:
                    print(f"处理item时出错: {str(e)}")
                    continue
            save_data.append(item)   
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            continue
            
    data["similar_pairs"] = save_data+profile_data
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
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
    plt.figure(figsize=(4, 3))
    
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
    # mean_text = "Means:\n" + "\n".join([f"{label}: {mean:.3f}" for label, mean in means])
    # plt.text(0.5, 0.5, mean_text, transform=plt.gca().transAxes, 
    #          bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('BLEU Score')
    plt.ylabel('Cumulative Probability')
    # plt.title('BLEU Score Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    
    # 将图例放在图表右上角
    plt.legend(loc='lower right')
    
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

def plot_meteor_by_window_size(input_path: str, save_path: str = "examples/pipeline/images/opus_meteor_by_window.png"):
    """绘制不同窗口大小下的METEOR分数的CDF曲线对比"""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 收集不同窗口大小的METEOR分数
    window_sizes = [6, 12, 24]  # 根据实际使用的窗口大小调整
    meteor_scores_by_window = {window: [] for window in window_sizes}
    
    for item in data["similar_pairs"]:
        try:
            for window_size in window_sizes:
                if f"meteor_w{window_size}" in item["reused_top1_w31"]:
                    meteor_scores_by_window[window_size].append(item["reused_top1_w31"][f"meteor_w{window_size}"]["meteor"])
        except Exception as e:
            print(f"处理数据时出错: {str(e)}")
            continue
    
    # 定义统一的颜色方案
    colors = {
        6: 'blue',    # 蓝色
        12: 'green',  # 绿色
        24: 'red'     # 红色
    }
    
    # 创建单个图
    plt.figure(figsize=(4, 3))
    
    # 绘制CDF曲线
    for window_size in window_sizes:
        scores = meteor_scores_by_window[window_size]
        if len(scores) == 0:
            print(f"窗口大小 {window_size} 没有数据")
            continue
            
        # 计算CDF
        sorted_scores = np.sort(scores)
        p = np.arange(1, len(scores) + 1) / len(scores)
        
        # 绘制CDF曲线
        plt.plot(sorted_scores, p, label=f'Window Size {window_size}', color=colors[window_size], alpha=0.7)
        
        # 计算并打印统计信息
        mean_value = np.mean(scores)
        median_value = np.median(scores)
        print(f"窗口大小 {window_size}:")
        print(f"  样本数量: {len(scores)}")
        print(f"  平均METEOR: {mean_value:.4f} ± {np.std(scores):.4f}")
        print(f"  中位数: {median_value:.4f}")
        print(f"  范围: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    plt.xlabel('METEOR Score')
    plt.ylabel('Cumulative Probability')
    plt.title('METEOR Score by Window Size')
    plt.grid(True, alpha=0.3)
    
    # 将图例放在图表右上角
    plt.legend(loc='lower right')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"图表已保存至: {save_path}")

def split_data_by_windows_size(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)
    similar_docs = data["similar_pairs"]
    all_data = data["all_translations"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    save_data = []
    windows_size = [31]
    # random_keys = random.sample(list(similar_docs.keys()),1000)
    for key,doc_item in tqdm(similar_docs.items(),desc="Processing"):
        # if key not in random_keys:
        #     continue
        target_doc = all_data[str(doc_item["id"])]
        target_doc_tokens = tokenizer.encode(target_doc["zh"])
        if len(target_doc_tokens) > 4096:
            continue
        term_items = []
        for reused_item in doc_item["similar_items"]:
            reused_doc = all_data[str(reused_item["id"])]
            if reused_item["similarity"] > 0.9995:
                continue
            reused_doc_tokens = tokenizer.encode(reused_doc["zh"])
            reused_item["reused_token_num"] = {}
            for window_size in windows_size:
                diff_report = KVEditor.find_text_differences(target_doc_tokens,reused_doc_tokens,window_size=window_size)
                if len(diff_report["moves"]) ==0:
                    reused_item["reused_token_num"][window_size] = 0
                else:
                    reused_item["reused_token_num"][window_size] = sum([move["to_position"][1]-move["to_position"][0]+1 for move in diff_report["moves"]])
            if reused_item["reused_token_num"][31] > 0:
                term_items.append(reused_item)
        # doc_item["similar_items"] = term_items
        if len(term_items) == 0:
            continue
        else:
            simi_top1 = sorted(term_items,key=lambda x:x["similarity"],reverse=True)[0]
            # reused_top1_w3 = sorted(term_items,key=lambda x:x["reused_token_num"][3],reverse=True)[0]
            # reused_top1_w7 = sorted(term_items,key=lambda x:x["reused_token_num"][7],reverse=True)[0]
            # reused_top1_w15 = sorted(term_items,key=lambda x:x["reused_token_num"][15],reverse=True)[0]
            reused_top1_w31 = sorted(term_items,key=lambda x:x["reused_token_num"][31],reverse=True)[0]
            doc_item["simi_top1"] = simi_top1["id"]
            doc_item["reused_items"] = term_items
            save_data.append({
                "id": doc_item["id"],
                "simi_top1": simi_top1["id"],
                # "reused_top1_w3": reused_top1_w3,
                # "reused_top1_w7": reused_top1_w7,
                # "reused_top1_w15": reused_top1_w15,
                "reused_top1_w31": reused_top1_w31
            })
    data["similar_pairs"] = save_data   
    print(f"处理后样本数量: {len(data['similar_pairs'])}")
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)

def compute_full_compute_acc(input_path,output_path,model_name,batch_size=32):
    with open(input_path, "r") as f:
        data = json.load(f)
    similar_pairs = data["similar_pairs"]
    all_data = data["all_translations"]
    meteor = evaluate.load('meteor')
    pipeline = KVShareNewPipeline(model_name=model_name)
    save_data = []
    for i in tqdm(range(0, len(similar_pairs), batch_size), desc="Processing batches"):
        try:
            batch_items = similar_pairs[i:i + batch_size]
            # 准备所有prompt
            all_target_prompts = []
            batch_answers = []
            
            for item in batch_items:
                question = all_data[str(item["id"])]["zh"]
                answer = all_data[str(item["id"])]["en"]
                batch_answers.append(answer)
                all_target_prompts.append(template.format(text=question))
                     
            # 批量计算full compute
            full_compute_outputs = KVShareNewPipeline.batch_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=512),
                all_target_prompts
            )

            for idx,item in enumerate(batch_items):
                acc = meteor.compute(predictions=[full_compute_outputs[idx].outputs[0].text], references=[batch_answers[idx]])
                item[idx]["output"] = full_compute_outputs[idx].outputs[0].text
                item[idx]["meteor"] = acc["meteor"]
            
            save_data.append(item)
        except Exception as e:
            print(f"处理批次时出错: {str(e)}")
            continue
    data["similar_pairs"] = save_data
    json.dump(data, open(output_path, "w"), indent=4, ensure_ascii=False)
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    input_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_copy.json"
    
    
    
    
    # split_data_by_windows_size(input_path, "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json")
    # windows_sizes = [3,5,7,9,11,13,15]
    
    
    # # 创建进程池，进程数取窗口大小数量和CPU核心数的较小值
    # num_processes = min(len(windows_sizes), mp.cpu_count())
    # pool = mp.Pool(processes=num_processes)
    
    # # 使用partial固定input_path参数
    # process_func = partial(process_window_size, input_path)
    
    # # 并行处理所有窗口大小
    # pool.map(process_func, windows_sizes)
    
    # # 关闭进程池
    # pool.close()
    # pool.join()
    
    template = qwen_template
    model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
    # template = llama3_template_text
    # model_name = "LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
    
    # window_size= 3
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_3_output_qwen2.5-32b.json"
    # generate_output_data(input_path,output_path,batch_size=1,model_name=model_name,window_size=3)
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_output_qwen2.5-32b.json"
    # generate_output_data(input_path,output_path,model_name=model_name,batch_size=32)
    
    input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json"
    output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_fc_output_qwen2.5-32b.json"
    compute_full_compute_acc(input_path,output_path,model_name,batch_size=24)
    # 绘制不同窗口大小下的METEOR分数的CDF曲线对比
    # plot_meteor_by_window_size(output_path, "examples/pipeline/images/opus_meteor_by_window_qwen2.5-32b.png")
    
    # input_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_7_output_qwen2.5-32b.json"
    # output_path = f"examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows_15_output_qwen2.5-32b.json"
    # generate_output_data(input_path,output_path,batch_size=1,model_name=model_name,window_size=15)
    
    # output_path = "examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json"
    # generate_output_data(input_path,output_path,batch_size=2,model_name=model_name)
    # plot_bleu_comparison(output_path)
    
    
    # plot_bleu_comparison("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json","examples/pipeline/images/opus_bleu_comparison_llama3.1-8b.pdf")
    # plot_bleu_comparison("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_test1_output_qwen2.5-7b.json","examples/pipeline/images/opus_bleu_comparison_qwen32-7b.pdf")
    # outputs = [
    #     ("Llama3.1-8B","examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_llama3.1-8b.json"),
    #     ("Qwen2.5-32B","examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_test1_output_qwen2.5-7b.json"),
    # ]
    # for model_name,output_path in outputs:
    #     generate_output_data(input_path,output_path,batch_size=2,model_name=model_name)
    #     plot_bleu_comparison(output_path)