import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from evaluate import load
import seaborn as sns
from scipy.stats import zscore
import math
import re
import evaluate
import matplotlib
from vllm import SamplingParams
import traceback
from libs.pipeline import KVShareNewPipeline
from libs.edit import KVEditor
import torch

# 设置中文字体
font_path = "/root/code/vllm_plus/examples/dataset/data/fonts"
font_files = font_manager.findSystemFonts(fontpaths=font_path)
for file in font_files:
    font_manager.fontManager.addfont(file)

# 设置字体
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.unicode_minus'] = False
qwen_template="""<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n
<|im_start|>user\n\n{text}\n<|im_end|>\n<|im_start|>assistant\n"""

def plot_reuse_fragments(results_list: list, answers: list = None, save_path: str = "examples/pipeline/images/reuse_fragments.png", top_k: int = 10):
    """绘制注意力热力图
    
    Args:
        results_list: 分析结果列表的列表，每个元素是一个results列表
        answers: 答案列表，每个元素为[(B答案, B正确性), (C答案, C正确性)]
        save_path: 保存图片的路径
        top_k: 显示差异最大的前k个token位置
    """
    def process_full_attention(attn, unreused_indices=None):
        # 将注意力矩阵重塑为 [seq_len, num_heads, head_dim]
        seq_len, _ = attn.shape
        attn = attn.reshape(seq_len, 28, 128)
        # 对28个head取平均，得到 [seq_len, head_dim]
        attn = attn.mean(dim=1)
        # 转置矩阵，使head_dim在行，token在列
        attn = attn.t()
        
        # 选择8-32和80-96维度
        selected_dims = list(range(8, 33)) + list(range(80, 97))
        attn = attn[selected_dims, :]
        
        # 如果提供了unreused_indices，则只保留这些位置的attention
        if unreused_indices is not None:
            # 创建一个新的attention矩阵，只包含unreused位置的attention
            unreused_attn = torch.zeros_like(attn)
            # 确保索引在有效范围内
            valid_indices = [idx for idx in unreused_indices if idx < attn.shape[1]]
            for idx in valid_indices:
                unreused_attn[:, idx] = attn[:, idx]
            attn = unreused_attn
        
        return attn.cpu().numpy()
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
    
    # 遍历每组结果
    for group_idx, results in enumerate(results_list):
        print(f"Processing group {group_idx + 1} with {len(results)} samples")
        
        # 遍历每个样本
        for sample_idx in range(len(results)):
            item = results[sample_idx]
            print(f"  Processing sample {sample_idx + 1}")
            
            # 获取B和C的正确性和输出
            b_correct = item.get("correct_b", "unknown")
            c_correct = item.get("correct_c", "unknown")
            b_output = item.get("reuse_b_answer", "No B output")
            c_output = item.get("reuse_c_answer", "No C output")
            
            # 创建图表，每个样本2x2的注意力图
            fig = plt.figure(figsize=(30, 20))
            gs = fig.add_gridspec(2, 2)
            
            # 创建热力图子图
            ax_full_b = fig.add_subplot(gs[0, 0])
            ax_b = fig.add_subplot(gs[0, 1])
            ax_full_c = fig.add_subplot(gs[1, 0])
            ax_c = fig.add_subplot(gs[1, 1])
            
            # 设置子图之间的间距
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            
            # 获取token信息
            tokens_a = item["token_info"]["tokens_a"]
            unreused_indices_b = item["attention_data"]["unreused_indices_b"]
            unreused_indices_c = item["attention_data"]["unreused_indices_c"]
            
            # 获取问题部分的token数量
            num_tokens = len(tokens_a)
            
            # 解码token
            token_texts = [tokenizer.decode(token) for token in tokens_a]
            
            # 处理Full Attention数据
            full_attn = torch.tensor(item["attention_data"]["full_attn"])
            full_attn_processed = process_full_attention(full_attn)
            
            # 处理B Attention数据
            batch_attn_b = torch.tensor(item["attention_data"]["batch_attn_b"])
            batch_attn_b_processed = process_full_attention(batch_attn_b, unreused_indices_b)
            
            # 处理C Attention数据
            batch_attn_c = torch.tensor(item["attention_data"]["batch_attn_c"])
            batch_attn_c_processed = process_full_attention(batch_attn_c, unreused_indices_c)
            
            # 获取所有样本中的最大token数量
            max_tokens = max(len(tokens_a) for results in results_list for item in results)
            
            # 1. 绘制Full Attention热力图（B对比）
            sns.heatmap(full_attn_processed, ax=ax_full_b, cmap='YlOrRd', 
                       xticklabels=False, yticklabels=True,
                       cbar=True, vmin=0, vmax=1, square=False)
            ax_full_b.set_title(f'Group {group_idx+1} Sample {sample_idx+1}: Full Attention (vs B)\nB: {b_correct}', fontsize=14)
            ax_full_b.set_xlabel('Token', fontsize=12)
            ax_full_b.set_ylabel('Dimension', fontsize=12)
            # 设置x轴范围
            ax_full_b.set_xlim(0, max_tokens)
            # 设置y轴刻度，显示选定的维度
            y_ticks = np.arange(0, 41, 8)  # 总共41个维度（25+16）
            y_labels = [f'{i}' for i in range(8, 33, 8)] + [f'{i}' for i in range(80, 97, 8)]
            ax_full_b.set_yticks(y_ticks)
            ax_full_b.set_yticklabels(y_labels[:len(y_ticks)], fontsize=12)
            
            # 2. 绘制B Attention热力图（只显示unreused位置）
            sns.heatmap(batch_attn_b_processed, ax=ax_b, cmap='YlOrRd', 
                       xticklabels=False, yticklabels=True,
                       cbar=True, vmin=0, vmax=1, square=False)
            ax_b.set_title(f'Group {group_idx+1} Sample {sample_idx+1}: B Unreused Attention\nB: {b_correct}', fontsize=14)
            ax_b.set_xlabel('Token', fontsize=12)
            ax_b.set_ylabel('Dimension', fontsize=12)
            ax_b.set_yticks(y_ticks)
            ax_b.set_yticklabels(y_labels[:len(y_ticks)], fontsize=12)
            
            # 3. 绘制Full Attention热力图（C对比）
            sns.heatmap(full_attn_processed, ax=ax_full_c, cmap='YlOrRd', 
                       xticklabels=False, yticklabels=True,
                       cbar=True, vmin=0, vmax=1, square=False)
            ax_full_c.set_title(f'Group {group_idx+1} Sample {sample_idx+1}: Full Attention (vs C)\nC: {c_correct}', fontsize=14)
            ax_full_c.set_xlabel('Token', fontsize=12)
            ax_full_c.set_ylabel('Dimension', fontsize=12)
            ax_full_c.set_yticks(y_ticks)
            ax_full_c.set_yticklabels(y_labels[:len(y_ticks)], fontsize=12)
            
            # 4. 绘制C Attention热力图（只显示unreused位置）
            sns.heatmap(batch_attn_c_processed, ax=ax_c, cmap='YlOrRd', 
                       xticklabels=False, yticklabels=True,
                       cbar=True, vmin=0, vmax=1, square=False)
            ax_c.set_title(f'Group {group_idx+1} Sample {sample_idx+1}: C Unreused Attention\nC: {c_correct}', fontsize=14)
            ax_c.set_xlabel('Token', fontsize=12)
            ax_c.set_ylabel('Dimension', fontsize=12)
            ax_c.set_yticks(y_ticks)
            ax_c.set_yticklabels(y_labels[:len(y_ticks)], fontsize=12)
            
            # 保存当前group的图表
            group_save_path = save_path.replace('.png', f'_group{group_idx+1}_sample{sample_idx+1}.png')
            plt.savefig(group_save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Figure saved to: {group_save_path}")

def analyze_accuracy(input_path: str, output_path: str, max_generate_len: int = 4096, window_size: int = 5, batch_size: int = 4):
    """分析不同方法的精度表现
    
    Args:
        input_path: 输入数据文件路径
        output_path: 输出分析结果文件路径
        max_generate_len: 最大生成长度
        window_size: 窗口大小
        batch_size: 批处理大小
    """
    model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    device = "cuda:0"
    max_model_len = 8192
    pipeline = KVShareNewPipeline(model_name, device, max_model_len)
    tokenizer = pipeline.model.get_tokenizer()
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 初始化结果字典
    results = []
    
    # 按batch_size分批处理数据
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        try:
            # 准备所有prompt
            prompts_a = []
            prompts_b = []
            prompts_c = []
            references = []
            
            for item in batch_data:
                prompts_a.append(qwen_template.format(text=item["question_a"]))
                prompts_b.append(qwen_template.format(text=item["question_b"]))
                prompts_c.append(qwen_template.format(text=item["question_c"]))
                references.append(item["answer"])
            
            # 编码所有token
            tokens_a = [tokenizer.encode(prompt) for prompt in prompts_a]
            tokens_b = [tokenizer.encode(prompt) for prompt in prompts_b]
            tokens_c = [tokenizer.encode(prompt) for prompt in prompts_c]

            # 1. 全量计算A
            full_kv_cache, full_outputs, _, full_attn = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=max_generate_len),
                prompts_a
            )
            
            # 2. 获取B和C的KV缓存
            source_kv_cache_b, source_outputs_b, _, source_attn_b = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=1),
                prompts_b
            )
            
            source_kv_cache_c, source_outputs_c, _, source_attn_c = KVShareNewPipeline.get_kvcache_by_full_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=1),
                prompts_c
            )
            
            source_token_ids_b = [output.prompt_token_ids for output in source_outputs_b]
            source_token_ids_c = [output.prompt_token_ids for output in source_outputs_c]

            from libs.edit import KVEditor
            # 3. 重用B的KV缓存
            target_kv_cache_b, reused_indices_b, unreused_indices_b, selected_tokens_b, target_slices_b = KVEditor.batch_kvedit(
                tokens_a,
                source_token_ids_b,
                source_kv_cache_b,
                window_size=window_size
            )
            
            # 4. 重用C的KV缓存
            target_kv_cache_c, reused_indices_c, unreused_indices_c, selected_tokens_c, target_slices_c = KVEditor.batch_kvedit(
                tokens_a,
                source_token_ids_c,
                source_kv_cache_c,
                window_size=window_size
            )
            
            max_request_id = max([int(output.request_id) for output in source_outputs_c])+1
            # 5. 使用B的KV缓存生成
            outputs_b, partial_kv_cache_b, batch_attn_b = KVShareNewPipeline.partial_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=max_generate_len),
                prompts_a,
                target_kv_cache_b,
                reused_indices_b,
                unreused_indices_b,
                selected_tokens_b,
                target_slices_b,
                [max_request_id+i for i in range(len(prompts_a))]
            )
            
            max_request_id = max([int(output.request_id) for output in outputs_b])+1
            # 6. 使用C的KV缓存生成
            outputs_c, partial_kv_cache_c, batch_attn_c = KVShareNewPipeline.partial_compute(
                pipeline.model,
                SamplingParams(temperature=0, max_tokens=max_generate_len),
                prompts_a,
                target_kv_cache_c,
                reused_indices_c,
                unreused_indices_c,
                selected_tokens_c,
                target_slices_c,
                [max_request_id+i for i in range(len(prompts_a))]
            )
            
            # 处理每个样本的结果
            for idx, item in enumerate(batch_data):
                try:
                    # 保存结果
                    item["full_compute_answer"] = full_outputs[idx].outputs[0].text
                    item["reuse_b_answer"] = outputs_b[idx].outputs[0].text
                    item["reuse_c_answer"] = outputs_c[idx].outputs[0].text
                    
                    # 保存token信息
                    item["token_info"] = {
                        "tokens_a": tokens_a[idx],
                        "reused_indices_b": reused_indices_b[idx],
                        "reused_indices_c": reused_indices_c[idx],
                        "unreused_indices_b": unreused_indices_b[idx],
                        "unreused_indices_c": unreused_indices_c[idx]
                    }
                    
                    # 保存注意力数据
                    item["attention_data"] = {
                        "full_attn": full_attn[idx].cpu().numpy().tolist(),
                        "batch_attn_b": batch_attn_b[idx].cpu().numpy().tolist(),
                        "unreused_indices_b": unreused_indices_b[idx],
                        "reused_indices_b": reused_indices_b[idx],
                        "batch_attn_c": batch_attn_c[idx].cpu().numpy().tolist(),
                        "unreused_indices_c": unreused_indices_c[idx],
                        "reused_indices_c": reused_indices_c[idx],
                    }
                    
                    results.append(item)
                except Exception as e:
                    print(f"处理样本 {idx} 时出错: {str(e)}")
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"处理批次 {i//batch_size} 时出错: {str(e)}")
            traceback.print_exc()
            continue

    # 保存分析结果
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_USE_MODELSCOPE"]="True"
    # 设置输入输出路径
    input_path = "examples/final/test_acc_show copy.json"
    output_path = "examples/final/test_accuracy_analysis.json"
    
    # 运行分析
    results = analyze_accuracy(input_path, output_path,max_generate_len=4096,window_size=5,batch_size=16)
    
    
    # 将单个结果列表包装成一个列表的列表
    results = [json.load(open(output_path))]
    plot_reuse_fragments(results, top_k=20)
