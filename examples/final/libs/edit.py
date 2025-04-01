from collections import defaultdict
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from functools import lru_cache


class KVEditor:
    """KV缓存编辑器类，用于处理和优化文本差异比较及KV缓存重用"""
    
    @staticmethod
    def find_text_differences(source_tokens, target_tokens, window_size=2, tokenizer=None):
        """
        使用Rabin-Karp算法比较两段文本的差异，找出可复用的文本片段
        
        Args:
            tokenizer: 分词器，用于文本解码
            source_tokens: 源文本的token序列
            target_tokens: 目标文本的token序列
            window_size: 滑动窗口大小，用于控制匹配的最小长度
            synonym_dict: 同义词字典（当前未使用）
            
        Returns:
            dict: 包含匹配片段、移动操作和统计信息的差异报告
        """
        # 输入验证
        if not source_tokens or not target_tokens:
            raise ValueError("源文本和目标文本不能为空")

        # 提取哈希计算为独立函数
        def rolling_hash(tokens, start, window_size, prev_hash=None):
            """计算滑动窗口的哈希值"""
            if prev_hash is None:
                hash_val = 0
                for i in range(window_size):
                    hash_val = (hash_val * base + tokens[start + i]) % modulus
                return hash_val
            
            old_val = tokens[start - 1]
            new_val = tokens[start + window_size - 1]
            hash_val = ((prev_hash - old_val * base_power) * base + new_val) % modulus
            return hash_val

        # 哈希算法参数设置
        base = 256
        modulus = 1_000_000_007
        base_power = pow(base, window_size - 1, modulus)
        source_len, target_len = len(source_tokens), len(target_tokens)
        
        # 初始化匹配状态追踪
        source_matched = [False] * source_len
        target_matched = [False] * target_len
        matching_segments = []
        
        # 构建源文本的哈希索引
        source_hash_index = {}
        current_hash = None
        for i in range(source_len - window_size + 1):
            current_hash = rolling_hash(source_tokens, i, window_size, current_hash)
            if current_hash in source_hash_index:
                source_hash_index[current_hash].append(i)
            else:
                source_hash_index[current_hash] = [i]
        
        # 在目标文本中查找匹配
        current_hash = None
        for j in range(target_len - window_size + 1):
            current_hash = rolling_hash(target_tokens, j, window_size, current_hash)
            if current_hash in source_hash_index:
                for i in source_hash_index[current_hash]:
                    if source_tokens[i:i+window_size] == target_tokens[j:j+window_size]:
                        matching_segments.append({
                            "source_span": (i, i+window_size-1),
                            "target_span": (j, j+window_size-1),
                            "text": source_tokens[i:i+window_size]
                        })
                        source_matched[i:i+window_size] = [True] * window_size
                        target_matched[j:j+window_size] = [True] * window_size


        def merge_overlapping_segments(segments):
            """
            合并重叠的文本片段
            Args:
                segments: 待合并的文本片段列表
            Returns:
                list: 合并后的文本片段列表
            """
            if not segments:
                return []
            
            # 按源文本位置和目标文本位置排序
            sorted_segments = sorted(segments, 
                                key=lambda x: (x["source_span"][0], x["target_span"][0], x["source_span"][1]))
            
            merged_segments = [sorted_segments[0]]
    
            for current_segment in sorted_segments[1:]:
                previous_segment = merged_segments[-1]
                # 检查源文本和目标文本中的重叠
                source_overlaps = previous_segment["source_span"][1] >= current_segment["source_span"][0]
                target_overlaps = previous_segment["target_span"][1] >= current_segment["target_span"][0]
                
                if source_overlaps and target_overlaps:
                    # 计算合并后的范围
                    merged_source_start = previous_segment["source_span"][0]
                    merged_source_end = max(previous_segment["source_span"][1], 
                                        current_segment["source_span"][1])
                    merged_target_start = previous_segment["target_span"][0]
                    merged_target_end = max(previous_segment["target_span"][1], 
                                        current_segment["target_span"][1])
                    
                    # 验证合并的有效性
                    merged_source_text = source_tokens[merged_source_start:merged_source_end + 1]
                    merged_target_text = target_tokens[merged_target_start:merged_target_end + 1]
                    
                    if merged_source_text == merged_target_text:
                        merged_segments[-1] = {
                            "source_span": (merged_source_start, merged_source_end),
                            "target_span": (merged_target_start, merged_target_end),
                            "text": merged_source_text
                        }
                    else:
                        merged_segments.append(current_segment)
                else:
                    merged_segments.append(current_segment)
            
            return merged_segments


        def analyze_text_changes(source_tokens, target_tokens, common_segments,tokenizer=None):
            """分析文本变化，找出相同子串需要的移动操作"""
            # FIXME: 合并片段会出现出问题
            merged_segments = merge_overlapping_segments(common_segments)
            # merged_segments = common_segments
            moves = []
            
            # 使用集合来优化查找操作
            reused_tokens = set()
            for segment in merged_segments:
                target_start, target_end = segment["target_span"]
                for i in range(target_start, target_end + 1):
                    reused_tokens.add(i)
            
            reused_count = len(reused_tokens)
            reuse_ratio = reused_count / len(target_tokens) * 100
            

            # 对每个匹配的子串，记录其移动操作（包括位置相同的）
            for segment in merged_segments:
                source_start, source_end = segment["source_span"]
                target_start, target_end = segment["target_span"]
                
                moves.append({
                    "text": '' if tokenizer is None else tokenizer.decode(source_tokens[source_start:source_end + 1]),
                    "from_position": (source_start, source_end),
                    "to_position": (target_start, target_end)
                })
            
            
            # 更新报告格式
            report = {
                    "common_segments": merged_segments,  # 所有匹配的子串（已合并）
                    "moves": moves,  # 过滤后的移动操作
                    "summary": {
                        "source_length": len(source_tokens),
                        "target_length": len(target_tokens),
                        "common_segments_count": len(merged_segments),
                        "moves_count": len(moves),
                        "reused_tokens_count": reused_count,
                        "reuse_ratio": reuse_ratio
                    }
                }
                
            return report

        # common_segments 去重,可能有多个相同片段会移动到相同位置
        unique_segments = []
        unique_target_span = set()
        for segment in matching_segments:
            if segment["target_span"] in unique_target_span:
                continue
            unique_target_span.add(segment["target_span"])
            unique_segments.append(segment)
        

        changes = analyze_text_changes(source_tokens, target_tokens, unique_segments,tokenizer)
        
        # 直接返回changes，不需要额外包装
        return changes

    @staticmethod
    def apply_change(target_token_length, source_kvcache: torch.Tensor, moves):
        """
        根据差异报告对源KV缓存进行重用和更新
        
        Args:
            source_tokens: 源文本的token序列
            target_tokens: 目标文本的token序列
            source_kvcache: 源KV缓存，shape为[layer, 2, token, head*dim]
            diff_report: 差异报告，包含token移动信息
            
        Returns:
            tuple: (target_kvcache, reused_map_indices, unreused_map_indices)
                - target_kvcache: 更新后的目标KV缓存
                - reused_map_indices: 复用的token索引列表
                - unreused_map_indices: 未复用的token索引列表
        """
        # 初始化目标KV缓存
        a,b,c,d = source_kvcache.shape
        device = source_kvcache.device
        dtype = source_kvcache.dtype
        target_kvcache = torch.zeros([a, b, target_token_length, d], 
                                   dtype=dtype, 
                                   device=device)
        
        reused_map_indices = []
        
        # 根据移动信息更新KV缓存
        for move in moves:       
            source_start, source_end = move[0] # from_position
            target_start, target_end = move[1] # to_position
            # 复制KV缓存值
            target_kvcache[:, :, target_start:target_end+1, :] = source_kvcache[:, :, source_start:source_end+1, :]
            reused_map_indices.extend(list(range(
                target_start,
                target_end+1
            )))
            
        # 计算未复用的token索引
        unreused_map_indices = list(set(range(target_token_length)) - set(reused_map_indices))
        
        return target_kvcache, reused_map_indices, unreused_map_indices
    
    
    def batch_kvedit(batch_targets_token_ids, batch_sources_token_ids, source_kvcache: torch.Tensor):
        """
        批量应用移动操作到KV缓存
        
        Args:
            batch_targets_token_ids: 目标文本的token序列列表，每个元素是一个样本的token序列
            batch_sources_token_ids: 源文本的token序列列表，每个元素是一个样本的token序列
            source_kvcache: 源KV缓存，shape为[layer, 2, token, head*dim]
        
        Returns:
            tuple: (target_kvcache, batch_reused_map_indices, batch_unreused_map_indices, sample_selected_token_indices)
                - target_kvcache: 更新后的目标KV缓存
                - batch_reused_map_indices: 所有样本中被复用的token位置索引
                - batch_unreused_map_indices: 所有样本中未被复用的token位置索引
                - sample_selected_token_indices: 每个样本中需要重新计算的token数量的累积和
        """
        # 将所有样本的token序列合并成一个大序列
        combined_source_tokens = sum(batch_sources_token_ids, [])
        combined_target_tokens = sum(batch_targets_token_ids, [])
        
        # 获取source_kvcache的维度信息并初始化目标kv缓存
        a,b,c,d = source_kvcache.shape
        device = source_kvcache.device
        dtype = source_kvcache.dtype
        target_kvcache = torch.zeros([a, b, len(combined_target_tokens), d], 
                                   dtype=dtype, 
                                   device=device)
        
        # 用于记录当前处理到的位置
        batch_source_prefix_len = 0  # 源序列的累积长度
        batch_target_prefix_len = 0  # 目标序列的累积长度
        
        # 初始化结果列表
        batch_reused_map_indices = []    # 存储所有被复用的token位置
        batch_unreused_map_indices = []  # 存储所有未被复用的token位置
        sample_selected_token_indices = []  # 存储每个样本需要重新计算的token数量
        
        # 逐个处理每个样本
        for idx,(source_token_ids,target_token_ids) in enumerate(zip(batch_sources_token_ids,batch_targets_token_ids)):
            # 计算当前样本的文本差异
            diff_report = KVEditor.find_text_differences(source_token_ids,target_token_ids)
            reused_map_indices = []
            
            # 处理每个移动操作
            for move in diff_report["moves"]:
                # 调整源位置和目标位置的索引，加上之前样本的长度
                from_position =  list(move["from_position"])
                from_position[0] += batch_source_prefix_len
                from_position[1] += batch_source_prefix_len
                to_position = list(move["to_position"])
                to_position[0] += batch_target_prefix_len
                to_position[1] += batch_target_prefix_len

                if from_position[1]-from_position[0] != to_position[1]-to_position[0]:
                    print(from_position,to_position)
                    continue
                # 复制对应位置的kv缓存
                try:
                    target_kvcache[:, :, to_position[0]:to_position[1]+1, :] = source_kvcache[:, :, from_position[0]:from_position[1]+1, :]
                    reused_map_indices.extend(list(range(to_position[0],to_position[1]+1)))
                except Exception as e:
                    print(e)
                    print(from_position,to_position)
                    continue
            
            # 去重并计算未被复用的位置
            reused_map_indices = list(set(reused_map_indices))
            unreused_map_indices = list(set(range(batch_target_prefix_len,batch_target_prefix_len+len(target_token_ids))) - set(reused_map_indices))
            
            # 确保最后一个token总是被重新计算
            if len(target_token_ids)-1 not in unreused_map_indices:
                unreused_map_indices.append(len(target_token_ids)-1 + batch_target_prefix_len)
            
            # # 调整索引，加上之前样本的累积长度
            # reused_map_indices = [reused_map_indices[i]+batch_target_prefix_len for i in range(len(reused_map_indices))]
            # unreused_map_indices = [unreused_map_indices[i]+batch_target_prefix_len for i in range(len(unreused_map_indices))]
            
            
            # 更新累积长度
            batch_source_prefix_len += len(source_token_ids)
            batch_target_prefix_len += len(target_token_ids)
            
            
            # 保存结果
            batch_reused_map_indices.append(reused_map_indices)
            batch_unreused_map_indices.append(unreused_map_indices)
            
            # 计算需要重新计算的token数量的累积和
            if idx == 0:
                sample_selected_token_indices.append(len(unreused_map_indices)-1)
            else:
                sample_selected_token_indices.append(len(unreused_map_indices)-1 + sample_selected_token_indices[-1] + 1)
        

        # 合并所有样本的结果
        batch_reused_map_indices = sum(batch_reused_map_indices,[])
        batch_unreused_map_indices = sum(batch_unreused_map_indices,[])
        
        return target_kvcache,batch_reused_map_indices,batch_unreused_map_indices,sample_selected_token_indices
    
    def test_acc():
        # Test data setup
        test_cases = [
            {
                "source": "I come from China. My name is Huan.",
                "target": "I come from Japan. My name is Wu."
            },
            {
                "source": "I am from China. My name is Chen.",
                "target": "I am from Japan. My name is Ku."
            },
            {
                "source": "I come from USA. My name is Jake.",
                "target": "I am Japan man. My name is Ku."
            }
        ]

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

        def process_batch_tokens(test_cases, tokenizer):
            # Tokenize all texts
            batch_source_tokens = [tokenizer.encode(case["source"]) for case in test_cases]
            batch_target_tokens = [tokenizer.encode(case["target"]) for case in test_cases]
            
            # Combine tokens
            combined_source_tokens = sum(batch_source_tokens, [])
            combined_target_tokens = sum(batch_target_tokens, [])
            
            # Create source KV cache
            source_kvcache = torch.tensor(combined_source_tokens).view(1, 1, -1, 1)
            
            return {
                "batch_source_tokens": batch_source_tokens,
                "batch_target_tokens": batch_target_tokens,
                "combined_source_tokens": combined_source_tokens,
                "combined_target_tokens": combined_target_tokens,
                "source_kvcache": source_kvcache
            }

        def get_batch_moves(batch_data):
            batch_moves = []
            source_prefix_len = 0
            target_prefix_len = 0
            total_source_tokens = 0
            total_target_tokens = 0
            
            # Process each pair of source and target tokens
            for source_tokens, target_tokens in zip(
                batch_data["batch_source_tokens"], 
                batch_data["batch_target_tokens"]
            ):
                # Get differences report
                diff_report = KVEditor.find_text_differences(source_tokens, target_tokens)
                
                # Update token counts
                total_source_tokens += len(source_tokens)
                total_target_tokens += len(target_tokens)
                
                # Adjust positions based on prefix lengths
                for move in diff_report["moves"]:
                    from_pos = [
                        move["from_position"][0] + source_prefix_len,
                        move["from_position"][1] + source_prefix_len
                    ]
                    to_pos = [
                        move["to_position"][0] + target_prefix_len,
                        move["to_position"][1] + target_prefix_len
                    ]
                    batch_moves.append([from_pos, to_pos])
                
                # Update prefix lengths
                source_prefix_len += len(source_tokens)
                target_prefix_len += len(target_tokens)
                
            return batch_moves, total_target_tokens

        def print_results(batch_data, batch_moves, target_kv_cache, reused_map_indices, tokenizer):
            print("Source text:", tokenizer.decode(batch_data["combined_source_tokens"]))
            print("Target text:", tokenizer.decode(batch_data["combined_target_tokens"]))
            
            print("\nReused segments:")
            for move in batch_moves:
                s, e = move[1]  # target position
                print(tokenizer.decode(batch_data["combined_target_tokens"][s:e+1]))
            
            print("\nReused tokens:")
            reused_tokens = target_kv_cache[0,0,reused_map_indices,0].to(torch.int64).tolist()
            print(' '.join(tokenizer.decode(id) for id in reused_tokens))

        # Main execution
        batch_data = process_batch_tokens(test_cases, tokenizer)
        batch_moves, total_target_tokens = get_batch_moves(batch_data)
        
        # Apply changes to KV cache
        target_kv_cache, reused_map_indices, unreused_map_indices = KVEditor.apply_change(
            total_target_tokens,
            batch_data["source_kvcache"],
            batch_moves
        )
        
        # Print results
        print_results(batch_data, batch_moves, target_kv_cache, reused_map_indices, tokenizer)

if __name__=="__main__":
    KVEditor.test_acc()