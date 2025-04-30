from collections import defaultdict
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from functools import lru_cache
from typing import List


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
            reuse_ratio = reused_count / (len(target_tokens)+1e-8) * 100
            

            # 对每个匹配的子串，记录其移动操作（包括位置相同的）
            for segment in merged_segments:
                source_start, source_end = segment["source_span"]
                target_start, target_end = segment["target_span"]
                
                moves.append({
                    # "text": '' if tokenizer is None else tokenizer.decode(source_tokens[source_start:source_end + 1]),
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
        expand_num = 1
        # 根据移动信息更新KV缓存
        for move in moves:       
            source_start, source_end = move[0] # from_position
            target_start, target_end = move[1] # to_position
            
            source_end = source_end-expand_num
            source_start = source_start + expand_num
            
            target_start = target_start + expand_num
            target_end = target_end - expand_num
            
            if target_start >= target_end or source_start >=source_end:
                continue
            
            # 复制KV缓存值
            target_kvcache[:, :, target_start:target_end+1, :] = source_kvcache[:, :, source_start:source_end+1, :]
            
            
            reused_map_indices.extend(list(range(
                target_start,
                target_end+1
            )))
            
        # 计算未复用的token索引
        unreused_map_indices = list(set(range(target_token_length)) - set(reused_map_indices))
        
        return target_kvcache, reused_map_indices, unreused_map_indices
    
    
    def batch_kvedit(batch_targets_token_ids, batch_sources_token_ids, source_kvcache: torch.Tensor,tokenizer=None,window_size=2):
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
        layer_num = source_kvcache.shape[0]
        a,b,c,d = source_kvcache.shape
        device = source_kvcache.device
        dtype = source_kvcache.dtype
        target_kvcache = [torch.zeros([a, b, len(combined_target_tokens), d], 
                                   dtype=dtype, 
                                   device=device) for _ in range(layer_num)]
        
        # 用于记录当前处理到的位置
        batch_source_prefix_len = 0  # 源序列的累积长度
        batch_target_prefix_len = 0  # 目标序列的累积长度
        
        # 初始化结果列表
        batch_reused_map_indices = []    # 存储所有被复用的token位置
        batch_unreused_map_indices = []  # 存储所有未被复用的token位置
        batch_sample_selected_token_indices = []  # 存储每个请求prefill阶段采样的token索引
        batch_target_slice_list = []
        
        num_pad = 1
        
        for idx,(source_token_ids,target_token_ids) in enumerate(zip(batch_sources_token_ids,batch_targets_token_ids)):
            # 计算当前样本的文本差异
            diff_report = KVEditor.find_text_differences(source_token_ids,target_token_ids,window_size=window_size)
            reused_map_indices = []
            batch_target_slice_list.append((batch_target_prefix_len,batch_target_prefix_len+len(target_token_ids)))
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
                    reused_map_indices.extend(list(range(to_position[0]-batch_target_prefix_len,to_position[1]+1-batch_target_prefix_len)))
                except Exception as e:
                    print(e)
                    print(from_position,to_position)
                    continue
            
            # 去重并计算未被复用的位置
            reused_map_indices = list(set(reused_map_indices))
            unreused_map_indices = list(set(range(0,len(target_token_ids))) - set(reused_map_indices))
            
            # 标点符号，换行符不服用
            if tokenizer is not None:
                ttt =[',','\n','.','\t']
                ttt = [tokenizer.encode(t)[0] for t in ttt]
                for t in ttt:
                    if t in target_token_ids:
                        indices = [i for i, token in enumerate(target_token_ids) if token == t]
                        unreused_map_indices = list(set(unreused_map_indices+indices))
                        reused_map_indices = list(set(reused_map_indices)-set(indices))
                    
            
            # 确保最后一个token总是被重新计算
            if len(target_token_ids)-1  not in unreused_map_indices:
                unreused_map_indices.append(len(target_token_ids)-1)
            
            # 更新累积长度
            batch_source_prefix_len += len(source_token_ids)
            batch_target_prefix_len += len(target_token_ids)
            
            
            # 保存结果
            batch_reused_map_indices.append(reused_map_indices)
            batch_unreused_map_indices.append(unreused_map_indices)
            
            # 计算需要重新计算的token数量的累积和
            
            batch_sample_selected_token_indices.append(len(unreused_map_indices)-1)
    
        # 合并所有样本的结果
        return target_kvcache,batch_reused_map_indices,batch_unreused_map_indices,batch_sample_selected_token_indices,batch_target_slice_list
    
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

    @staticmethod
    def kvedit_v2(target_token_ids, candidates_token_ids_list, candidates_kvcache_list=None,
                  window_size=5,
                  tokenizer=None):
        """
        从多个候选文本中找出可以最大化复用的KV缓存片段
        
        Args:
            target_token_ids: 目标文本的token序列
            candidates_token_ids_list: 候选文本的token序列列表，每个元素是一个候选文本
            candidates_kvcache_list: 候选KV缓存列表，与candidates_token_ids_list对应
            window_size: 滑动窗口大小，用于控制匹配的最小长度
            tokenizer: 分词器，用于文本解码和处理特殊标点符号
        
        Returns:
            tuple: (target_kvcache, reused_map_indices, unreused_map_indices)
                - target_kvcache: 更新后的目标KV缓存
                - reused_map_indices: 被复用的token位置索引
                - unreused_map_indices: 未被复用的token位置索引
        """
        # 如果传入的候选文本是单个文本而非列表，包装成列表形式
        if not isinstance(candidates_token_ids_list[0], list):
            candidates_token_ids_list = [candidates_token_ids_list]
            if candidates_kvcache_list is not None:
                candidates_kvcache_list = [candidates_kvcache_list]
        
        # 初始化最佳匹配跟踪结构
        best_matches = {}  # 目标位置 -> (候选索引, 候选位置, 匹配长度)
        target_len = len(target_token_ids)
        
        # 遍历所有候选文本
        for cand_idx, cand_tokens in enumerate(candidates_token_ids_list):
            # 获取当前候选文本与目标文本的差异报告
            diff_report = KVEditor.find_text_differences(cand_tokens, target_token_ids, window_size=window_size)
            
            # 处理当前候选文本的移动操作
            for move in diff_report["moves"]:
                from_pos = move["from_position"]
                to_pos = move["to_position"]
                match_length = to_pos[1] - to_pos[0] + 1
                
                # 对移动覆盖的每个目标位置，检查是否是更好的匹配
                for i in range(to_pos[0], to_pos[1] + 1):
                    match_start = from_pos[0] + (i - to_pos[0])
                    if i not in best_matches or match_length > best_matches[i][2]:
                        best_matches[i] = (cand_idx, match_start, match_length)
        
        # 初始化结果
        reused_map_indices = []
        if not best_matches:
            # 如果没有匹配，返回空的KV缓存和全部未复用的索引
            unreused_map_indices = list(range(target_len))
            if candidates_kvcache_list is None or len(candidates_kvcache_list) == 0:
                return None, [], unreused_map_indices
            
            # 初始化一个空的KV缓存
            layer_num = candidates_kvcache_list[0].shape[0]
            key_dim = candidates_kvcache_list[0][0].shape[-1]
            device = candidates_kvcache_list[0][0].device
            dtype = candidates_kvcache_list[0][0].dtype
            
            target_kvcache = []
            for _ in range(layer_num):
                key_cache = torch.zeros([target_len, key_dim], device=device, dtype=dtype)
                value_cache = torch.zeros([target_len, key_dim], device=device, dtype=dtype)
                target_kvcache.append(torch.stack([key_cache, value_cache], dim=0))
            
            return torch.stack(target_kvcache, dim=0), [], unreused_map_indices
        
        # 从每个候选KV缓存中提取最佳匹配的片段
        if candidates_kvcache_list is not None:
            layer_num = candidates_kvcache_list[0].shape[0]
            key_dim = candidates_kvcache_list[0][0].shape[-1]
            device = candidates_kvcache_list[0][0].device
            dtype = candidates_kvcache_list[0][0].dtype
            
            target_kvcache = []
            for layer_idx in range(layer_num):
                key_cache = torch.zeros([target_len, key_dim], device=device, dtype=dtype)
                value_cache = torch.zeros([target_len, key_dim], device=device, dtype=dtype)
                
                # 应用每个最佳匹配
                for target_idx, (cand_idx, cand_idx_pos, _) in best_matches.items():
                    key_cache[target_idx] = candidates_kvcache_list[cand_idx][layer_idx, 0, cand_idx_pos]
                    value_cache[target_idx] = candidates_kvcache_list[cand_idx][layer_idx, 1, cand_idx_pos]
                    
                    if layer_idx == 0:  # 只在第一层时记录复用位置，避免重复
                        reused_map_indices.append(target_idx)
                
                target_kvcache.append(torch.stack([key_cache, value_cache], dim=0))
        else:
            target_kvcache = None
            # 仅记录复用位置
            reused_map_indices = list(best_matches.keys())
        
        # 计算未复用的位置
        reused_map_indices = sorted(list(set(reused_map_indices)))
        unreused_map_indices = sorted(list(set(range(target_len)) - set(reused_map_indices)))
        
        # 处理标点符号和确保最后一个token重新计算
        if tokenizer is not None:
            # 定义需要去除复用的标点符号
            punctuation_tokens = ['.', ',', '!', '?', ';', ':', '\n', '。', '，', '！', '？', '；', '：']
            punctuation_tokens = [tokenizer.encode(t)[0] for t in punctuation_tokens]
            
            # 去除标点符号的复用
            for idx in reused_map_indices[:]:
                if idx < len(target_token_ids) and target_token_ids[idx] in punctuation_tokens:
                    reused_map_indices.remove(idx)
                    if idx not in unreused_map_indices:
                        unreused_map_indices.append(idx)
        
        # 确保最后一个token被重新计算
        if target_len - 1 not in unreused_map_indices:
            unreused_map_indices.append(target_len - 1)
            if target_len - 1 in reused_map_indices:
                reused_map_indices.remove(target_len - 1)
        
        # 确保索引有序
        unreused_map_indices = sorted(list(set(unreused_map_indices)))
        reused_map_indices = sorted(list(set(reused_map_indices)))
        
        return (torch.stack(target_kvcache, dim=0) if target_kvcache is not None else None, 
                reused_map_indices, 
                unreused_map_indices)

    def batch_kvedit_v2(batch_targets_token_ids, batch_candidates_token_ids_list, batch_candidates_kvcache_list=None,
                        window_size=2,
                        tokenizer=None,
                        max_move_distance=9999,padded_target_token_length=None):
        """
        批量处理目标文本与多个候选文本的KV缓存复用
        
        Args:
            batch_targets_token_ids: 目标文本token序列的批次
            batch_candidates_token_ids_list: 每个目标文本对应的候选文本token序列列表的批次
            batch_candidates_kvcache_list: 每个目标文本对应的候选KV缓存列表的批次
            window_size: 滑动窗口大小
            tokenizer: 分词器
            max_move_distance: 最大移动距离限制
            padded_target_token_length: 目标token长度的填充值
            
        Returns:
            tuple: (batch_target_kvcache, batch_reused_map_indices, batch_unreused_map_indices)
        """
        batch_target_kvcache = []
        batch_reused_map_indices = []
        batch_unreused_map_indices = []
        
        for batch_idx, target_token_ids in enumerate(batch_targets_token_ids):
            # 获取当前批次项对应的候选文本和KV缓存
            candidates_token_ids_list = batch_candidates_token_ids_list[batch_idx]
            candidates_kvcache_list = None
            if batch_candidates_kvcache_list is not None:
                candidates_kvcache_list = batch_candidates_kvcache_list[batch_idx]
            
            # 应用优化的kvedit_v2函数
            target_kvcache, reused_map_indices, unreused_map_indices = KVEditor.kvedit_v2(
                target_token_ids,
                candidates_token_ids_list,
                candidates_kvcache_list,
                window_size,
                tokenizer
            )
            assert len(target_token_ids)== target_kvcache.shape[2]
            batch_target_kvcache.append(target_kvcache)
            batch_reused_map_indices.append(reused_map_indices)
            batch_unreused_map_indices.append(unreused_map_indices)
            
        return batch_target_kvcache, batch_reused_map_indices, batch_unreused_map_indices


def test_acc():
    import copy
    data = json.load(open("examples/dataset/data/opus/opus_dataset_en-zh_similar_docs_top50_250403_windows.json", "r"))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    all_data = data["all_translations"]
    similar_pairs = data["similar_pairs"]
    from tqdm import tqdm
    for item in tqdm(similar_pairs,desc="Processing"):
        target_text = all_data[str(item["id"])]["zh"]
        source_text = all_data[str(item["reused_top1_w7"]["id"])]["zh"]
        
        target_token_ids = tokenizer.encode(target_text)
        source_token_ids = tokenizer.encode(source_text)
        
        diff_report = KVEditor.find_text_differences(source_token_ids,target_token_ids,window_size=7)
        modified_token_ids = copy.deepcopy(target_token_ids)
        
        for move in diff_report["moves"]:
            from_pos = move["from_position"]
            to_pos = move["to_position"]
            modified_token_ids[to_pos[0]:to_pos[1]+1] = source_token_ids[from_pos[0]:from_pos[1]+1]
        
        if modified_token_ids == target_token_ids:
            continue
        else:
            print(tokenizer.decode(modified_token_ids))
            print(tokenizer.decode(target_token_ids))
            # break
    
    
if __name__=="__main__":
    # KVEditor.test_acc()
    # test kvedit_v2
    a = "\n banana, orange, pear, pineapple, mango, strawberry, grape, apple."
    b = [
          "\n mango, mango, mango, mango, mango, mango, grape, apple.",
          "\n mango, orange, pear, pineapple, time, time, time, time."
    ]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    a_token_ids = tokenizer.encode(a)
    b_token_ids = [tokenizer.encode(b_item) for b_item in b]
    b_kvcache = [torch.randn(1,2,len(b_item),128) for b_item in b_token_ids]
    batch_target_kvcache, batch_reused_map_indices, batch_unreused_map_indices = KVEditor.kvedit_v2(a_token_ids,b_token_ids,b_kvcache)
    # print(res)
    for i in range(len(batch_reused_map_indices)):
        print(tokenizer.decode(a_token_ids[batch_reused_map_indices[i]]))
        
