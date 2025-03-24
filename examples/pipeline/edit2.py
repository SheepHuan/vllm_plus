from collections import defaultdict
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
import json

def find_text_differences(tokenizer, source_tokens, target_tokens, window_size=2):
    """
    比较两段文本的差异，使用Rabin-Karp算法进行高效匹配，输出source变到target需要进行的操作
    Args:
        tokenizer: 分词器
        source_tokens: 源文本的token序列
        target_tokens: 目标文本的token序列
        window_size: 滑动窗口大小
    """
    # 添加参数验证
    if not source_tokens or not target_tokens:
        raise ValueError("源文本和目标文本不能为空")
    
    if window_size < 1:
        raise ValueError("窗口大小必须大于0")
    
    if window_size > min(len(source_tokens), len(target_tokens)):
        window_size = min(len(source_tokens), len(target_tokens))
        print(f"警告：窗口大小已调整为 {window_size}")
    
    # 输入验证
    if len(source_tokens) < window_size or len(target_tokens) < window_size:
        return {
            "common_segments": [],
            "source_unique": list(range(len(source_tokens))),
            "target_unique": list(range(len(target_tokens))),
            "difference_summary": "文本太短，无法进行有效比较"
        }
    
    def rolling_hash(tokens, start, window_size, prev_hash=None):
        if prev_hash is None:
            hash_val = 0
            for i in range(window_size):
                hash_val = (hash_val * base + tokens[start + i]) % modulus
            return hash_val
        
        old_val = tokens[start - 1]
        new_val = tokens[start + window_size - 1]
        hash_val = ((prev_hash - old_val * base_power) * base + new_val) % modulus
        return hash_val

    # 哈希参数
    base = 256
    modulus = 1_000_000_007
    base_power = pow(base, window_size - 1, modulus)
    source_len, target_len = len(source_tokens), len(target_tokens)
    
    # 跟踪匹配状态
    source_matched = [False] * source_len
    target_matched = [False] * target_len
    common_segments = []
    
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
                    common_segments.append({
                        "source_span": (i, i+window_size-1),
                        "target_span": (j, j+window_size-1),
                        "text": source_tokens[i:i+window_size]
                    })
                    source_matched[i:i+window_size] = [True] * window_size
                    target_matched[j:j+window_size] = [True] * window_size
    
    
    # 重新组织差异报告的输出
    def merge_segments(segments):
        """合并在source和target中都有重叠的子串段"""
        if not segments:
            return []
        
        # 按source位置排序
        sorted_segs = sorted(segments, key=lambda x: (x["source_span"][0], x["target_span"][0], x["source_span"][1]))
        merged = [sorted_segs[0]]
 
        for next_seg in sorted_segs[1:]:
            last_seg = merged[-1]
            # 同时检查source和target中是否都有重叠
            source_has_overlap = last_seg["source_span"][1] >= next_seg["source_span"][0]
            target_has_overlap = last_seg["target_span"][1] >= next_seg["target_span"][0]
            
            if source_has_overlap and target_has_overlap:
                # 检查合并后的文本是否在source和target中都匹配
                source_start = last_seg["source_span"][0]
                source_end = max(last_seg["source_span"][1], next_seg["source_span"][1])
                target_start = last_seg["target_span"][0]
                target_end = max(last_seg["target_span"][1], next_seg["target_span"][1])
                
                # 验证合并后的文本在source和target中是否一致
                source_text = source_tokens[source_start:source_end + 1]
                target_text = target_tokens[target_start:target_end + 1]
                
                if source_text == target_text:
                    merged[-1] = {
                        "source_span": (source_start, source_end),
                        "target_span": (target_start, target_end),
                        "text": source_text
                    }
                else:
                    merged.append(next_seg)
            else:
                merged.append(next_seg)
        
        return merged

    def analyze_text_changes(tokenizer, source_tokens, target_tokens,common_segments):
        """分析文本变化，找出相同子串需要的移动操作"""
        # FIXME: 合并片段会出现出问题
        merged_segments = merge_segments(common_segments)
        # merged_segments = common_segments
        moves = []
        
        # 计算target中复用的token数量
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
            text = tokenizer.decode(source_tokens[source_start:source_end + 1])
            
            moves.append({
                "text": text,
                "from_position": (source_start, source_end),
                "to_position": (target_start, target_end)
            })
        
        # 过滤掉目标位置是其他移动子集的移动
        # filtered_moves = []
        # for i, move1 in enumerate(moves):
        #     is_subset = False
        #     t1_start, t1_end = move1["to_position"]
            
        #     for j, move2 in enumerate(moves):
        #         if i == j:
        #             continue
        #         t2_start, t2_end = move2["to_position"]
                
        #         # 检查move1的目标位置是否是move2的子集
        #         if (t1_start >= t2_start and t1_end <= t2_end and 
        #             (t1_end - t1_start) < (t2_end - t2_start)):  # 确保是真子集
        #             is_subset = True
        #             break
            
        #     if not is_subset:
        #         filtered_moves.append(move1)
        
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
    for segment in common_segments:
        if segment["target_span"] in unique_target_span:
            continue
        unique_target_span.add(segment["target_span"])
        unique_segments.append(segment)
    

    changes = analyze_text_changes(tokenizer, source_tokens, target_tokens, unique_segments)
    
    # 直接返回changes，不需要额外包装
    return changes

def apply_text_changes(source_tokens: list, target_tokens: list, diff_report: dict, tokenizer,show_detail=False) -> list:
    """
    根据差异报告对源tokens进行修改
    Args:
        source_tokens: 源文本的token序列
        target_tokens: 目标文本的token序列
        diff_report: 差异报告
        tokenizer: 分词器
    Returns:
        modified_tokens: 修改后的token序列
    """
    # 1. 创建target长度的占位列表，初始值设为None
    result = [None] * len(target_tokens)
    
    # 记录token的来源信息
    token_sources = {
        'reused': [],    # (位置, token文本)
        'new': []        # (位置, token文本)
    }
    
    # 2. 根据moves信息，将source中的token移动到对应位置
    for move in diff_report['moves']:
        source_start, source_end = move['from_position']
        target_start, target_end = move['to_position']
        
        # 获取source中的tokens
        source_segment = source_tokens[source_start:source_end + 1]
        
        # 验证目标位置的有效性
        if target_end >= len(target_tokens) and show_detail:
            print(f"警告：移动操作的目标位置 {target_start}:{target_end} 超出目标文本范围 {len(target_tokens)}")
            continue
            
        # 验证source和target段长度匹配
        if len(source_segment) != (target_end - target_start + 1) and show_detail:
            print(f"警告：源段长度 {len(source_segment)} 与目标段长度 {target_end - target_start + 1} 不匹配")
            continue
        
        # 将tokens放入目标位置
        try:
            for i, token in enumerate(source_segment):
                pos = target_start + i
                if 0 <= pos < len(result):  # 确保位置在有效范围内
                    result[pos] = token
                    token_sources['reused'].append((pos, tokenizer.decode([token])))
                else:
                    if show_detail:
                        print(f"警告：跳过无效位置 {pos}")
        except Exception as e:
            if show_detail:
                print(f"处理移动操作时出错: {str(e)}")
            continue
    
    # 3. 检查还有哪些位置是None，这些位置需要从target_tokens中填充
    for i in range(len(result)):
        if result[i] is None:
            result[i] = target_tokens[i]
            token_sources['new'].append((i, tokenizer.decode([target_tokens[i]])))
    
    # 4. 验证结果
    if len(result) != len(target_tokens) and show_detail:
        print("警告：生成的token序列长度与目标不符")
        print(f"目标长度: {len(target_tokens)}, 生成长度: {len(result)}")
    if show_detail:
        # 5. 输出详细的token填充信息
        print("\n=== Token填充详情 ===")
        
        # 按位置排序
        token_sources['reused'].sort(key=lambda x: x[0])
        token_sources['new'].sort(key=lambda x: x[0])
        
        print("\n1. 复用的token位置:")
        for pos, text in token_sources['reused']:
            print(f"  位置 {pos}: '{text}'")
        
        print("\n2. 新填充的token位置:")
        for pos, text in token_sources['new']:
            print(f"  位置 {pos}: '{text}'")
        
        # 6. 输出完整的token序列对比
        print("\n=== Token序列对比 ===")
        print("位置\t目标Token\t生成Token\t来源")
        for i in range(len(result)):
            target_text = tokenizer.decode([target_tokens[i]])
            result_text = tokenizer.decode([result[i]])
            source = "复用" if i in [x[0] for x in token_sources['reused']] else "新增"
            print(f"{i}\t{target_text}\t{result_text}\t{source}")
        
        # 7. 统计信息
        print("\n=== 统计信息 ===")
        print(f"总token数: {len(result)}")
        print(f"复用token数: {len(token_sources['reused'])}")
        print(f"新增token数: {len(token_sources['new'])}")
        print(f"复用比例: {len(token_sources['reused'])/len(result)*100:.2f}%")
    
    return result

def test_performance(data_path:str):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    data = json.load(open(data_path))
    
    for item in data:
        source_text = item["source_text"]
        target_text = item["target_text"]
        modified_text = item["modified_text"]
        
        source_tokens = tokenizer.encode(source_text)
        target_tokens = tokenizer.encode(target_text)
        
        diff_report = find_text_differences(tokenizer, source_tokens, target_tokens,window_size=3)
        modified_tokens = apply_text_changes(source_tokens, target_tokens, diff_report, tokenizer)
        cur_modified_text = tokenizer.decode(modified_tokens)
        if modified_tokens != target_tokens:
            print("source_text:",target_text)
            print("target_text:",modified_text)
            # 打印diff_report
            for move in diff_report['moves']:
                print(f"移动操作: {move['text']} -> {move['from_position']} -> {move['to_position']}")
            pass
if __name__ == "__main__":
    test_performance("examples/dataset/data/similar/instruction_wildv2/error_edit.json")