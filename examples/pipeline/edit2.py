from collections import defaultdict
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
import json
# from intervaltree import IntervalTree
from functools import lru_cache

def find_text_differences(tokenizer, source_tokens, target_tokens, window_size=2):
    """
    使用Rabin-Karp算法比较两段文本的差异，找出可复用的文本片段
    Args:
        tokenizer: 分词器，用于文本解码
        source_tokens: 源文本的token序列
        target_tokens: 目标文本的token序列
        window_size: 滑动窗口大小，用于控制匹配的最小长度
    Returns:
        dict: 包含匹配片段、移动操作和统计信息的差异报告
    """
    # 输入参数验证
    if not source_tokens or not target_tokens:
        raise ValueError("源文本和目标文本不能为空")
    
    def compute_rolling_hash(tokens, start_pos, window_size, previous_hash=None):
        """
        计算或更新滚动哈希值
        Args:
            tokens: token序列
            start_pos: 窗口起始位置
            window_size: 窗口大小
            previous_hash: 前一个位置的哈希值
        """
        if previous_hash is None:
            # 计算初始哈希值
            current_hash = 0
            for i in range(window_size):
                current_hash = (current_hash * HASH_BASE + tokens[start_pos + i]) % HASH_MODULUS
            return current_hash
        
        # 更新滚动哈希值
        removed_token = tokens[start_pos - 1]
        new_token = tokens[start_pos + window_size - 1]
        current_hash = ((previous_hash - removed_token * base_power) * HASH_BASE + new_token) % HASH_MODULUS
        return current_hash

    # 哈希算法参数
    HASH_BASE = 256
    HASH_MODULUS = 1_000_000_007
    base_power = pow(HASH_BASE, window_size - 1, HASH_MODULUS)
    
    # 初始化匹配状态跟踪
    source_length = len(source_tokens)
    target_length = len(target_tokens)
    matched_positions_source = [False] * source_length
    matched_positions_target = [False] * target_length
    matching_segments = []
    
    # 构建源文本的哈希索引表
    source_hash_table = {}
    current_hash = None
    for source_pos in range(source_length - window_size + 1):
        current_hash = compute_rolling_hash(source_tokens, source_pos, window_size, current_hash)
        source_hash_table.setdefault(current_hash, []).append(source_pos)
    
    # 在目标文本中查找匹配
    current_hash = None
    for target_pos in range(target_length - window_size + 1):
        current_hash = compute_rolling_hash(target_tokens, target_pos, window_size, current_hash)
        if current_hash in source_hash_table:
            # 处理哈希碰撞，验证实际内容
            for source_pos in source_hash_table[current_hash]:
                if source_tokens[source_pos:source_pos+window_size] == target_tokens[target_pos:target_pos+window_size]:
                    matching_segments.append({
                        "source_span": (source_pos, source_pos+window_size-1),
                        "target_span": (target_pos, target_pos+window_size-1),
                        "text": source_tokens[source_pos:source_pos+window_size]
                    })
                    # 标记已匹配的位置
                    matched_positions_source[source_pos:source_pos+window_size] = [True] * window_size
                    matched_positions_target[target_pos:target_pos+window_size] = [True] * window_size

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

    def analyze_text_changes(tokenizer, source_tokens, target_tokens, common_segments):
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
            text = tokenizer.decode(source_tokens[source_start:source_end + 1])
            
            moves.append({
                "text": text,
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


def apply_change(source_tokens: list, target_tokens: list, source_kvcache):
    target_kvcache = []
    pass
    



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