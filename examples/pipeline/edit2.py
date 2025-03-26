from collections import defaultdict
from transformers import AutoTokenizer
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from functools import lru_cache

def preprocess_tokens(source, target, synonym_dict):
    if not synonym_dict:
        return source, target
        
    # 构建反向映射表（近义词token -> 主token）
    reverse_map = {}
    for main_token, synonyms in synonym_dict.items():
        # 主token映射到自身
        reverse_map[main_token] = main_token
        # 所有近义词都映射到主token
        for syn_token in synonyms:
            reverse_map[syn_token] = main_token

    # 替换token序列中的近义词
    processed_source = [reverse_map.get(t, t) for t in source]
    processed_target = [reverse_map.get(t, t) for t in target]
    
    return processed_source, processed_target

def find_text_differences(source_tokens, target_tokens, window_size=2,tokenizer=None,synonym_dict = None):
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
    source_tokens, target_tokens = preprocess_tokens(
        source_tokens, target_tokens, synonym_dict
    )

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


def apply_change(source_tokens: list, target_tokens: list, source_kvcache: torch.Tensor, diff_report:dict):
    """
    根据差异报告对源tokens进行修改
    Args:
        source_tokens: 源文本的token序列
        target_tokens: 目标文本的token序列
        source_kvcache: 源kvcache, shape[layer, 2, token, head*dim]
        diff_report: 差异报告
    """
    # 先根据目标文本生成申请内存
    num_layer,_,_,dim = source_kvcache.shape
    target_kvcache = torch.zeros([num_layer,2,len(target_tokens),dim])
    reused_map_indices = []
    # 根据diff_report的moves信息，将source_kvcache中的token移动到target_kvcache中
    for move in diff_report['moves']:       
        source_start, source_end = move['from_position']
        target_start, target_end = move['to_position']
        target_kvcache[:,:,target_start:target_end+1,:] = source_kvcache[:,:,source_start:source_end+1,:]
        reused_map_indices.extend(list(range(target_start,target_end+1)))
    # 计算得到未复用kvcache的索引
    unreused_map_indices = list(set(list(range(len(target_tokens)))) - set(reused_map_indices))
    # 将未复用kvcache的索引对应的kvcache设置为0
    return target_kvcache,reused_map_indices,unreused_map_indices
    
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
    # test_performance("examples/dataset/data/similar/instruction_wildv2/error_edit.json")
    
    
    # AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # source_text = "Can you write a Synonyms for these words:\n\nbelarusian\nbhopal\nbigamous\nFell\nsought-after\nand after you generate them Translate them with the words i already gave you into Arabic"
    # target_text = "Can you write a synonyms for these words:\n\nboomer\nbursar\nbutty\ncadge\ncarbonization\n\nand after you generate them turn them with the words i already gave you into Arabic"    
    # source_tokens = tokenizer.encode(source_text)
    # target_tokens = tokenizer.encode(target_text)
    
    # source_kvcache = torch.randn([28,2,len(source_tokens),256])
    # diff_report = find_text_differences(source_tokens, target_tokens, window_size=3)
    # target_kvcache,reused_map_indices,unused_map_indices = apply_change(source_tokens, target_tokens, source_kvcache, diff_report)
    # synonyms_text = {"Synonyms": ["Synonyms", "synonyms"]}
    # synonyms_tokens = {}
    # for key,synonyms in synonyms_text.items():
    #     for i in synonyms:
    #         synonyms_tokens[tokenizer.encode(key)[0]] = tokenizer.encode(i)[0]
    
    # modified_tokens = apply_text_changes(source_tokens, target_tokens, diff_report, tokenizer,synonyms_tokens)
    # print(target_kvcache.shape)
    # print(reused_map_indices)
    # print(unused_map_indices)
    # reuse_rate = len(reused_map_indices)/len(target_tokens)
    # # # for segment in diff_report["common_segments"]:
    # # #     print(tokenizer.decode(segment["text"]))
    # # # for move in diff_report["moves"]:
    # # #     print(move["text"],move["from_position"],move["to_position"])
    # print("reuse_ratio:",diff_report["summary"]["reuse_ratio"])
    # # print("reuse_rate:",reuse_rate)
    # print("correct:",modified_tokens == target_tokens)
    
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    
    # a = 'Synonyms'
    # b = 'synonyms'
    # c = ' Synonyms'
    
    # token_a = tokenizer.encode(a)
    # for i in token_a:
    #     print(tokenizer.decode([i]),end=' ')
    # print()
    # token_b = tokenizer.encode(b)
    # for i in token_b:   
    #     print(tokenizer.decode([i]),end=' ')
    # print()
    # token_c = tokenizer.encode(c)
    # for i in token_c:
    #     print(tokenizer.decode([i]),end=' ')
    # print()
    # 构建近义词字典（使用token ID）
    # synonyms_dict = {
    #     tokenizer.encode("Synonyms")[0]: [
    #         tokenizer.encode("Synonyms")[0],
    #         tokenizer.encode("synonyms")[0]
    #     ],
    #     # 可以添加更多近义词组
    # }
    
    # source_text = "Can you write a Synonyms for these words..."
    # target_text = "Can you write a synonyms for these words..."
    source_text= "Classify the following keyword list in groups based on their search intent, whether commercial, transactional or informational:\npadel\npadel tennis\npadel racket\npadel court\npadel london\npadel courts\nstratford padel club\npdel courts london\npadel rackets\npadel tennis near me\npadel courts near me\npadel sport\npadel tennis london\nworld padel tour\npadels\npadel ball\npadel rules\npadel rackets uk\npadel shoes\nadidas padel\nasics padel\npadel shop\npadel player\neverything padel\nbest padel rackets 2021"
    target_text= "Translate the following keywords from English to Spanish and generate the results in a table with two columns, with the keywords in English in the first one and their translation to Spanish in the second:\npadel\npadel tennis\npadel racket\npadel court\npadel london\nstratford padel club\npdel courts london\npadel rackets\npadel tennis near me\npadel courts near me\npadel sport\npadel tennis london\nworld padel tour\npadels\npadel ball\npadel rules\npadel shoes\nadidas padel\nasics padel\npadel shop\npadel player\neverything padel\nbest padel rackets 2021"
    
    source_tokens = tokenizer.encode(source_text)
    target_tokens = tokenizer.encode(target_text)
    
    # 在find_text_differences中传入近义词字典
    diff_report = find_text_differences(
        source_tokens, 
        target_tokens, 
        window_size=3,
        tokenizer=tokenizer,
        synonym_dict=None
    )
    for move in diff_report["moves"]:
        print(move["text"],move["from_position"],move["to_position"])
    print(diff_report["summary"]["reuse_ratio"])