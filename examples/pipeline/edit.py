from transformers import AutoTokenizer
import copy
import numpy as np
from typing import List

def edit_distance_with_operations(str1, str2, tokenizer):
    """
    使用动态规划计算编辑距离并返回编辑操作序列，优化空间复杂度
    
    Args:
        str1: 源字符串
        str2: 目标字符串
        tokenizer: 分词器
    
    Returns:
        tuple: (编辑距离, 编辑操作列表)
    """
    m, n = len(str1), len(str2)
    
    # 只保存两行的dp值和操作，降低空间复杂度
    current_row = list(range(n + 1))
    previous_row = [0] * (n + 1)
    
    # 保存回溯路径
    operations = {}  # 使用字典存储操作，键为(i,j)
    
    # 动态规划填充
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                delete_cost = previous_row[j] + 1
                insert_cost = current_row[j - 1] + 1
                replace_cost = previous_row[j - 1] + 1
                
                # 优先选择替换操作
                current_row[j] = min(replace_cost, delete_cost, insert_cost)
                
                # 记录操作
                if current_row[j] == replace_cost:
                    operations[(i,j)] = ["Replace", str1[i-1], str2[j-1], i-1]
                elif current_row[j] == delete_cost:
                    operations[(i,j)] = ["Delete", str1[i-1], i-1]
                else:
                    operations[(i,j)] = ["Insert", str2[j-1], i]
    
    # 收集编辑操作
    edit_ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1[i-1] == str2[j-1]:
            i, j = i-1, j-1
            continue
            
        op = operations.get((i,j))
        if op:
            edit_ops.append(op)
            if op[0] == "Replace":
                i, j = i-1, j-1
            elif op[0] == "Delete":
                i -= 1
            else:  # Insert
                j -= 1
        else:
            i, j = i-1, j-1
            
    edit_ops.reverse()
    
    # 校正索引
    correct_operations = []
    offset = 0
    for op in edit_ops:
        if op[0] == "Delete":
            index = op[-1] + offset
            correct_operations.append(["Delete", op[1], index])
            offset -= 1
        elif op[0] == "Insert":
            index = op[-1] + offset
            correct_operations.append(["Insert", op[1], index])
            offset += 1
        elif op[0] == "Replace":
            index = op[-1] + offset
            correct_operations.append(["Replace", op[1], op[2], index])
            
    return current_row[n], correct_operations


def transform_operations(str1,operations):
    result = copy.deepcopy(str1)    
    for op in operations:
        if op[0]=="Delete":
            index = int(op[-1])
            del result[index]
        elif op[0]=="Insert":
            char = op[1]
            index = int(op[-1])
            result.insert(index, char)
        elif op[0]=="Replace":
            new_char = op[2]
            index = int(op[-1])
            result[index] = new_char
    return result

def edit_kvcache(old_prompt,new_prompt,old_kvcache,head_dim=1,num_layer=1):
    _,ops = edit_distance_with_operations(old_prompt,new_prompt,tokenizer)
    new_kvcache: np.ndarray = copy.deepcopy(old_kvcache)
    new_kvcache = new_kvcache.transpose((1, 0, 2))
    additional_indices = []
    delete_indices = []
    for op in ops:
        if op[0]=="Delete":
            index = int(op[-1])
            print(f"\033[91m{tokenizer.decode(new_kvcache[index,0,0])}\033[0m")
            new_kvcache = np.delete(new_kvcache, index, axis=0)
            delete_indices.append(index)
            
        elif op[0]=="Insert":
            char = op[1]
            index = int(op[-1])
            new_kv = np.zeros([1,num_layer,head_dim])
            new_kv[0,0] = char
            new_kvcache = np.insert(new_kvcache, index, new_kv, axis=0)
            additional_indices.append(index)
            print(f"\033[92m{tokenizer.decode(char)}\033[0m")
        elif op[0]=="Replace":
            char = op[2]
            index = int(op[-1])
            new_kv = np.zeros([1,num_layer,head_dim])
            new_kv[0,0] = char
            new_kvcache[index] = new_kv
            additional_indices.append(index)
            print(f"\033[93m{tokenizer.decode(char)}\033[0m")
    new_kvcache = new_kvcache.transpose((1, 0, 2))
    return new_kvcache,additional_indices,delete_indices

if __name__ == "__main__":
    text1 = "Ok, 121 121 i can you again summarize the different interventions that could addresses each of the key components of the COM-B model and gets people to fill in the questionnaire?"
    text2 = "Ok, ok okr how would a comprehensive intervention look like that a addresses each of the key components of the COM-B model and gets a people to fill in the questionnaire?"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokens1 = tokenizer.encode(text1)
    tokens2 = tokenizer.encode(text2)

    kv_token1 = np.zeros([1,len(tokens1),1]).astype(np.int64)
    kv_token1[0,:,0] = np.array(tokens1).astype(np.int64)
    # kv_token2 = np.zeros([len(tokens2),1,1])
    
    kv_token2,additional_indices,delete_indices= edit_kvcache(tokens1,tokens2,kv_token1)
    
    
    for i in range(len(tokens2)):
        # i in additional_indices，那么print绿色的字符，否则print白色的字符
       
        print(tokenizer.decode(tokens2[i]),end=" ")
    print()
    
    
    kv_token2 = kv_token2[0,:,0].tolist()
    for i in range(len(kv_token2)):
        # i in additional_indices，那么print绿色的字符，否则print白色的字符
        if i in additional_indices:
            print(f"\033[92m{tokenizer.decode(kv_token2[i])}\033[0m",end=" ")
        else:
            print(tokenizer.decode(kv_token2[i]),end=" ")
    print()
    # distance, operations = edit_distance_with_operations(tokens1, tokens2,tokenizer)
    # result = transform_operations(tokens1,operations)
    # print(distance)
    # print(operations)
    # print(tokenizer.decode(result))
    # print(tokenizer.decode(tokens2))
