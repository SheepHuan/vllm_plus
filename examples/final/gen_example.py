import statistics
import numpy as np
import json
import random
def generate_continuous_array(start, end, step=1, data_type=int):
    """生成连续数组，支持整数和浮点数"""
    if data_type == float:
        return np.round(np.arange(start, end+step/2, step), 3).tolist()  # 防止浮点精度问题
    return list(range(start, end+1, step))

def calculate_median(arr):
    """两种计算方式：手动实现 vs 内置库"""
    sorted_arr = sorted(arr)
    n = len(sorted_arr)
    # 方法二：使用内置库（推荐实际使用）
    median = statistics.median(arr)
    return median


def gen_example():
    template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant. <|im_end|>\n<|im_start|>user\n{text}\n<|im_end|>\n<|im_start|>assistant\n"
    prompt= "Please calculate the median of this array: {array}."
    # 计算中位数
    START = random.randint(1, 30)
    LEN = random.randint(50, 60)
    END = random.randint(START+LEN, START+LEN+100)
    array = random.sample(range(START, END+1), LEN)
    array = sorted(array)
    text_a = ', '.join([str(i) for i in array])
    # B是从A中随机删除连续的0.2,0.3,0.4,0.6,0.8比例的元素
    # C是A中删除 随机删除多个连续子数组，子数组之间不连续
    save_data = []
    ratio_b = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    ratio_c = [(0.1,7),(0.2,4),(0.3,2)]  # (删除比例, 子数组数量)
    median = calculate_median(array)
    for r in ratio_b:
        delete_num = int(LEN * r)
        
        delete_start = random.randint(0, LEN-delete_num)
        delete_end = delete_start + delete_num
        array_b = array[:delete_start] + array[delete_end:]
        text_b = ', '.join([str(i) for i in array_b])
        
        for r_c, num_subarrays in ratio_c:
            # 生成C：随机删除多个不连续的连续子数组
            array_c = array.copy()
            total_delete = int(LEN * r_c)
            remaining_delete = total_delete
            
            # 确保子数组之间不重叠
            used_indices = set()
            for _ in range(num_subarrays):
                if remaining_delete <= 0:
                    break
                    
                # 计算当前子数组的最大可能长度
                max_subarray_len = min(remaining_delete, LEN // num_subarrays)
                if max_subarray_len <= 0:
                    break
                    
                # 随机选择子数组长度
                subarray_len = random.randint(1, max_subarray_len)
                
                # 找到可用的起始位置
                available_starts = [i for i in range(LEN - subarray_len + 1) 
                                  if not any(i <= x < i + subarray_len for x in used_indices)]
                if not available_starts:
                    break
                    
                # 随机选择起始位置
                start_idx = random.choice(available_starts)
                
                # 记录已使用的索引
                for i in range(start_idx, start_idx + subarray_len):
                    used_indices.add(i)
                
                # 删除子数组
                array_c[start_idx:start_idx + subarray_len] = []
                remaining_delete -= subarray_len
            
            text_c = ', '.join([str(i) for i in array_c])
            
            save_data.append({
                "question_a": template.format(text=prompt.format(array=text_a)),
                "question_b": template.format(text=prompt.format(array=text_b)),
                "question_c": template.format(text=prompt.format(array=text_c)),
                "answer": median
            })
    return save_data

if __name__ == "__main__":
    # 生成连续数组
    data = []
    for i in range(300):    
        save_data = gen_example()
        data.extend(save_data)
    with open("examples/final/test_example.json", "w") as f:
        json.dump(data, f, indent=4)