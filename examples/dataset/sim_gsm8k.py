import json
import random
import datasets
from transformers import AutoTokenizer
from openai import OpenAI
import multiprocessing
import itertools


def load_gsm8k_data(max_num=128,save_path=None):
    dataset = datasets.load_dataset(
        "openai/gsm8k",
        'main',
        split="test"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    # dataset = json.load(open("examples/dataset/data/samsum/train.json")) + json.load(open("examples/dataset/data/samsum/test.json"))
    data = []
    # dataset = random.sample(dataset,max_num)
    for item in dataset:
        tokens = tokenizer.encode(item["question"])
        # if len(tokens) <= 32 or len(tokens) >= 1024:
        #     continue
        data.append(
            {
                "answer": item["answer"],
                "target_doc": "\t "+item["question"]+" \t",
                "candidate_doc": item["question"]
            }
        )
    data = random.sample(data,max_num)
    print(len(data))
    json.dump(data,open(save_path,"w"),indent=4,ensure_ascii=False)


from transformers import AutoTokenizer
import random

import spacy

def random_split_chunks(text: str, chunk_size: int, num_permutations: int = 10):
    """
    生成所有可能的排列序列的函数

    参数：
    text: 原始文本字符串
    chunk_size: 初始分块的token数（N）
    num_permutations: 需要生成的排列序列数量

    返回：
    包含多个排列序列的列表以及每个排列序列对应的打乱顺序
    """
    nlp = spacy.load('en_core_web_sm')  # 加载英文模型
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    sentences = [" \n\n\n\n\n\n\n\n\n\n\n\n\n\n " + item + " \n\n\n\n\n\n\n\n\n\n\n\n\n\n " for item in sentences]
    return sentences,[]
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    
    
    # # 仅处理句子数量为 4 的情况
    # if len(sentences) != 4:
    #     return [], []
    # # 如果句子数量太多，只取前几个句子进行排列
    # if len(sentences) > 15:
    #     sentences = sentences[:15]

    # # 生成所有可能的排列
    # all_permutations = list(itertools.permutations(sentences))

    # # 如果排列数量超过num_permutations，选择均匀分布的排列
    # if len(all_permutations) > num_permutations:
    #     step = len(all_permutations) // num_permutations
    #     selected_permutations = all_permutations[::step][:num_permutations]
    # else:
    #     selected_permutations = all_permutations

    # shuffled_texts = []
    # shuffle_orders = []
    # for perm in selected_permutations:
    #     # 将排列后的句子重新组合成一句话
    #     prefix = " "+"\n" * 16 + " "
    #     shuffled_text = prefix.join(perm)
    #     shuffled_text =  prefix + shuffled_text + prefix 
    #     shuffled_texts.append(shuffled_text)
    #     # 记录打乱顺序
    #     order = [sentences.index(sent) for sent in perm]
    #     shuffle_orders.append(order)

    # return shuffled_texts, shuffle_orders


def gsm8k_candidate_generate(text):
    # 创建新的客户端实例，避免多进程共享问题
    local_client = OpenAI(
        api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",
        base_url="https://www.dmxapi.cn/v1",
    )
    chat_completion = local_client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"""" \nText: {text}\n Please modify the characters related to the array in the input text. You can change them to any numbers. For example, change 50 to 77, and 18:00 to 23:39. Just give me the modified text.""",
        }
    ],
    model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
    )
    return chat_completion.choices[0].message.content

def process_item(item):
    input_text = item["target_doc"]
    output = gsm8k_candidate_generate(input_text)
    item["candidate_doc"] = output
    return item

def chunk_data(data, chunk_size=96, num_permutations=10):
    new_data = []
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for item in data:
        # 生成多个打乱序列
        text:str = item["candidate_doc"]
        
        setnece,order = random_split_chunks(text,32)
        item["candidates"] = setnece
        new_data.append(item)
        # tokens = tokenizer.encode(text)
        # token_groups = [tokens[i:i + 8] for i in range(0, len(tokens), 8)]
        # for _ in range(num_permutations):
        #     # 复制一份分组列表，避免修改原始列表
        #     shuffled_groups = token_groups.copy()
        #     # 随机打乱分组列表
        #     random.shuffle(shuffled_groups)
        #     # 记录打乱顺序
        #     shuffle_order = [token_groups.index(group) for group in shuffled_groups]
        #     # 将打乱后的分组重新组合成 token 列表
        #     shuffled_tokens = [token for group in shuffled_groups for token in group]
        #     # 将 token 列表解码为文本
        #     shuffled_text = tokenizer.decode(shuffled_tokens)
        #     # 创建新的数据项
        #     new_item = item.copy()
        #     new_item["candidate_doc"] = shuffled_text
        #     new_item["candidates"] = [" \n "+ shuffled_text+" \n "]
        #     new_item["shuffle_order"] = shuffle_order
        #     new_data.append(new_item)
    return new_data


if __name__ == "__main__":  
    save_path = "examples/dataset/data/gsm8k/sim_gsm8k_benchmark_dataset.json"
    gpt_save_path ="examples/dataset/data/gsm8k/benchmark_gsm8k_gpt.json"
    gpt_save_chunk_path ="examples/dataset/data/gsm8k/sim_gsm8k_benchmark_dataset_chunk.json"
    load_gsm8k_data(save_path=save_path,max_num=512)
    
    
    # data = json.load(open(save_path,"r"))
    # num_processes = min(64, len(data))
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     gpt_data = pool.map(process_item, data)
    
    # json.dump(gpt_data, open(gpt_save_path, "w"), indent=4, ensure_ascii=False)

    
    data = json.load(open(save_path,"r",encoding="utf-8"))
    data = chunk_data(data,chunk_size=48)
    print("final len:",len(data))
    json.dump(data,open(gpt_save_chunk_path,"w"),indent=4,ensure_ascii=False)
