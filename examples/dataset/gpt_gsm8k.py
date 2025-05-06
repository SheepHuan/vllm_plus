import json
from openai import OpenAI
import multiprocessing
from functools import partial

client = OpenAI(
    api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.com/v1 ，这是已经改好的。
)



# print(chat_completion)

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
            "content": f""""
            1. Please modify and replace the names of the main characters in the text I input. For example, change Lisa to LiasABC, change Peter to PeterDFS. You need change all the names in text.
            2. Expand and modify the numbers that appear in the text. For instance, change 1 to 11, 13 to 133, and 4 to 554. 
            3. Also modify the time that appears in the text. For example, change 7:00 to 18:00. 
            You can use your own discretion when modifying the numbers and time. 
            \nText: {text} \n Just give me the modified text.
            """,
        }
    ],
    model="gpt-4.1-mini",    #  替换成你先想用的模型全称， 模型全称可以在DMXAPI 模型价格页面找到并复制。
    )
    return chat_completion.choices[0].message.content

def process_item(item):
    input_text = item["target_doc"]
    output = gsm8k_candidate_generate(input_text)
    item["gpt_candidate_doc"] = output
    return item

if __name__ == "__main__":
    
    data = json.load(open("examples/dataset/data/gsm8k/benchmark_gsm8k.json", "r"))
    # targets = data["targets"]
    
    # 使用多进程处理
    num_processes = min(128, len(data))
    with multiprocessing.Pool(processes=num_processes) as pool:
        gpt_data = pool.map(process_item, data)
    
    json.dump(gpt_data, open("examples/dataset/data/gsm8k/benchmark_gsm8k_gpt.json", "w"), indent=4, ensure_ascii=False)
