import openai
import json
import os
from openai import OpenAI

# 设置OpenAI API密钥
client = OpenAI(
    api_key="sk-PMl5s5V78VDlTQoRhledqZ41fJIWJKTgjprIkYZrg7TxdvWK",  # 替换成你的 DMXapi 令牌key
    base_url="https://www.dmxapi.cn/v1",  # 需要改成DMXAPI的中转 https://www.dmxapi.cn/v1 ，这是已经改好的。
)

# 读取数据集
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 调用OpenAI API获取答案
def get_answer(question):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-latest",
            messages=[
                {"role": "system", "content": "你是一个专业的助手，准确回答问题。"},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用OpenAI API时出错: {e}")
        return None

# 处理数据集并添加答案
def process_dataset(dataset):
    for item in dataset:
        question = item["target_text"]["text"]
        answer = get_answer(question)
        if answer:
            item["target_text"]["answer"] = answer
    return dataset

# 保存处理后的数据集
def save_dataset(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

# 主函数
def main():
    input_file = 'examples/dataset/data/instruction_wildv2_similar_250331_clean.json'  # 输入数据集文件路径
    output_file = 'examples/dataset/data/instruction_wildv2_similar_250331_answer.json'  # 输出数据集文件路径

    # 读取数据集
    dataset = read_dataset(input_file)

    # 处理数据集并添加答案
    processed_dataset = process_dataset(dataset)

    # 保存处理后的数据集
    save_dataset(processed_dataset, output_file)
    print("处理完成，结果已保存到 output_dataset.json")

if __name__ == "__main__":
    main()
    