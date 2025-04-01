import os
import json
import time
import tempfile
from openai import OpenAI, APIError, APITimeoutError
from tqdm import tqdm

CONFIG = {
    "input_path": "examples/dataset/data/instruction_wildv2_similar_250331_clean.json",
    "output_path": "examples/dataset/data/instruction_wildv2_similar_250331_Answer.json",
    "model_name": "chatgpt-4o-latest",
    "max_tokens": 4096,
    "temperature": 0.3,
    "system_prompt": "You are a helpful assistant. Please answer my questions in detail, and make your answers as comprehensive as possible.",  # 根据问题领域调整
    "request_timeout": 45,
    "max_retries": 3,
    "retry_delay": 5,
    "rate_limit_delay": 1  # 请求间隔防止触发限流
}

client = OpenAI(
    api_key=os.getenv("DMX_API_KEY"),
    base_url="https://www.dmxapi.cn/v1"
)

def validate_config():
    """验证配置文件"""
    required_keys = ["input_path", "output_path", "model_name"]
    for key in required_keys:
        if key not in CONFIG:
            raise ValueError(f"Missing required config: {key}")
    if not os.path.exists(CONFIG["input_path"]):
        raise FileNotFoundError(f"Input file not found: {CONFIG['input_path']}")

def process_question(question_text):
    """处理单个问题"""
    messages = [
        {"role": "system", "content": CONFIG["system_prompt"]},
        {"role": "user", "content": question_text}
    ]
    
    for attempt in range(CONFIG["max_retries"]):
        try:
            response = client.chat.completions.create(
                model=CONFIG["model_name"],
                messages=messages,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
                timeout=CONFIG["request_timeout"]
            )
            return {
                "answer": response.choices[0].message.content.strip(),
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except APITimeoutError:
            time.sleep(CONFIG["retry_delay"] * (attempt + 1))
        except APIError as e:
            return {"error": f"API Error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    return {"error": "Max retries exceeded"}

def main():
    validate_config()
    
    # 加载原始数据
    with open(CONFIG["input_path"], "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
    
    try:
        # 尝试加载已有进度
        processed_ids = set()
        if os.path.exists(CONFIG["output_path"]):
            with open(CONFIG["output_path"], "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                processed_ids = {item["target_text"]["id"] for item in existing_data}
            # 将已有数据写入临时文件
            json.dump(existing_data, temp_file, ensure_ascii=False, indent=2)
        else:
            json.dump([], temp_file, ensure_ascii=False, indent=2)
        
        temp_file.close()
        
        # 处理数据
        with tqdm(total=len(original_data), desc="Processing legal cases") as pbar:
            for item in original_data:
                # 跳过已处理条目
                if item["target_text"]["id"] in processed_ids:
                    pbar.update(1)
                    continue
                
                # 处理目标问题
                result = process_question(item["target_text"]["text"])
                
                # 更新数据结构
                item["target_text"].update(result)
                
                # 追加写入临时文件
                with open(temp_file.name, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data.append(item)
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 更新进度
                processed_ids.add(item["target_text"]["id"])
                pbar.update(1)
                
                # 速率控制
                time.sleep(CONFIG["rate_limit_delay"])
        
        # 原子替换文件
        os.replace(temp_file.name, CONFIG["output_path"])
    
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

if __name__ == "__main__":
    main()