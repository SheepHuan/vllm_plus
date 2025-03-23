
import os
import time

def response_text(openai_resp):
    return openai_resp['choices'][0]['message']['content']

"""
export OPENAI_API_KEY="sk-IkYKfZaQiz2JQtgz4b741eEaCb604046B6927489B7B94cF6"
export OPENAI_API_BASE="https://www.DMXapi.com/v1"

export VLLM_USE_MODELSCOPE=True

export OPENAI_API_KEY="sk-3Y6tMZsynAzUTKCwo8enJ4EkRMLdxjt2Nq0WZDxZW9Ic7Ftn"
export OPENAI_API_BASE="http://0.0.0.0:8000/v1"

huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct

vllm serve Qwen/Qwen2-1.5B-Instruct  --dtype bfloat16 --gpu-memory-utilization 0.8
"""


from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding.huggingface import Huggingface
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
# openai.api_key = "sk-IkYKfZaQiz2JQtgz4b741eEaCb604046B6927489B7B94cF6"
print("Cache loading.....")
# os.environ["OPENAI_API_KEY"] = "sk-IkYKfZaQiz2JQtgz4b741eEaCb604046B6927489B7B94cF6" 
# os.environ["OPENAI_API_BASE"] = "https://www.DMXapi.com/v1"
sm = Huggingface(model='sentence-transformers/all-MiniLM-L6-v2')
data_manager = get_data_manager(CacheBase("sqlite"), VectorBase("faiss", dimension=sm.dimension))
cache.init(
    embedding_func=sm.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    )

questions = [
    "小米15Ultra的摄影能力非常强,你能详细描述一下吗?",
    "小米14Pro的摄影能力非常强,你能详细描述一下吗?",
    "华为Mate70Pro+的摄影能力非常强,你能详细描述一下吗?",
]

for question in questions:
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model='gpt-4o',
        messages=[
            {
                'role': 'user',
                'content': question
            }
        ],
    )
    print(f'Question: {question}')
    print("Time consuming: {:.2f}s".format(time.time() - start_time))
    print(f'Answer: {response_text(response)}\n')