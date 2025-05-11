from vllm.engine.llm_engine import LLMEngine



if __name__ == "__main__":
    # 初始化一个引擎，先预先计算所有请求的kvcache
    # 建立索引表
    # 新请求来了，根据索引查询kvcache，如果kvcache命中，则直接计算
    pass