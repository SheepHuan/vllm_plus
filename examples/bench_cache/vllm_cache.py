from vllm import LLM, SamplingParams
import vllm

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm = LLM(model=model_name,worker_cls="custom_worker.CustomGPUWorker")
    sampling_params = SamplingParams(temperature=0.5,)



    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:

    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?"
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation,
                    sampling_params=sampling_params,
                    use_tqdm=False)
    print_outputs(outputs)

    # You can run batch inference with llm.chat API
    # conversation = [
    #     {
    #         "role": "system",
    #         "content": "You are a helpful assistant"
    #     },
    #     {
    #         "role": "user",
    #         "content": "Hello"
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "Hello! How can I assist you today?"
    #     },
    #     {
    #         "role": "user",
    #         "content": "Write an essay about the importance of higher education.",
    #     },
    # ]
    # conversations = [conversation for _ in range(1)]

    # # We turn on tqdm progress bar to verify it's indeed running batch inference
    # outputs = llm.chat(messages=conversations,
    #                 sampling_params=sampling_params,
    #                 use_tqdm=True)
    # print_outputs(outputs)
