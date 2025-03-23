# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch

# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
# processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, device_map="cuda")
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "examples/mllm/data/dog.jpg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device)

# # Inference: Generation of the output
# output_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
# output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# print(output_text)

from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset,video_to_ndarrays
from PIL import Image
from vllm.utils import FlexibleArgumentParser

def get_multi_modal_input(path,modality="image"):
    """
    return {
        "data": image or video,
        "question": question,
    }
    """
    if modality == "image":
        # Input image and question
        image = Image.open(path) \
            .convert("RGB")
        img_question = "What is the content of this image?"

        return {
            "data": image,
            "question": img_question,
        }

    if modality == "video":
        # Input video and question
        video = video_to_ndarrays(path,2)
        vid_question = "Why is this video funny?"

        return {
            "data": video,
            "question": vid_question,
        }

    msg = f"Modality {modality} is not supported."
    raise ValueError(msg)

# Qwen2-VL
def run_qwen2_5_vl(question: str, modality: str):

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        # Note - mm_processor_kwargs can also be passed to generate/chat calls
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        # disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids


def main():
    path = "/root/code/vllm_plus/examples/mllm/data/dog.jpg"
    modality = "image"
    mm_input = get_multi_modal_input(path, modality)
    data = mm_input["data"]
    question = mm_input["question"]

    llm, prompt, stop_token_ids = run_qwen2_5_vl(question, modality)

    sampling_params = SamplingParams(temperature=0.2,
                                     max_tokens=64,
                                     stop_token_ids=stop_token_ids)
    inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                modality: data
            },
        }
    
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    
if __name__ == "__main__":
    main()
