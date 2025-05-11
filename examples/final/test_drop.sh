#!/bin/bash
# conda activate kvshare

python examples/final/test_drop.py \
--gpu 0 \
--model Qwen/Qwen2.5-7B-Instruct \
--batch_size 16 \
--generate_kvcache