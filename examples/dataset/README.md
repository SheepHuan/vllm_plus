```bash

export HF_ENDPOINT=https://hf-mirror.com


huggingface-cli download --repo-type dataset --resume-download Samsung/samsum
huggingface-cli download --repo-type dataset --resume-download wmt/wmt19

huggingface-cli download --repo-type dataset --resume-download Helsinki-NLP/opus-100
```