#!/bin/sh
export CKPT_DIR="$HOME/projects/LLaMA-data/7B/"
export TOKENIZER_PATH="$HOME/projects/LLaMA-data/tokenizer.model"

export CUDA_VISIBLE_DEVICES=1
python inference_driver.py --ckpt_dir $CKPT_DIR --tokenizer_path $TOKENIZER_PATH
