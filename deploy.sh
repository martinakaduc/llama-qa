#!/bin/bash
# export SCRIPT_NAME=/llama/
export PATH="/root/miniconda3/envs/llm/bin:/root/miniconda3/bin:/root/miniconda3/condabin:/root/miniconda3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
gunicorn --config config.py app:app