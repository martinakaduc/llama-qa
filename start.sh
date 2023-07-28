#!/bin/bash
export TRANSFORMERS_CACHE="/priority/.cache/huggingface"
conda activate geometric
streamlit run app.py --server.baseUrlPath llama-qa --server.enableCORS true &
./ngrok start --all --config ./ngrok.yml
