import os
os.environ["OPENAI_API_KEY"] = "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"

import torch
import transformers
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from hf_embedding import HuggingFaceEmbeddings

# model_id = "gpt2"
model_id = "meta-llama/Llama-2-7b-chat-hf"
pipeline_kwargs={"temperature":0.0, "max_new_tokens": 200, "repetition_penalty": 1.1}

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# begin initializing HF items, need auth token for these
model_config = transformers.AutoConfig.from_pretrained(
    model_id
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)
model.eval()
print(f"Model loaded")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id
)

pipeline = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    **pipeline_kwargs
)

llm = HuggingFacePipeline(pipeline=pipeline)
llm.pipeline.tokenizer.return_token_type_ids = False
llm.pipeline.tokenizer.pad_token = llm.pipeline.tokenizer.eos_token

hfe = HuggingFaceEmbeddings(
    model=model,
    tokenizer=tokenizer
)
print("Embeddings loaded")

qaprompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\n"
    "Question: {question}\n"
    "Answer: ")