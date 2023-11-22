import os
import torch
import transformers
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

model_id = "ura-hcmut/ura-llama-7b"

en_pipeline_kwargs={"temperature": 1.0, 
                    "max_new_tokens": 256, 
                    "top_k": 1, 
                    "repetition_penalty": 1.1}
vi_pipeline_kwargs={
    "bos_token_id": 1,
    "do_sample": True,
    "eos_token_id": 2,
    "max_new_tokens": 512,
    "pad_token_id": 0,
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 3,
    "repetition_penalty": 1.1
}

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.float16
)

vi_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=True
)
vi_model.config.use_cache = True
vi_model.config.pretraining_tp = 1
vi_model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    truncation=True, 
    max_length=256,
    trust_remote_code=True,
    use_auth_token=True
)

print(f"Model loaded")
vi_pipeline = transformers.pipeline(
    model=vi_model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    **vi_pipeline_kwargs
)


llm = HuggingFacePipeline(pipeline=vi_pipeline)
llm.pipeline.tokenizer.return_token_type_ids = False
llm.pipeline.tokenizer.pad_token = llm.pipeline.tokenizer.eos_token