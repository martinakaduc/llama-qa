import os
import torch
import transformers

model_id = "ura-hcmut/ura-llama-7b"

en_pipeline_kwargs={"temperature": 1.0, 
                    "max_new_tokens": 250, 
                    "top_k": 1, 
                    "repetition_penalty": 1.1}
vi_pipeline_kwargs={"temperature": 1.0, 
                    "max_new_tokens": 250, 
                    "top_k": 1,
                    "repetition_penalty": 1.1}

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
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
    trust_remote_code=True,
    use_auth_token=True
)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded")

vi_pipeline = transformers.pipeline(
    model=vi_model, 
    tokenizer=tokenizer,
    return_full_text=False,  # langchain expects the full text
    task='text-generation',
    **vi_pipeline_kwargs
)