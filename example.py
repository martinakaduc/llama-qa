import torch
import transformers
from peft import PeftModel, PeftConfig

peft_model_id = "martinakaduc/llama-2-7b-hf-vi"
en_pipeline_kwargs={"temperature":0.0, "max_new_tokens": 200, "repetition_penalty": 1.1}
vi_pipeline_kwargs={"temperature": 0.0, 
                    "max_new_tokens": 500, 
                    "repetition_penalty": 1.1}

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_config = PeftConfig.from_pretrained(peft_model_id)

en_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.base_model_name_or_path,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto'
)
vi_model = PeftModel.from_pretrained(en_model, peft_model_id)
vi_model.eval()
en_model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.base_model_name_or_path)

print(f"Model loaded")