import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv

load_dotenv('.env')
logger = logging.getLogger(__name__)
logger.info("Logging in HF")
login(token=os.getenv("HF_TOKEN"))

def quantize_gptq(model_id: str, quant_config: dict, prefix_dir: str = './') -> str:
    prefix_dir += '/' if prefix_dir[-1] != '/' else ''
    model_path = prefix_dir + model_id.split('/')[1] if os.path.exists(prefix_dir + model_id.split('/')[1]) else model_id
    quant_path = model_id.split('/')[1] + f"-GPTQ-{quant_config['bits']}bit"

    if os.path.exists(prefix_dir + quant_path):
        logger.info("Skipping GPTQ quantization because it already exists")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        config = GPTQConfig(**quant_config, dataset="c4", tokenizer=tokenizer) # exllama_config={"version":2}

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=False,
            quantization_config=config,
            revision="main"
        )
        
        logger.info("Save GPTQ quantized model")
        os.makedirs(prefix_dir + quant_path, exist_ok=True)
        model.save_pretrained(prefix_dir + quant_path)
        tokenizer.save_pretrained(prefix_dir + quant_path)

        logger.info("Push to hub GPTQ quantized model")
        model.push_to_hub(quant_path)
        tokenizer.push_to_hub(quant_path)

    return prefix_dir + quant_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prefix_dir = f'models/{model_id.split("/")[1]}'
    gptq_config = {"bits": 4}

    quantize_gptq(model_id=model_id, quant_config=gptq_config, prefix_dir=prefix_dir)
