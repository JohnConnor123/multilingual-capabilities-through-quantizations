import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
from utils import push_to_hub

load_dotenv('.env')
logger = logging.getLogger(__name__)
logger.info("Logging in HF")
login(token=os.getenv("HF_TOKEN"))

def quantize_bnb(model_id: str, quant_config: dict, prefix_dir: str = './') -> str:
    prefix_dir += '/' if prefix_dir[-1] != '/' else ''
    model_path = prefix_dir + model_id.split('/')[-1] if os.path.exists(prefix_dir + model_id.split('/')[-1]) else model_id
    quant_path = model_id.split('/')[-1] + \
        f"-BNB-{8 if ('load_in_8bit' in quant_config and quant_config['load_in_8bit'] == True) else 4}bit"

    if os.path.exists(prefix_dir + quant_path):
        logger.info("Skipping Bitsandbytes quantization because it already exists")
    else:
        quantization_config = BitsAndBytesConfig(**quant_config)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype="auto",
        )

        # Сохраняем модель и токенизатор локально
        logger.info("Save Bitsandbytes quantized model locally")
        os.makedirs(prefix_dir + quant_path, exist_ok=True)
        model.save_pretrained(prefix_dir + quant_path)
        tokenizer = AutoTokenizer.from_pretrained(prefix_dir + model_id.split('/')[-1])
        tokenizer.save_pretrained(prefix_dir + quant_path)

    # Push to Hugging Face Hub with Model Card metadata
    push_to_hub(
        quant_dir=prefix_dir + quant_path,
        base_model=model_id,
        description=f"## Bitsandbytes quantization config\n\n>{quant_config}"
    )
    return prefix_dir + quant_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prefix_dir = f'models/{model_id.split("/")[-1]}'
    bnb_config = {
        "load_in_4bit": True,  # 4-bit quantization
        "bnb_4bit_quant_type": 'nf4',  # Normalized float 8
        "bnb_4bit_use_double_quant": True,  # Second quantization after the first
        "bnb_4bit_compute_dtype": 'bfloat16',  # Computation type
    }

    quantize_bnb(model_id=model_id, quant_config=bnb_config, prefix_dir=prefix_dir)
