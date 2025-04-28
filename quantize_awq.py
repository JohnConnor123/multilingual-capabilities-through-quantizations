import os
import logging
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
from utils import push_to_hub

load_dotenv('.env')
logger = logging.getLogger(__name__)
logger.info("Logging in HF")
login(token=os.getenv("HF_TOKEN"))

def quantize_awq(model_id: str, quant_config: dict, prefix_dir: str = './') -> str:
    prefix_dir += '/' if prefix_dir[-1] != '/' else ''
    model_path = prefix_dir + model_id.split('/')[-1] if os.path.exists(prefix_dir + model_id.split('/')[-1]) else model_id
    quant_path = model_id.split('/')[-1] + f'-AWQ-{quant_config["q_group_size"]}G-INT{quant_config["w_bit"]}-v{quant_config["version"]}'

    if os.path.exists(prefix_dir + quant_path):
        logger.info("Skipping AWQ quantization because it already exists")
    else:
        logger.info("Load model")
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, **{"low_cpu_mem_usage": True, "use_cache": False, "device_map": None}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        logger.info("Quantize model")
        model.quantize(tokenizer, quant_config=quant_config)

        logger.info("Save quantized model")
        os.makedirs(prefix_dir + quant_path, exist_ok=True)
        model.save_quantized(prefix_dir + quant_path)
        tokenizer.save_pretrained(prefix_dir + quant_path)

    # Push to Hugging Face Hub with Model Card metadata
    push_to_hub(
        quant_dir=prefix_dir + quant_path,
        base_model=model_id,
        description=f"## AWQ quantization config\n\n>{quant_config}"
    )
    return prefix_dir + quant_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_id = "RefalMachine/RuadaptQwen2.5-14B-Instruct-1M"
    prefix_dir = f'models/{model_id.split("/")[-1]}'
    awq_config = {"zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

    quantize_awq(model_id=model_id, quant_config=awq_config, prefix_dir=prefix_dir)
