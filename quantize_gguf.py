import os
import shutil
import logging
import subprocess
from huggingface_hub import login, snapshot_download, HfApi, create_repo
from dotenv import load_dotenv

load_dotenv('.env')
logger = logging.getLogger(__name__)
logger.info("Logging in HF")
login(token=os.getenv("HF_TOKEN"))


def download_and_prepare_model(model_id: str, path_to_llama_cpp: str, prefix_dir: str = './') -> None:
    prefix_dir += '/' if prefix_dir[-1] != '/' else ''
    model_path = prefix_dir + model_id.split("/")[1]
    path_to_llama_cpp += '/' if path_to_llama_cpp[-1] != '/' else ''

    logger.info("Downloading model")
    snapshot_download(repo_id=model_id, local_dir=model_path, revision="main")
                
    logger.info("Converting to bfloat16 before quantizations")
    if not os.path.exists(model_path + '-bf16'):
        subprocess.run([
            "python",
            path_to_llama_cpp + "llama.cpp/convert_hf_to_gguf.py",
            model_path,
            "--outfile", model_path + '-bf16.gguf',
            "--outtype", "bf16"
        ], check=True)
    

def quantize_gguf(model_id: str, quant_type: str, prefix_dir: str = './', path_to_llama_cpp: str = './') -> str:
    prefix_dir += '/' if prefix_dir[-1] != '/' else ''
    path_to_llama_cpp += '/' if path_to_llama_cpp[-1] != '/' else ''

    model_path = prefix_dir + model_id.split('/')[1] if os.path.exists(prefix_dir + model_id.split('/')[1]) else model_id
    
    quant_dir = prefix_dir + model_id.split('/')[1] + '-' + quant_type
    quant_name = f"{model_id.split('/')[1]}-{quant_type}.gguf"
    quant_path = quant_dir + '/' + quant_name

    if os.path.exists(quant_dir):
        logger.info(f"Skipping {quant_type} quantization because it already exists")
    else:
        quant_script_path = path_to_llama_cpp + "llama.cpp/build/bin/llama-quantize"
        os.makedirs(quant_dir, exist_ok=True)

        # Копируем исходные файлы модели (кроме .safetensors и .bin)
        source_dir = prefix_dir + model_id.split('/')[1]
        for file_name in os.listdir(source_dir):
            if not file_name.endswith(('.safetensors', '.bin')) and not os.path.isdir(os.path.join(source_dir, file_name)):
                src_path = os.path.join(source_dir, file_name)
                dst_path = os.path.join(quant_dir, file_name)
                shutil.copy2(src_path, dst_path)

        logger.info(f"Quantizing to {quant_type} GGUF")
        subprocess.run([
            quant_script_path,
            model_path + '-bf16.gguf',
            quant_path,
            quant_type,
        ], check=True)

    return quant_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prefix_dir = f'models/{model_id.split("/")[1]}'
    gguf_types = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_0"]
    path_to_llama_cpp = '/home/calibri/experiments/quantization_benchmark'

    download_and_prepare_model(model_id, path_to_llama_cpp, prefix_dir=prefix_dir)

    for gguf_type in gguf_types:
        quantize_gguf(model_id=model_id, quant_type=gguf_type, prefix_dir=prefix_dir, path_to_llama_cpp=path_to_llama_cpp)
