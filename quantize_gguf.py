import os
import shutil
import logging
import subprocess
from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv
from utils import push_to_hub

load_dotenv('.env')
logger = logging.getLogger(__name__)
logger.info("Logging in HF")
login(token=os.getenv("HF_TOKEN"))


def download_and_prepare_model(model_id: str, path_to_llama_cpp: str, prefix_dir: str = './') -> None:
    """Скачивает модель с HF, либо использует уже существующую локальную папку.

    Если `model_id` указывает на путь, который уже существует на диске, то
    пропускаем загрузку из HuggingFace Hub и используем его напрямую.
    В противном случае считаем, что это remote‑id и загружаем снапшот в
    директорию `<prefix_dir>/<model_name>`.
    """

    # Приводим входные пути к единообразному виду
    if prefix_dir and prefix_dir[-1] != '/':
        prefix_dir += '/'
    if path_to_llama_cpp and path_to_llama_cpp[-1] != '/':
        path_to_llama_cpp += '/'

    # Определяем, является ли model_id локальным путём
    is_local_model = os.path.isdir(model_id)

    model_name = os.path.basename(model_id.rstrip('/')) if is_local_model else model_id.split('/')[-1]
    model_path = model_id if is_local_model else prefix_dir + model_name

    if not is_local_model:
        logger.info(f"Downloading model '{model_id}' to '{model_path}'")
        snapshot_download(repo_id=model_id, local_dir=model_path, revision="main")
    else:
        logger.info(f"Using local model directory '{model_path}'")
        if os.getenv("UPLOAD_ORIGINAL_MODEL").lower() == 'true':
            logger.info("UPLOAD_ORIGINAL_MODEL is True, uploading original local model to HF hub.")
            push_to_hub(
                quant_dir=model_path,
                base_model=None,
                description="Original local model upload"
            )
        else:
            logger.info("Skipping upload of original local model as UPLOAD_ORIGINAL_MODEL is not set to True.")

    # Конвертация в bf16 (если ещё не конвертировали)
    bf16_path = model_path + '-bf16.gguf'
    if not os.path.exists(bf16_path):
        logger.info("Converting to bfloat16 before quantizations")
        subprocess.run([
            "python",
            path_to_llama_cpp + "llama.cpp/convert_hf_to_gguf.py",
            model_path,
            "--outfile", bf16_path,
            "--outtype", "bf16"
        ], check=True)


def quantize_gguf(model_id: str, quant_type: str, prefix_dir: str = './', path_to_llama_cpp: str = './') -> str:
    prefix_dir += '/' if prefix_dir[-1] != '/' else ''
    path_to_llama_cpp += '/' if path_to_llama_cpp[-1] != '/' else ''

    model_path = prefix_dir + model_id.split('/')[-1] if os.path.exists(prefix_dir + model_id.split('/')[-1]) else model_id
    
    quant_dir = prefix_dir + model_id.split('/')[-1] + '-' + quant_type
    quant_name = f"{model_id.split('/')[-1]}-{quant_type}.gguf"
    quant_path = quant_dir + '/' + quant_name

    if os.path.exists(quant_dir):
        logger.info(f"Skipping {quant_type} quantization because it already exists")
    else:
        quant_script_path = path_to_llama_cpp + "llama.cpp/build/bin/llama-quantize"
        os.makedirs(quant_dir, exist_ok=True)

        # Копируем исходные файлы модели (кроме .safetensors и .bin)
        source_dir = prefix_dir + model_id.split('/')[-1]
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

    # Push to Hugging Face Hub with Model Card
    push_to_hub(quant_dir, base_model=model_id, description=f"GGUF quantization type {quant_type}")
    return quant_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    prefix_dir = f'models/{model_id.split("/")[-1]}'
    gguf_types = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_0"]
    path_to_llama_cpp = '/home/calibri/experiments/quantization_benchmark'

    download_and_prepare_model(model_id, path_to_llama_cpp, prefix_dir=prefix_dir)

    for gguf_type in gguf_types:
        quantize_gguf(model_id=model_id, quant_type=gguf_type, prefix_dir=prefix_dir, path_to_llama_cpp=path_to_llama_cpp)
