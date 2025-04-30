# Сначала загружаем .env
from dotenv import load_dotenv
load_dotenv('.env')

# Основные модули для логирования и фильтрации вывода
import os
import sys
import warnings
import logging
from transformers import logging as hf_logging

def disable_warnings():
    # Отключаем python warnings
    warnings.filterwarnings("ignore")
    # Подавляем все логи уровня WARNING и ниже
    logging.disable(logging.WARNING)
    # Подавляем логирование HF, Optimum и AutoGPTQ
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("optimum").setLevel(logging.ERROR)
    logging.getLogger("autogptq").setLevel(logging.ERROR)
    # Фильтруем STDOUT/STDERR на уровне сообщений
    orig_write = sys.stdout.write
    def filtered_write(text):
        if isinstance(text, str) and any(substr in text for substr in [
            "Note: Enviro", "Unused kwargs", "WARNING - AutoGPTQ"
        ]):
            return len(text)
        return orig_write(text)
    sys.stdout.write = filtered_write
    sys.stderr.write = filtered_write

# Читаем флаг из окружения и при необходимости вызываем функцию
FILTER_WARNINGS = os.getenv("FILTER_WARNINGS", "False").lower() in ("1", "true", "yes")
if FILTER_WARNINGS:
    disable_warnings()

# Далее обычные импорты
import time
from typing import Any
import pandas as pd
import torch
from quantize_gguf import quantize_gguf, download_and_prepare_model
from quantize_gptq import quantize_gptq
from quantize_awq import quantize_awq
from quantize_bnb import quantize_bnb

def get_prefix() -> str:
    return f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO: "

def get_quantization_methods() -> dict[str, Any]:
    """Возвращает маппинг названий методов квантизации на функции их реализации."""
    path_to_llama_cpp = os.getenv("LLAMA_CPP_PATH", '/home/calibri/experiments/quantization_benchmark/llama.cpp')

    def quantize_gguf_wrapper(model_id: str, quant_config: dict[str, Any], prefix_dir: str):
        print(get_prefix() + f"Using llama.cpp path: {path_to_llama_cpp}")
        return quantize_gguf(
            model_id=model_id,
            quant_type=quant_config.get('quant_type', 'Q8_0'),
            prefix_dir=prefix_dir,
            path_to_llama_cpp=path_to_llama_cpp
        )

    return {
        "gptq": quantize_gptq,
        "awq": quantize_awq,
        "bnb": quantize_bnb,
        "gguf": quantize_gguf_wrapper,
    }

if __name__ == "__main__":
    # model_id = "meta-llama/Meta-Llama-3-8B"
    model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
    # model_id = 'moonshotai/Kimi-VL-A3B-Thinking'
    prefix_dir = f"models/{model_id.split('/')[-1]}"
    path_to_llama_cpp = os.getenv("LLAMA_CPP_PATH")

    configs = {
        "gguf": {"quant_type": ["Q8_0", "Q6_K"]},
        "gptq": [
            {"bits": 4, "q_group_size": 128},
            {"bits": 8, "q_group_size": 64}
        ],
        "awq": {"w_bit": 4, "q_group_size": 64, "zero_point": True, "version": "GEMM"},
        "bnb": [{"load_in_4bit": True}, {"load_in_8bit": True}]
    }
    
    print(get_prefix() + f"Downloading and preparing model: {model_id}")
    download_and_prepare_model(model_id=model_id, path_to_llama_cpp=path_to_llama_cpp, prefix_dir=prefix_dir)
    print(get_prefix() + f"Model prepared in: {prefix_dir + model_id.split('/')[-1]}")

    print(get_prefix() + "Starting quantization for all configured methods...")
    quantization_methods = get_quantization_methods()
    quantized_model_paths = []

    for method, method_configs in configs.items():
        quantization_fn = quantization_methods.get(method)
        if not quantization_fn:
            print(get_prefix() + f"Warning: Quantization function for method '{method}' not found. Skipping.")
            continue

        if method == "gguf" and isinstance(method_configs, dict) and isinstance(method_configs.get("quant_type"), list):
            current_configs = [{"quant_type": qt} for qt in method_configs["quant_type"]]
        elif not isinstance(method_configs, list):
            current_configs = [method_configs]
        else:
            current_configs = method_configs

        for config_idx, config in enumerate(current_configs):
            print(get_prefix() + f"Running {method} quantization (config {config_idx + 1}/{len(current_configs)}): {config}")
            try:
                quantized_path = quantization_fn(
                    model_id=model_id,
                    quant_config=config,
                    prefix_dir=prefix_dir
                )
                if quantized_path:
                    print(get_prefix() + f"Quantization successful. Output: {quantized_path}")
                    quantized_model_paths.append(quantized_path)
                else:
                    print(get_prefix() + f"Quantization function for {method} did not return a path.")
            except Exception as e:
                print(get_prefix() + f"Error during {method} quantization with config {config}: {e}")

    print("\n" + get_prefix() + "Quantization process finished.")
    if quantized_model_paths:
        print(get_prefix() + "Successfully quantized models saved at:")
        for path in quantized_model_paths:
            print(f"  - {path}")
        print("\n" + get_prefix() + "You can now run the evaluation script:")
        paths_str = " ".join([f'\"{p}\"' for p in quantized_model_paths])
        gguf_paths = [p for p in quantized_model_paths if p.lower().endswith('.gguf')]
        use_gguf_flag = " --use_gguf" if len(gguf_paths) == len(quantized_model_paths) else ""
        if gguf_paths and len(gguf_paths) != len(quantized_model_paths):
             use_gguf_flag = ""

        print(f"python evaluate.py {paths_str}{use_gguf_flag} --output_csv {prefix_dir}/evaluation_results.csv")
    else:
        print(get_prefix() + "No models were successfully quantized.")
