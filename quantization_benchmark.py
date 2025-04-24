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
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize_gguf import quantize_gguf, download_and_prepare_model
from quantize_gptq import quantize_gptq
from quantize_awq import quantize_awq
from quantize_bnb import quantize_bnb

def get_prefix() -> str:
    return f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO: "

def get_model_size(model_path: str) -> float:
    """Возвращает размер модели в гигабайтах"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)

def measure_inference_time(model, tokenizer, test_input: str, num_runs: int = 5) -> dict:
    """Замеряет метрики времени инференса"""
    # Подготовка входа и настройка pad_token_id для корректной генерации
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    # Устанавливаем pad_token_id, если он не задан (избегаем ошибок генерации)
    if model.config.pad_token_id is None and hasattr(tokenizer, 'eos_token_id'):
        model.config.pad_token_id = tokenizer.eos_token_id
    input_tokens = inputs["input_ids"].shape[1]
    
    print(get_prefix() + "Measuring inference metrics...")
    times = []
    generated_tokens = []
    
    for run in range(num_runs):
        print(get_prefix() + f"Run {run + 1}/{num_runs}")
        start_time = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    use_cache=False,
                    pad_token_id=model.config.pad_token_id
                )
        except RuntimeError as err:
            print(get_prefix() + f"Generation error on run {run + 1}: {err}")
            break
        elapsed = time.time() - start_time
        times.append(elapsed)
        generated_tokens.append(outputs.shape[1] - input_tokens)
    
    if not times:
        # Не удалось провести ни одного успешного прогона
        return {
            "inference_time": float('nan'),
            "input_tokens_per_sec": float('nan'),
            "generated_tokens_per_sec": float('nan')
        }
    avg_time = sum(times) / len(times)
    avg_gen_tokens = sum(generated_tokens) / len(generated_tokens)
    
    return {
        "inference_time": avg_time,
        "input_tokens_per_sec": input_tokens / avg_time,
        "generated_tokens_per_sec": avg_gen_tokens / avg_time
    }

def measure_memory_usage(model) -> float:
    """Возвращает использование памяти в гигабайтах"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Выполняем небольшой форвард пасс для активации всей памяти
    dummy_input = torch.randint(0, 100, (1, 128)).to(model.device)
    with torch.no_grad():
        _ = model(dummy_input)
    
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return memory_usage

def calculate_perplexity(model, tokenizer, test_dataset: str, max_length: int = 1024, stride: int = 512) -> float:
    """Вычисляет перплексию на тестовом наборе.

    Для избежания переполнения памяти и получения более корректной оценки
    рассчитываем кросс‑энтропийный лосс по всему набору данных батчами
    фиксированной длины. Затем усредняем лосс по всем токенам и берём
    экспоненту.

    Parameters
    ----------
    model : PreTrainedModel
        Модель, для которой считается перплексия.
    tokenizer : PreTrainedTokenizer
        Токенизатор.
    test_dataset : str
        Текстовая последовательность, на которой измеряется перплексия.
    max_length : int, optional (default=1024)
        Максимальная длина последовательности, подаваемой в модель за один
        проход.
    stride : int, optional (default=512)
        Шаг скользящего окна. Для stride < max_length сохраняется пересечение
        окон, чтобы каждое окно имело достаточно контекста.
    """
    model.eval()

    print(get_prefix() + "Calculating perplexity...")

    encodings = tokenizer(test_dataset, return_tensors="pt")
    input_ids = encodings["input_ids"].squeeze(0)

    total_loss = 0.0
    total_tokens = 0

    # Перебираем датасет скользящим окном
    for i in range(0, len(input_ids), stride):
        begin = i
        end = min(i + max_length, len(input_ids))
        input_slice = input_ids[begin:end].unsqueeze(0).to(model.device)

        # Вычисляем лосс. HuggingFace автоматически делает shift‑labels для
        # моделей CausalLM, поэтому достаточно передать labels=input_slice.
        with torch.no_grad():
            outputs = model(input_ids=input_slice, labels=input_slice)
            loss = outputs.loss  # усреднённый по токенам лосс

        seq_len = input_slice.size(1)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len

        if end == len(input_ids):
            break  # достигли конца датасета

    # Средний лосс по всем токенам и перплексия
    mean_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(mean_loss)).item()
    return perplexity

def load_model_and_tokenizer(model_dir: str):
    """Загружает модель и токенизатор из *model_dir*.

    • Если в директории присутствует хотя бы один файл *.gguf*, используем его
      при вызове `from_pretrained(..., gguf_file=<file>)`.
    • В противном случае работаем со стандартными весами (.bin/.safetensors).

    Таким образом поддерживается как обычный PyTorch‑чекпойнт, так и GGUF без
    каких‑либо специальных условий в остальном коде.
    """
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
    common_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if gguf_files:
        gguf_file = gguf_files[0]
        model = AutoModelForCausalLM.from_pretrained(model_dir, gguf_file=gguf_file, **common_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, gguf_file=gguf_file)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **common_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer

def evaluate_quantization(
    model_id: str,
    quant_config: dict[str, Any],
    prefix_dir: str,
    quantization_fn,
    test_input: str = "Это тестовый текст для оценки производительности модели.",
    test_dataset: str = "Тестовый набор данных для оценки перплексии."
) -> dict[str, float]:
    """Оценивает квантизированную модель по нескольким метрикам"""
    
    print(get_prefix() + f"Starting evaluation for {quantization_fn.__name__}")
    
    # Квантизация модели
    print(get_prefix() + "Quantizing model...")
    quantized_path = quantization_fn(model_id, quant_config, prefix_dir)
    
    # Загрузка квантизированной модели (автоматически поддерживает GGUF)
    print(get_prefix() + "Loading quantized model...")
    model, tokenizer = load_model_and_tokenizer(quantized_path)
    
    # Сбор метрик
    print(get_prefix() + "Collecting metrics...")
    inference_metrics = measure_inference_time(model, tokenizer, test_input)
    memory_usage = measure_memory_usage(model)
    perplexity = calculate_perplexity(model, tokenizer, test_dataset)
    
    results = {
        "method": quantization_fn.__name__.replace("quantize_", ""),
        "model_size_gb": get_model_size(quantized_path),
        "memory_usage_gb": memory_usage,
        "perplexity": perplexity,
        **inference_metrics
    }
    
    print(get_prefix() + f"Results for {results['method']}:")
    for key, value in results.items():
        if key != "method":
            print(f"    {key}: {value:.4f}")
    
    del model
    torch.cuda.empty_cache()
    
    return results

def get_quantization_methods() -> dict[str, Any]:
    """Возвращает маппинг названий методов квантизации на функции их реализации."""
    return {
        "gptq": quantize_gptq,
        "awq": quantize_awq,
        "bnb": quantize_bnb,
        "gguf": quantize_gguf_wrapper,
    }

def compare_quantizations(
    model_id: str,
    test_input: str,
    test_dataset: str,
    configs: dict[str, Any],
    prefix_dir: str,
) -> pd.DataFrame:
    """Сравнивает различные методы квантизации"""
    
    print(get_prefix() + "Starting quantization comparison")
    results = []
    quantization_methods = get_quantization_methods()
    
    for method, method_configs in configs.items():
        # Специальная логика для GGUF: допускаем {'quant_type': [...]}
        if method == "gguf" and isinstance(method_configs, dict) and isinstance(method_configs.get("quant_type"), list):
            # Разворачиваем перечень типов в список конфигов вида {'quant_type': 'QX_Y'}
            method_configs = [{"quant_type": qt} for qt in method_configs["quant_type"]]

        # Обрабатываем как список конфигураций
        if not isinstance(method_configs, list):
            method_configs = [method_configs]
            
        for config_idx, config in enumerate(method_configs):
            print(get_prefix() + f"Evaluating {method} quantization (config {config_idx + 1}/{len(method_configs)})")
            result = evaluate_quantization(
                model_id=model_id,
                quant_config=config,
                prefix_dir=prefix_dir,
                quantization_fn=quantization_methods[method],
                test_input=test_input,
                test_dataset=test_dataset
            )
            # Добавляем параметры конфигурации в результаты
            result["config"] = str(config)
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(prefix_dir + '/comparison_results.csv', sep=';', encoding='utf-8', index=False)
    return results_df

# --- GGUF wrapper ----------------------------------------------------------

def quantize_gguf_wrapper(model_id: str, quant_config: dict[str, Any], prefix_dir: str):
    """Адаптер, позволяющий использовать quantize_gguf в общей схеме.

    Ожидает, что в `quant_config` присутствует ключ ``quant_type`` с одним
    из поддерживаемых режимов квантизации (например, ``"Q8_0"``).
    """
    quant_type = quant_config.get("quant_type")
    if quant_type is None:
        raise ValueError("Для GGUF‑квантизации требуется указать 'quant_type' в конфигурации")
    return quantize_gguf(model_id=model_id, quant_type=quant_type, prefix_dir=prefix_dir)


if __name__ == "__main__":
    # model_id = "crumb/nano-mistral"
    # model_id = "models/Qwen2.5-0.5B-Instruct/Qwen2.5-0.5B-Instruct"
    # model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    model_id = "./models/Meta-Llama-3-8B/Meta-Llama-3-8B"
    # model_id = "RefalMachine/RuadaptQwen2.5-14B-Instruct-1M"
    # model_id = "Qwen/Qwen2.5-14B-Instruct"
    prefix_dir = f"models/{model_id.split('/')[-1]}"
    path_to_llama_cpp = '/home/calibri/experiments/quantization_benchmark'

    configs = {
        "gguf": {"quant_type": ["Q8_0", "Q6_K"]},
        "gptq": [
            {"bits": 4, "q_group_size": 128},
            {"bits": 8, "q_group_size": 64}
        ],
        "awq": {"w_bit": 4, "q_group_size": 64, "zero_point": True, "version": "GEMM"},
        "bnb": [{"load_in_4bit": True}, {"load_in_8bit": True}]
    }
    
    test_input = "Это тестовый текст для оценки производительности модели."
    test_dataset = """
    Это тестовый набор данных для оценки перплексии модели.
    Он должен содержать репрезентативные примеры текста.
    """

    download_and_prepare_model(model_id=model_id, path_to_llama_cpp=path_to_llama_cpp, prefix_dir=prefix_dir)
    
    results_df = compare_quantizations(
        model_id=model_id,
        test_input=test_input,
        test_dataset=test_dataset,
        configs=configs,
        prefix_dir=prefix_dir,
    )
    print("\n" + get_prefix() + "Итоговые результаты сравнения квантизаций:")
    print(results_df)
