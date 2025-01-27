import os
import time
from typing import Any
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantize_gguf import quantize_gguf, download_and_prepare_model
from quantize_gptq import quantize_gptq
from quantize_awq import quantize_awq
from quantize_bnb import quantize_bnb
from dotenv import load_dotenv

load_dotenv('.env')


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
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]
    
    print(get_prefix() + "Measuring inference metrics...")
    times = []
    generated_tokens = []
    
    for run in range(num_runs):
        print(get_prefix() + f"Run {run + 1}/{num_runs}")
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        elapsed = time.time() - start_time
        times.append(elapsed)
        generated_tokens.append(outputs.shape[1] - input_tokens)
    
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

def calculate_perplexity(model, tokenizer, test_dataset: str) -> float:
    """Вычисляет перплексию на тестовом наборе"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print(get_prefix() + "Calculating perplexity...")
    with torch.no_grad():
        inputs = tokenizer(test_dataset, return_tensors="pt").to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

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
    
    # Загрузка квантизированной модели
    print(get_prefix() + "Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        quantized_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(quantized_path)
    
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
    quantization_methods = {
        "gptq": quantize_gptq,
        "awq": quantize_awq,
        "bnb": quantize_bnb
    }
    
    for method, method_configs in configs.items():
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


if __name__ == "__main__":
    model_id = "crumb/nano-mistral"
    prefix_dir = "models"
    path_to_llama_cpp = '/home/john/quantizations/quantize/'

    configs = {
        "gptq": [
            {"bits": 4, "q_group_size": 128},
            {"bits": 8, "q_group_size": 64}
        ],
        "awq": {"w_bit": 4, "q_group_size": 128, "version": "GEMM"},
        "bnb": [{"load_in_4bit": True}, {"load_in_8bit": True}]
    }
    # configs = {
    #     "gptq": {"bits": 4, "q_group_size": 128},
    #     "awq": {"w_bit": 4, "q_group_size": 128, "version": "GEMM"},
    #     "bnb": {"load_in_4bit": True}
    # }
    
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
