import os
import sys
import time
import argparse
import logging
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# (Предполагается, что утилиты вроде get_prefix останутся в utils.py или будут скопированы сюда)
# import utils # или скопировать нужные функции

# Константы или параметры по умолчанию
DEFAULT_TEST_INPUT = "Это тестовый текст для оценки производительности модели."
DEFAULT_TEST_DATASET = """
Это тестовый набор данных для оценки перплексии модели.
Он должен содержать репрезентативные примеры текста.
"""
DEFAULT_RESULTS_CSV = "evaluation_results.csv"

# --- Функции, перенесенные из quantization_benchmark.py ---

def get_prefix() -> str:
    # Простая реализация, можно улучшить или импортировать
    return f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO: "

def get_model_size(model_path: str) -> float:
    """Возвращает размер модели в гигабайтах"""
    total_size = 0
    # Проверяем, является ли путь файлом (GGUF) или директорией
    if os.path.isfile(model_path):
        total_size = os.path.getsize(model_path)
    elif os.path.isdir(model_path):
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # Пропускаем файлы логов или другие не относящиеся к модели файлы
                if not f.endswith(('.log', '.txt', '.md', '.csv')):
                     total_size += os.path.getsize(fp)
    else:
        print(get_prefix() + f"Warning: Path not found or not a file/directory: {model_path}")
        return float('nan')
    return total_size / (1024 ** 3)


def load_model_and_tokenizer(model_path: str, use_gguf: bool = False):
    """Загружает модель и токенизатор из *model_path*.

    Поддерживает стандартные HF модели и GGUF (если use_gguf=True и model_path - путь к файлу).
    """
    common_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16, # Может потребоваться настройка
        "trust_remote_code": False # По умолчанию False для безопасности
    }
    tokenizer_kwargs = {}

    print(get_prefix() + f"Loading model and tokenizer from: {model_path}")

    if use_gguf and os.path.isfile(model_path):
        # Загрузка GGUF
        print(get_prefix() + f"Loading GGUF model: {model_path}")
        try:
            # Путь к директории для токенизатора обычно рядом с GGUF файлом или в base_model_dir
            # Предполагаем, что токенизатор лежит в директории на уровень выше GGUF файла
            tokenizer_dir = os.path.dirname(os.path.dirname(model_path)) # Пример: models/model-name/model.gguf -> models/model-name
            if not os.path.exists(os.path.join(tokenizer_dir, 'tokenizer_config.json')):
                 # Пытаемся найти в директории с конфигами GGUF
                 tokenizer_dir = os.path.dirname(model_path)

            print(get_prefix() + f"Attempting to load tokenizer from: {tokenizer_dir}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, **tokenizer_kwargs)
            # Для GGUF могут потребоваться специфичные параметры optimum.exporters.gguf
            # Здесь упрощенный вариант, т.к. AutoModelForCausalLM может сам подхватить gguf_file
            model = AutoModelForCausalLM.from_pretrained(
                tokenizer_dir, # Передаем директорию с конфигом
                gguf_file=model_path,
                **common_kwargs
            )

        except Exception as e:
            print(get_prefix() + f"Error loading GGUF model: {e}")
            raise
    elif os.path.isdir(model_path):
        # Загрузка стандартной HF модели
        print(get_prefix() + f"Loading standard HF model from directory: {model_path}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        except Exception as e:
            print(get_prefix() + f"Error loading HF model: {e}")
            raise
    else:
        raise ValueError(f"Invalid model_path: {model_path}. Must be a directory or a GGUF file path (if use_gguf=True).")

    return model, tokenizer


def measure_inference_time(model, tokenizer, test_input: str, num_runs: int = 5, max_new_tokens: int = 50) -> dict:
    """Замеряет метрики времени инференса"""
    inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
    # Удаляем token_type_ids, если они есть и не поддерживаются моделью
    if "token_type_ids" in inputs:
         inputs.pop("token_type_ids")

    # Устанавливаем pad_token_id, если он не задан
    if model.config.pad_token_id is None:
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
             model.config.pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
             print(get_prefix() + "Warning: pad_token_id not set, using eos_token_id.")
             model.config.pad_token_id = tokenizer.eos_token_id
        else:
             print(get_prefix() + "Warning: pad_token_id not set and eos_token_id not available. Generation might fail.")
             # Установить значение по умолчанию, если возможно, или обработать ошибку
             # model.config.pad_token_id = 0 # Пример

    input_tokens = inputs["input_ids"].shape[1]

    print(get_prefix() + "Measuring inference metrics...")
    times = []
    generated_tokens_list = []

    for run in range(num_runs):
        print(get_prefix() + f"Run {run + 1}/{num_runs}")
        start_time = time.time()
        try:
            with torch.no_grad():
                # Убедимся, что pad_token_id установлен перед генерацией
                if model.config.pad_token_id is None:
                     raise ValueError("pad_token_id must be set for generation.")

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True, # Используем кэш для реалистичного замера
                    pad_token_id=model.config.pad_token_id
                )
        except Exception as e:
            print(get_prefix() + f"Generation error on run {run + 1}: {e}")
            # Попытка очистить кэш перед следующей попыткой или выходом
            torch.cuda.empty_cache()
            continue # Пропускаем неудачный запуск

        elapsed = time.time() - start_time
        times.append(elapsed)
        # Убедимся, что outputs имеет ожидаемую форму
        if hasattr(outputs, 'shape') and len(outputs.shape) > 1:
            current_generated_tokens = outputs.shape[1] - input_tokens
            generated_tokens_list.append(current_generated_tokens)
        else:
            print(get_prefix() + f"Warning: Unexpected output format on run {run + 1}. Skipping token count.")
            generated_tokens_list.append(0) # Или другое значение по умолчанию


    if not times: # Если все запуски завершились ошибкой
        print(get_prefix() + "All inference runs failed.")
        return {
            "inference_time_avg_s": float('nan'),
            "input_tokens_per_sec": float('nan'),
            "generated_tokens_per_sec": float('nan')
        }

    avg_time = sum(times) / len(times)
    avg_gen_tokens = sum(generated_tokens_list) / len(generated_tokens_list) if generated_tokens_list else 0

    return {
        "inference_time_avg_s": avg_time,
        "input_tokens_per_sec": input_tokens / avg_time if avg_time > 0 else float('inf'),
        "generated_tokens_per_sec": avg_gen_tokens / avg_time if avg_time > 0 else float('inf')
    }


def measure_memory_usage(model) -> float:
    """Возвращает пиковое использование GPU памяти в гигабайтах"""
    if not torch.cuda.is_available():
        print(get_prefix() + "CUDA not available, skipping memory measurement.")
        return float('nan')

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Выполняем небольшой форвард пасс для активации памяти
    try:
        # Пытаемся определить vocab_size из конфига модели или токенизатора
        vocab_size = getattr(model.config, 'vocab_size', None)
        if vocab_size is None and hasattr(model, 'tokenizer') and model.tokenizer:
            vocab_size = getattr(model.tokenizer, 'vocab_size', 32000) # Запасной вариант
        elif vocab_size is None:
             vocab_size = 32000 # Запасной вариант по умолчанию

        dummy_input = torch.randint(0, vocab_size, (1, 128), device=model.device)
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(get_prefix() + f"Error during dummy forward pass for memory measurement: {e}")
        # Можно попробовать сгенерировать выход для замера
        try:
            dummy_input_ids = torch.randint(0, vocab_size, (1, 10), device=model.device)
            with torch.no_grad():
                 _ = model.generate(dummy_input_ids, max_new_tokens=5)
        except Exception as e2:
             print(get_prefix() + f"Error during dummy generation for memory measurement: {e2}")
             return float('nan') # Не удалось замерить

    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return memory_usage


def calculate_perplexity(model, tokenizer, test_dataset: str, max_length: int = 1024, stride: int = 512) -> float:
    """Вычисляет перплексию на тестовом наборе."""
    model.eval()
    print(get_prefix() + "Calculating perplexity...")

    try:
        encodings = tokenizer(test_dataset, return_tensors="pt")
    except Exception as e:
        print(get_prefix() + f"Error tokenizing dataset for perplexity: {e}")
        return float('nan')

    if "input_ids" not in encodings or encodings["input_ids"].numel() == 0:
        print(get_prefix() + "Warning: Could not encode dataset or dataset is empty. Skipping perplexity calculation.")
        return float('nan')

    input_ids = encodings["input_ids"].squeeze(0)
    if input_ids.dim() == 0: # Если dataset состоит из одного токена
        input_ids = input_ids.unsqueeze(0)

    total_loss = 0.0
    total_tokens = 0

    # Устанавливаем pad_token_id, если он не задан (важно для корректного вычисления лосса)
    if model.config.pad_token_id is None:
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
             print(get_prefix() + "Warning: pad_token_id not set for perplexity, using eos_token_id.")
             model.config.pad_token_id = tokenizer.eos_token_id
        else:
             print(get_prefix() + "Warning: pad_token_id and eos_token_id not available. Perplexity calculation might be inaccurate.")
             # Установка значения по умолчанию может быть неверной, лучше пропустить
             # return float('nan')

    for i in range(0, input_ids.size(0), stride):
        begin_loc = i
        end_loc = min(i + max_length, input_ids.size(0))
        trg_len = end_loc - begin_loc # Длина текущего сегмента
        if trg_len <= 1: # Пропускаем сегменты недостаточной длины
            continue

        input_slice = input_ids[begin_loc:end_loc].unsqueeze(0).to(model.device)
        target_slice = input_slice.clone()
        # Для Causal LM лосс считается со сдвигом, таргеты должны начинаться со второго токена
        # Некоторые модели делают это внутри, но безопаснее сделать явно
        # target_slice[:, :-1] = input_slice[:, 1:]
        # target_slice[:, -1] = model.config.pad_token_id # Паддим последний токен

        with torch.no_grad():
            try:
                # Передаем labels=input_slice, HF обработает сдвиг для CausalLM
                outputs = model(input_ids=input_slice, labels=input_slice)
                log_likelihood = outputs.loss * trg_len # loss - средний по токенам

            except Exception as e:
                 print(get_prefix() + f"Error calculating loss for perplexity slice: {e}")
                 continue # Пропускаем этот батч

        total_loss += log_likelihood.item()
        total_tokens += trg_len

        if end_loc == input_ids.size(0):
            break

    if total_tokens == 0:
        print(get_prefix() + "Warning: No tokens processed for perplexity calculation.")
        return float('nan')

    mean_loss = total_loss / total_tokens
    try:
        perplexity = torch.exp(torch.tensor(mean_loss)).item()
    except OverflowError:
        print(get_prefix() + "Warning: Overflow calculating perplexity. Loss might be too high.")
        perplexity = float('inf')
    return perplexity


def evaluate_single_model(model_path: str, test_input: str, test_dataset: str, use_gguf: bool = False) -> dict:
    """Оценивает одну квантизированную модель."""
    print(get_prefix() + f"Evaluating model: {model_path}")
    results = {
        "model_path": model_path,
        "method": "unknown", # Будет извлечено из пути, если возможно
        "config": "unknown", # Будет извлечено из пути, если возможно
        "model_size_gb": float('nan'),
        "memory_usage_gb": float('nan'),
        "perplexity": float('nan'),
        "inference_time_avg_s": float('nan'),
        "input_tokens_per_sec": float('nan'),
        "generated_tokens_per_sec": float('nan')
    }

    # Пытаемся извлечь метод и конфиг из пути
    try:
        filename = os.path.basename(model_path)
        parts = filename.split('-')
        if len(parts) > 2:
            results["method"] = parts[1] # Например, GGUF, GPTQ, AWQ, BNB
            results["config"] = "-".join(parts[2:]) # Все остальное считаем конфигом
            if use_gguf:
                 results["method"] = "GGUF"
                 # Для GGUF конфиг это тип квантования
                 gguf_type = filename.split('.')[0].split('-')[-1] # model-Q8_0.gguf -> Q8_0
                 results["config"] = gguf_type

    except Exception as e:
        print(get_prefix() + f"Could not parse method/config from path {model_path}: {e}")


    try:
        # Загрузка модели
        print(get_prefix() + "Loading model...")
        model, tokenizer = load_model_and_tokenizer(model_path, use_gguf=use_gguf)

        # Сбор метрик
        print(get_prefix() + "Collecting metrics...")
        results["model_size_gb"] = get_model_size(model_path)
        results["memory_usage_gb"] = measure_memory_usage(model)
        results["perplexity"] = calculate_perplexity(model, tokenizer, test_dataset)
        inference_metrics = measure_inference_time(model, tokenizer, test_input)
        results.update(inference_metrics) # Добавляем метрики инференса

        print(get_prefix() + f"Results for {model_path}:")
        for key, value in results.items():
             if isinstance(value, float):
                 print(f"    {key}: {value:.4f}")
             else:
                 print(f"    {key}: {value}")

        # Освобождение памяти
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(get_prefix() + f"Failed to evaluate model {model_path}: {e}")
        # Оставляем значения NaN по умолчанию

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized language models.")
    parser.add_argument(
        "model_paths",
        nargs='+',
        help="Path(s) to the quantized model directories or GGUF files to evaluate."
    )
    parser.add_argument(
        "--test_input",
        type=str,
        default=DEFAULT_TEST_INPUT,
        help="Text input for measuring inference speed."
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=DEFAULT_TEST_DATASET,
        help="Text dataset for calculating perplexity."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=DEFAULT_RESULTS_CSV,
        help="Path to save the evaluation results CSV file."
    )
    parser.add_argument(
        "--use_gguf",
        action='store_true',
        help="Indicate that the model_paths are paths to GGUF files (requires appropriate tokenizer location)."
    )
    # Можно добавить другие аргументы при необходимости (num_runs, max_length и т.д.)

    args = parser.parse_args()

    # Настройка логирования (можно вынести в отдельную функцию)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Отключить излишние логи библиотек (опционально)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("optimum").setLevel(logging.ERROR)


    print(get_prefix() + "Starting evaluation process...")
    all_results = []

    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            print(get_prefix() + f"Warning: Skipping non-existent path: {model_path}")
            continue

        # Определяем, использовать ли GGUF для текущего пути
        # Если флаг --use_gguf установлен, считаем все пути GGUF
        # Иначе, проверяем расширение файла
        is_gguf = args.use_gguf or model_path.lower().endswith(".gguf")

        results = evaluate_single_model(
            model_path=model_path,
            test_input=args.test_input,
            test_dataset=args.test_dataset,
            use_gguf=is_gguf
        )
        all_results.append(results)

    if not all_results:
        print(get_prefix() + "No models were evaluated.")
        return

    # Сохранение результатов в CSV
    results_df = pd.DataFrame(all_results)
    # Упорядочивание столбцов для лучшей читаемости
    cols_order = [
        "model_path", "method", "config", "model_size_gb", "memory_usage_gb",
        "perplexity", "inference_time_avg_s", "input_tokens_per_sec", "generated_tokens_per_sec"
    ]
    # Добавляем столбцы, которые могли отсутствовать в cols_order (на всякий случай)
    final_cols = cols_order + [col for col in results_df.columns if col not in cols_order]
    results_df = results_df[final_cols]


    try:
        results_df.to_csv(args.output_csv, sep=';', encoding='utf-8', index=False, float_format='%.4f')
        print(get_prefix() + f"Evaluation results saved to {args.output_csv}")
    except Exception as e:
        print(get_prefix() + f"Error saving results to CSV: {e}")

    print(get_prefix() + "Evaluation finished. Results:")
    # Используем to_string для красивого вывода в консоль
    print(results_df.to_string(index=False, float_format='%.4f'))


if __name__ == "__main__":
    main() 