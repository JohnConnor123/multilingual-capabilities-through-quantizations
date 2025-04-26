import os
import logging
from huggingface_hub import HfApi, create_repo, ModelCard, ModelCardData
from huggingface_hub import repo_exists

logger = logging.getLogger(__name__)

def ensure_repo_consistency(api: HfApi, repo_id: str, quant_dir: str, max_retries: int = 1):
    """Проверяет, что в указанном репозитории на HF Hub есть все файлы из quant_dir; при пропусках пытается повторно залить."""
    # Список удалённых файлов
    remote_files = set(api.list_repo_files(repo_id))

    # Список локальных файлов относительно quant_dir
    local_files = {
        os.path.relpath(os.path.join(dp, f), quant_dir).replace('\\', '/')
        for dp, _, fs in os.walk(quant_dir) for f in fs
    }

    missing = local_files - remote_files
    if not missing:
        logger.info("Consistency check passed: all files uploaded successfully")
        return

    # Повторная загрузка при пропусках
    logger.warning(f"Missing {len(missing)} files on hub, retrying upload")
    api.upload_folder(folder_path=quant_dir, path_in_repo="", repo_id=repo_id)
    remote_files = set(api.list_repo_files(repo_id))
    missing = local_files - remote_files
    if not missing:
        logger.info("Consistency check passed after retry")
        return

    logger.error(f"Consistency check failed, still missing files: {missing}")

def validate_model_id(model_id: str) -> bool:
    """Проверяет, что model_id имеет корректный формат и существует на HF Hub."""
    parts = model_id.split('/')
    if len(parts) not in (1, 2) or any(not part for part in parts):
        return False
    return repo_exists(model_id)

def push_to_hub(quant_dir: str, base_model: str = None, description: str = ""):
    """Создает репозиторий на HF, заливает файлы из quant_dir и при необходимости создает Model Card."""
    # Валидируем идентификатор базовой модели
    if base_model and not validate_model_id(base_model):
        logger.warning(f"Invalid base_model '{base_model}', base_model will not be linked to the Model Card")
        base_model = None
    api = HfApi()
    user_info = api.whoami()
    username = user_info.get("name")
    repo_name = os.path.basename(quant_dir)
    repo_id = f"{username}/{repo_name}"

    # Создаём репозиторий на HF
    create_repo(repo_id=repo_id, exist_ok=True)
    logger.info(f"Created HF repo {repo_id}")

    # Загружаем файлы
    api.upload_folder(folder_path=quant_dir, path_in_repo="", repo_id=repo_id)
    logger.info(f"Pushed files to HF Hub: {repo_id}")

    # Проверяем консистентность и при необходимости повторяем заливку
    ensure_repo_consistency(api, repo_id, quant_dir)

    # Пушим Model Card, если указана базовая модель
    if base_model:
        # Заметка о фреймворке в начале карточки
        note = "> ## **This quantization was done using the [quantization-benchmark](https://github.com/JohnConnor123/quantization-benchmark) framework**\n\n"
        card_data = ModelCardData(
            language="en",
            base_model=base_model
        )
        card = ModelCard.from_template(
            card_data, model_id=repo_name, model_description=description
        )
        card.text = note + card.text

        card.push_to_hub(repo_id)
        logger.info(f"Pushed Model Card to HF Hub: {repo_id}")