import json
import logging
import os
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOGS_DIR


def setup_logger(name: str = "CallCenterAnalytics") -> logging.Logger:
    """Настраивает логгер с ротацией файлов."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ротируемый файловый хендлер
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Консольный хендлер
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


_logger = setup_logger("Utils")


def log_execution_time(logger: logging.Logger):
    """Декоратор для замера времени выполнения функций."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Функция '{func.__name__}' выполнена за {elapsed_time:.2f} сек.")
            return result

        return wrapper

    return decorator


def load_processed_registry(path) -> set:
    """
    Загружает множество уже обработанных файлов.

    Поддерживает оба формата:
      - {"processed": ["a.ogg", ...]}  (текущий)
      - ["a.ogg", ...]                 (старый, просто список)
    При повреждённом файле возвращает пустое множество и пишет предупреждение.
    """
    path = Path(path)
    if not path.exists():
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _logger.warning(f"Не удалось прочитать реестр {path}: {e}. Начинаем с пустого.")
        return set()

    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict):
        return {str(x) for x in data.get("processed", [])}

    _logger.warning(f"Неизвестный формат реестра {path} ({type(data).__name__}). Начинаем с пустого.")
    return set()


def save_processed_registry(path, registry: set):
    """
    Сохраняет реестр обработанных файлов.
    Запись атомарная (временный файл + replace), чтобы обрыв процесса
    не оставил повреждённый JSON.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump({"processed": sorted(registry)}, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)