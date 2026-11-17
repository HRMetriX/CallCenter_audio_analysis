import logging
import time
import json
from functools import wraps
from logging.handlers import RotatingFileHandler
from config.settings import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOGS_DIR


def setup_logger(name: str = "CallCenterAnalytics") -> logging.Logger:
    """Настраивает логгер с ротацией файлов."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Форматтер
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
    """Загружает множество уже обработанных файлов."""
    if not path.exists():
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get("processed", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def save_processed_registry(path, registry: set):
    """Сохраняет реестр обработанных файлов."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"processed": list(registry)}, f, indent=2)