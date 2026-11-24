import json
import logging
import os
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOGS_DIR


# ----------------------------------------------------------------------------
# Логирование
# ----------------------------------------------------------------------------
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

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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


# ----------------------------------------------------------------------------
# Атомарная запись JSON
# ----------------------------------------------------------------------------
def atomic_write_json(path, data) -> None:
    """
    Пишет JSON через временный файл и os.replace.
    Обрыв процесса посреди записи не оставит повреждённый файл.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ----------------------------------------------------------------------------
# Реестр обработанных файлов
# ----------------------------------------------------------------------------
class ProcessedRegistry:
    """
    Индекс обработанных файлов (ключ - имя файла без расширения, stem).

    ИСТОЧНИК ИСТИНЫ - папки результатов: файл считается обработанным, если в
    <out_dir>/<stem>/ есть <stem>_transcript.json (он пишется ПОСЛЕДНИМ, поэтому
    его наличие означает, что остальные файлы результата уже записаны).

    Реестр лишь кэш с метаданными. При создании он сверяется с папкой результатов:
      - записи без результатов удаляются (в т.ч. мусор из старых форматов);
      - результаты без записи добавляются (метаданные берутся из transcript.json).
    Поэтому рассинхрон реестра и результатов исправляется сам.

    Формат файла: {"version": 2, "entries": {stem: {метаданные}}}.
    Старые форматы (список или {"processed": [...]}) читаются как временные записи
    и проходят ту же сверку.
    """

    def __init__(self, path, out_dir):
        self.path = Path(path)
        self.out_dir = Path(out_dir)
        self.entries: Dict[str, dict] = self._read_disk()
        self.sync_with_output()

    # --- чтение / запись -------------------------------------------------
    def _read_disk(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning(f"Не удалось прочитать реестр {self.path}: {e}. Восстановим по {self.out_dir}.")
            return {}

        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return {str(k): (v if isinstance(v, dict) else {}) for k, v in data["entries"].items()}

        # Старые форматы: список или {"processed": [...]}
        names = data if isinstance(data, list) else (data.get("processed", []) if isinstance(data, dict) else [])
        return {Path(str(n)).stem: {} for n in names}

    def _save(self) -> None:
        atomic_write_json(self.path, {"version": 2, "entries": self.entries})

    # --- сверка с результатами ------------------------------------------
    @staticmethod
    def _meta_from_transcript(tr_path: Path) -> Optional[dict]:
        """Метаданные из transcript.json; None, если файл не читается."""
        try:
            with open(tr_path, "r", encoding="utf-8") as f:
                tr = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning(f"Повреждён {tr_path}: {e}. Файл считается необработанным.")
            return None
        channels = tr.get("channels", {})
        return {
            "source_file": tr.get("source_file"),
            "processed_at": tr.get("processed_at"),
            "duration_sec": round(tr.get("duration_sec", 0) or 0, 3),
            "asr_errors": sum(c.get("asr_errors", 0) for c in channels.values()),
            "longform_segments": sum(c.get("longform_segments", 0) for c in channels.values()),
            "pipeline_version": tr.get("pipeline_version"),
        }

    def sync_with_output(self) -> None:
        """Приводит реестр в соответствие с папкой результатов."""
        done_dirs: Dict[str, Path] = {}
        if self.out_dir.is_dir():
            for d in self.out_dir.iterdir():
                tr = d / f"{d.name}_transcript.json"
                if d.is_dir() and tr.is_file():
                    done_dirs[d.name] = tr

        changed = False
        removed = [s for s in self.entries if s not in done_dirs]
        for stem in removed:
            del self.entries[stem]
            changed = True
        if removed:
            _logger.info(f"Реестр: удалено записей без результатов: {len(removed)}")

        added = 0
        for stem, tr in done_dirs.items():
            if self.entries.get(stem):
                continue
            meta = self._meta_from_transcript(tr)
            if meta is not None:
                self.entries[stem] = meta
                added += 1
                changed = True
        if added:
            _logger.info(f"Реестр: добавлено записей по найденным результатам: {added}")

        if changed or not self.path.exists():
            self._save()
        _logger.info(f"Реестр: обработано файлов: {len(self.entries)}")

    # --- API --------------------------------------------------------------
    def is_done(self, stem: str) -> bool:
        return stem in self.entries

    def files_with_errors(self) -> List[str]:
        """Файлы, в которых были сегменты с ошибкой ASR (для --retry-errors)."""
        return [s for s, m in self.entries.items() if m.get("asr_errors", 0) > 0]

    def mark_done(self, stem: str, meta: dict) -> None:
        """Добавляет запись, объединяя с версией на диске (не затирает чужие записи)."""
        self.entries = {**self._read_disk(), **self.entries}
        self.entries[stem] = meta
        self._save()


# ----------------------------------------------------------------------------
# Журнал сбоев
# ----------------------------------------------------------------------------
class FailureLog:
    """Файлы, обработка которых упала с исключением. Они повторяются при следующем запуске."""

    def __init__(self, path):
        self.path = Path(path)

    def _read(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def record(self, stem: str, error: str) -> None:
        data = self._read()
        attempts = data.get(stem, {}).get("attempts", 0) + 1
        data[stem] = {
            "error": error,
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attempts": attempts,
        }
        atomic_write_json(self.path, data)

    def clear(self, stem: str) -> None:
        data = self._read()
        if stem in data:
            del data[stem]
            atomic_write_json(self.path, data)


# ----------------------------------------------------------------------------
# Защита от одновременного запуска
# ----------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """Проверка живого процесса без os.kill (на Windows он убивает процесс)."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ACCESS_DENIED: процесс есть, но чужой
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PipelineLock:
    """
    Lock-файл с PID. Второй экземпляр пайплайна не стартует, пока жив первый
    (два процесса затирали бы друг другу реестр и результаты).
    Устаревший lock (процесса уже нет) удаляется автоматически.
    """

    def __init__(self, path):
        self.path = Path(path)

    def _read_pid(self) -> Optional[int]:
        try:
            return int(self.path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pid = self._read_pid()
                if pid and _pid_alive(pid):
                    raise RuntimeError(
                        f"Пайплайн уже запущен (PID {pid}). Lock-файл: {self.path}. "
                        f"Дождитесь завершения или остановите процесс."
                    )
                _logger.warning(f"Найден устаревший lock-файл {self.path}, удаляю.")
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            return self
        raise RuntimeError(f"Не удалось получить lock: {self.path}")

    def __exit__(self, exc_type, exc, tb):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False