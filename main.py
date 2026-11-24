import os
import sys

from dotenv import load_dotenv

# 1. Переменные из .env нужны до импорта остальных модулей (HF_TOKEN, FFMPEG_DLL_DIR)
load_dotenv()

# 2. Windows: transcribe_longform использует torchcodec, которому нужны DLL FFmpeg
#    (shared-сборка). Python 3.8+ не ищет их в PATH, поэтому папку bin нужно добавить явно
#    и до импорта gigaam / pyannote.
_DLL_HANDLE = None


def _setup_ffmpeg_dlls():
    global _DLL_HANDLE
    dll_dir = os.getenv("FFMPEG_DLL_DIR")
    if not dll_dir or not hasattr(os, "add_dll_directory"):
        return
    if os.path.isdir(dll_dir):
        _DLL_HANDLE = os.add_dll_directory(dll_dir)
    else:
        print(f"ВНИМАНИЕ: FFMPEG_DLL_DIR не найден: {dll_dir}", file=sys.stderr)


_setup_ffmpeg_dlls()

# 3. Остальные импорты (после настройки DLL)
import argparse  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List  # noqa: E402

from config.settings import (  # noqa: E402
    DATA_OUT,
    FAILED_REGISTRY,
    LOCK_FILE,
    PROCESSED_REGISTRY,
)
from src.utils import FailureLog, PipelineLock, ProcessedRegistry, setup_logger  # noqa: E402

logger = setup_logger("Main")


def collect_files(args) -> List[Path]:
    """Список входных файлов по --file или --dir."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            sys.exit(1)
        return [file_path]

    dir_path = Path(args.dir)
    if not dir_path.is_dir():
        logger.error(f"Папка не найдена: {dir_path}")
        sys.exit(1)
    return sorted(dir_path.glob("*.ogg"))


def select_pending(files: List[Path], registry: ProcessedRegistry,
                   force: bool, retry_errors: bool) -> List[Path]:
    """
    Что нужно обработать:
      - необработанные файлы (нет в реестре; сюда же попадают упавшие ранее);
      - при --retry-errors: обработанные, у которых были сегменты с ошибкой ASR;
      - при --force: все файлы.
    """
    retry = set(registry.files_with_errors()) if retry_errors else set()
    pending: List[Path] = []
    skipped = 0
    retried = 0

    for f in files:
        if force or not registry.is_done(f.stem):
            pending.append(f)
        elif f.stem in retry:
            pending.append(f)
            retried += 1
        else:
            skipped += 1

    logger.info(
        f"Всего файлов: {len(files)} | уже обработано (пропуск): {skipped} | "
        f"к обработке: {len(pending)} (повторно из-за ошибок ASR: {retried})"
    )
    return pending


def main():
    parser = argparse.ArgumentParser(description="Пайплайн расшифровки записей call-центра (Pyannote + GigaAM v3)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Путь к конкретному аудиофайлу (.ogg)")
    group.add_argument("--dir", type=str, help="Путь к папке с аудиофайлами")

    parser.add_argument("--force", action="store_true",
                        help="Переобработать ВСЕ файлы, даже уже готовые (результаты перезаписываются)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Переобработать готовые файлы, где были сегменты с ошибкой ASR (asr_errors > 0)")
    parser.add_argument("--random", type=int, default=None,
                        help="Обработать N случайных файлов из необработанных")

    args = parser.parse_args()

    try:
        with PipelineLock(LOCK_FILE):
            files = collect_files(args)
            if not files:
                logger.warning("Входных .ogg файлов не найдено.")
                return

            registry = ProcessedRegistry(PROCESSED_REGISTRY, DATA_OUT)
            pending = select_pending(files, registry, args.force, args.retry_errors)

            if args.random and 0 < args.random < len(pending):
                pending = random.sample(pending, args.random)
                logger.info(f"🎲 Выбрано {len(pending)} случайных файлов из необработанных")

            if not pending:
                logger.info("Нечего обрабатывать: все файлы уже готовы. Модели не загружаются.")
                return

            if args.force:
                logger.warning("Режим --force: готовые результаты будут перезаписаны.")

            # Тяжёлые модели грузим только когда точно есть работа
            from src.pipeline import CallCenterPipeline
            pipeline = CallCenterPipeline(registry=registry, failures=FailureLog(FAILED_REGISTRY))
            pipeline.run_batch(pending)

    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
