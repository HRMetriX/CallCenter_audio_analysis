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
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from config.settings import DATA_OUT  # noqa: E402
from src.pipeline import CallCenterPipeline  # noqa: E402
from src.utils import setup_logger  # noqa: E402

logger = setup_logger("Main")


def _read_duration(stem: str) -> float:
    """Длительность записи из сохранённого transcript.json (0, если файла нет)."""
    tr_path = DATA_OUT / stem / f"{stem}_transcript.json"
    if not tr_path.exists():
        return 0.0
    with open(tr_path, "r", encoding="utf-8") as f:
        return json.load(f).get("duration_sec", 0)


def main():
    parser = argparse.ArgumentParser(description="Пайплайн расшифровки записей call-центра (Pyannote + GigaAM v3)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Путь к конкретному аудиофайлу (.ogg)")
    group.add_argument("--dir", type=str, help="Путь к папке с аудиофайлами")

    parser.add_argument("--force", action="store_true", help="Принудительная обработка (игнорировать реестр)")
    parser.add_argument("--random", type=int, default=None,
                        help="Обработать N случайных файлов вместо всех")

    args = parser.parse_args()

    pipeline = CallCenterPipeline(random_count=args.random)

    if args.force:
        logger.warning("Режим принудительной обработки (--force). Реестр очищен.")
        pipeline.processed_files.clear()

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            sys.exit(1)

        t_start = time.time()
        success = pipeline.process_file(file_path)
        t_elapsed = time.time() - t_start

        if success:
            try:
                dur = _read_duration(file_path.stem)
                logger.info(
                    f"📊 ИТОГ: {file_path.name} | Длит: {dur:.1f}с | "
                    f"Время обр: {t_elapsed:.1f}с | Speed: {dur / t_elapsed:.2f}x"
                )
            except Exception:
                logger.info(f"✅ Обработка завершена за {t_elapsed:.1f} сек.")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"Папка не найдена: {dir_path}")
            sys.exit(1)

        files = sorted(dir_path.glob("*.ogg"))
        if not files:
            logger.warning(f"В папке {dir_path} не найдено .ogg файлов.")
            return

        if args.random and args.random > 0:
            if args.random >= len(files):
                logger.info(f"Запрос {args.random} файлов, но всего доступно {len(files)}. Обработаем все.")
            else:
                import random
                files = random.sample(files, args.random)
                logger.info(f"🎲 Выбрано {len(files)} случайных файлов из {dir_path}")

        logger.info(f"Найдено {len(files)} файлов. Запуск пакетной обработки...")
        total_start = time.time()
        success_count = 0
        total_duration = 0.0
        total_processing_time = 0.0

        for i, f in enumerate(files):
            t_file_start = time.time()
            if pipeline.process_file(f):
                success_count += 1
                t_file_elapsed = time.time() - t_file_start
                try:
                    dur = _read_duration(f.stem)
                    total_duration += dur
                    total_processing_time += t_file_elapsed
                    speed = dur / t_file_elapsed if t_file_elapsed > 0 else 0
                    logger.info(
                        f"[{i + 1}/{len(files)}] {f.name} | Длит: {dur:.1f}с | "
                        f"Обр: {t_file_elapsed:.1f}с | {speed:.2f}x"
                    )
                except Exception:
                    pass

        total_elapsed = time.time() - total_start
        avg_speed = total_duration / total_processing_time if total_processing_time > 0 else 0

        logger.info("=" * 60)
        logger.info("🏁 ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
        logger.info(f"Успешно: {success_count}/{len(files)}")
        logger.info(f"Общее время: {total_elapsed:.1f} сек ({total_elapsed / 60:.1f} мин)")
        logger.info(f"Суммарная длительность аудио: {total_duration:.1f} сек")
        logger.info(f"Средняя скорость обработки: {avg_speed:.2f}x realtime")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
