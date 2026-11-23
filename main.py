import argparse
import sys
import time
from pathlib import Path
from src.pipeline import CallCenterPipeline
from src.utils import setup_logger
from config.settings import DATA_IN

from dotenv import load_dotenv
import os

# Загружаем переменные из .env файла
load_dotenv()

logger = setup_logger("Main")


def main():
    parser = argparse.ArgumentParser(description="Пайплайн расшифровки записей call-центра (Pyannote + GigaAM v3)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Путь к конкретному аудиофайлу (.ogg)")
    group.add_argument("--dir", type=str, help="Путь к папке с аудиофайлами")

    parser.add_argument("--force", action="store_true", help="Принудительная обработка (игнорировать реестр)")
    parser.add_argument("--random", type=int, default=None,
                        help="Обработать N случайных файлов вместо всех")

    args = parser.parse_args()

    # Передаем random_count в конструктор пайплайна
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
                import json
                from config.settings import DATA_OUT
                stem = file_path.stem
                tr_path = DATA_OUT / stem / f"{stem}_transcript.json"
                if tr_path.exists():
                    with open(tr_path, 'r') as f:
                        data = json.load(f)
                        dur = data.get('duration_sec', 0)
                        logger.info(
                            f"📊 ИТОГ: {file_path.name} | Длит: {dur:.1f}с | Время обр: {t_elapsed:.1f}с | Speed: {dur / t_elapsed:.2f}x")
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

        # Логика случайной выборки
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
        total_duration = 0
        total_processing_time = 0

        for i, f in enumerate(files):
            t_file_start = time.time()
            if pipeline.process_file(f):
                success_count += 1
                t_file_elapsed = time.time() - t_file_start

                try:
                    import json
                    from config.settings import DATA_OUT
                    stem = f.stem
                    tr_path = DATA_OUT / stem / f"{stem}_transcript.json"
                    if tr_path.exists():
                        with open(tr_path, 'r') as jf:
                            data = json.load(jf)
                            dur = data.get('duration_sec', 0)
                            total_duration += dur
                            total_processing_time += t_file_elapsed
                            speed = dur / t_file_elapsed if t_file_elapsed > 0 else 0
                            logger.info(
                                f"[{i + 1}/{len(files)}] {f.name} | Длит: {dur:.1f}с | Обр: {t_file_elapsed:.1f}с | {speed:.2f}x")
                except Exception:
                    pass

        total_elapsed = time.time() - total_start
        avg_speed = total_duration / total_processing_time if total_processing_time > 0 else 0

        logger.info("=" * 60)
        logger.info(f"🏁 ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
        logger.info(f"Успешно: {success_count}/{len(files)}")
        logger.info(f"Общее время: {total_elapsed:.1f} сек ({total_elapsed / 60:.1f} мин)")
        logger.info(f"Суммарная длительность аудио: {total_duration:.1f} сек")
        logger.info(f"Средняя скорость обработки: {avg_speed:.2f}x realtime")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()