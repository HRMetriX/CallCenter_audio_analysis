import argparse
import sys
from pathlib import Path
from src.pipeline import CallCenterPipeline
from src.utils import setup_logger
from config.settings import DATA_IN

logger = setup_logger("Main")


def main():
    parser = argparse.ArgumentParser(description="Пайплайн расшифровки записей call-центра (GigaAM v3)")

    # Группа для выбора режима: файл или папка
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Путь к конкретному аудиофайлу (.ogg)")
    group.add_argument("--dir", type=str, help="Путь к папке с аудиофайлами")

    # Флаг принудительной обработки
    parser.add_argument("--force", action="store_true",
                        help="Принудительно обработать файлы, даже если они есть в реестре")

    args = parser.parse_args()

    pipeline = CallCenterPipeline()

    # Если передан флаг --force, очищаем реестр или удаляем из него текущие файлы
    if args.force:
        logger.warning("Режим принудительной обработки (--force). Реестр будет обновлен.")
        # В простой реализации мы просто не будем проверять реестр внутри process_file,
        # но для чистоты можно было бы передать флаг в конструктор пайплайна.
        # Для простоты сейчас мы полагаемся на то, что пользователь сам очистит processed_files.json при необходимости,
        # или мы можем временно отключить проверку.
        # Давайте сделаем проще: передадим флаг в пайплайн.
        pipeline.processed_files.clear()

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            sys.exit(1)
        pipeline.process_file(file_path)

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"Папка не найдена: {dir_path}")
            sys.exit(1)
        pipeline.run_batch(dir_path)


if __name__ == "__main__":
    main()