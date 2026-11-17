import json
import time
from pathlib import Path
from typing import List, Dict

from config.settings import DATA_OUT, PROCESSED_REGISTRY
from src.audio_processor import AudioProcessor
from src.asr_engine import ASREngine
from src.utils import setup_logger, log_execution_time, load_processed_registry, save_processed_registry

logger = setup_logger("Pipeline")


class CallCenterPipeline:
    def __init__(self):
        self.audio_proc = AudioProcessor()
        self.asr_eng = ASREngine()
        self.processed_files = load_processed_registry(PROCESSED_REGISTRY)

    def _save_dialogue_json(self, dialogue: List[Dict], out_path: Path):
        """Сохраняет хронологию диалога."""
        data = {
            "source_file": out_path.stem.replace("_dialogue", ""),
            "dialogue": dialogue
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_transcript_json(self, full_data: Dict, out_path: Path):
        """Сохраняет полную транскрипцию с метаданными."""
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)

    @log_execution_time(logger)
    def process_file(self, file_path: Path) -> bool:
        """Обрабатывает один аудиофайл."""
        if file_path.name in self.processed_files:
            logger.info(f"Файл {file_path.name} уже обработан. Пропуск.")
            return False

        logger.info(f"Начало обработки: {file_path.name}")
        start_time = time.time()

        try:
            # 1. Декодирование и разделение каналов
            channels_data = self.audio_proc.decode_and_split(str(file_path))

            full_transcript = {
                "source_file": file_path.name,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_sec": len(list(channels_data.values())[0]) / 16000,  # Берем длину первого канала
                "channels": {}
            }

            all_dialogue_segments = []

            # 2. Обработка каждого канала
            for ch_idx, audio_array in channels_data.items():
                speaker_role = "operator" if ch_idx == 0 else "client"
                logger.info(f"Обработка канала {ch_idx} ({speaker_role})...")

                # VAD
                segments = self.audio_proc.get_speech_segments(audio_array)

                channel_info = {
                    "speech_sec": sum(s['end'] - s['start'] for s in segments),
                    "segments_count": len(segments),
                    "full_text": "",
                    "words": []
                }

                channel_texts = []

                # ASR по сегментам
                for seg in segments:
                    seg_audio = self.audio_proc.extract_segment_audio(audio_array, seg['start'], seg['end'])
                    asr_res = self.asr_eng.transcribe_segment(seg_audio)

                    if asr_res['text']:
                        channel_texts.append(asr_res['text'])

                        # Добавляем слова с абсолютными таймстампами
                        for w in asr_res['words']:
                            w_abs = {
                                "word": w['word'],
                                "start": round(seg['start'] + w['start'], 3),
                                "end": round(seg['start'] + w['end'], 3)
                            }
                            channel_info['words'].append(w_abs)

                        # Для диалога берем весь текст сегмента
                        all_dialogue_segments.append({
                            "speaker": speaker_role,
                            "start": round(seg['start'], 3),
                            "end": round(seg['end'], 3),
                            "text": asr_res['text']
                        })

                channel_info['full_text'] = " ".join(channel_texts)
                full_transcript['channels'][speaker_role] = channel_info

            # 3. Сохранение результатов
            stem = file_path.stem
            out_dir = DATA_OUT / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            # Transcript JSON
            tr_path = out_dir / f"{stem}_transcript.json"
            self._save_transcript_json(full_transcript, tr_path)

            # Dialogue JSON
            # Сортируем диалог по времени начала
            all_dialogue_segments.sort(key=lambda x: x['start'])
            diag_path = out_dir / f"{stem}_dialogue.json"
            self._save_dialogue_json(all_dialogue_segments, diag_path)

            # 4. Обновление реестра
            self.processed_files.add(file_path.name)
            save_processed_registry(PROCESSED_REGISTRY, self.processed_files)

            elapsed = time.time() - start_time
            logger.info(f"Успешно завершено: {file_path.name} за {elapsed:.2f} сек.")
            return True

        except Exception as e:
            logger.error(f"Критическая ошибка при обработке {file_path.name}: {e}", exc_info=True)
            return False

    def run_batch(self, input_dir: Path):
        """Запускает обработку всех .ogg файлов в папке."""
        files = sorted(input_dir.glob("*.ogg"))
        if not files:
            logger.warning(f"В папке {input_dir} не найдено .ogg файлов.")
            return

        logger.info(f"Найдено {len(files)} файлов для обработки.")
        success_count = 0

        for f in files:
            if self.process_file(f):
                success_count += 1

        logger.info(f"Пакетная обработка завершена. Успешно: {success_count}/{len(files)}")