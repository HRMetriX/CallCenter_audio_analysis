# src/diarizer.py
"""
Модуль диаризации на базе Pyannote Audio.
Принимает уже декодированный моно-массив 16 кГц (один канал записи).
"""
import os

import numpy as np
import torch
from pyannote.audio import Pipeline

from config.settings import TARGET_SAMPLE_RATE
from src.utils import setup_logger

logger = setup_logger("Diarizer")

MIN_SEGMENT_SEC = 0.1


class SpeakerDiarizer:
    """Обертка над Pyannote Diarization для call-center аналитики."""

    def __init__(self, hf_token: str = None):
        self.pipeline = None
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN не найден! Установите переменную окружения или передайте токен в конструктор."
            )

        self._load_pipeline()

    def _load_pipeline(self):
        logger.info("Загрузка модели pyannote/speaker-diarization-3.1...")
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
            if torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("Диаризация будет работать на GPU.")
            else:
                logger.warning("GPU не найден. Диаризация на CPU может быть медленной.")
            logger.info("Модель диаризации успешно загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель диаризации: {e}")
            raise

    @staticmethod
    def _get_annotation(output):
        """Совместимость версий pyannote (3.x возвращает Annotation, 4.x - объект с полями)."""
        for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
            if hasattr(output, attr):
                return getattr(output, attr)
        return output

    def diarize_channel(self, audio16: np.ndarray, channel_idx: int = -1) -> list:
        """
        Диаризация одного канала.

        Args:
            audio16: mono float32, TARGET_SAMPLE_RATE
            channel_idx: только для логов

        Returns:
            Список словарей: start, end, speaker, duration
        """
        if audio16.ndim != 1:
            raise ValueError(f"Ожидается mono, shape={audio16.shape}")

        logger.debug(f"Диаризация канала {channel_idx} ({len(audio16) / TARGET_SAMPLE_RATE:.1f} с)...")

        input_data = {
            "waveform": torch.from_numpy(np.ascontiguousarray(audio16, dtype=np.float32)).unsqueeze(0),
            "sample_rate": TARGET_SAMPLE_RATE,
        }

        try:
            annotation = self._get_annotation(self.pipeline(input_data))

            segments = []
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                duration = segment.end - segment.start
                if duration > MIN_SEGMENT_SEC:
                    segments.append({
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "speaker": speaker,
                        "duration": round(duration, 3),
                    })

            segments.sort(key=lambda x: x["start"])
            logger.info(f"Канал {channel_idx}: найдено {len(segments)} речевых сегментов.")
            return segments

        except Exception as e:
            logger.error(f"Ошибка диаризации канала {channel_idx}: {e}", exc_info=True)
            return []