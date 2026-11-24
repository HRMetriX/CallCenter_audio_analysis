import os
import ssl
import tempfile
from pathlib import Path

import gigaam
import numpy as np
import soundfile as sf
import soxr
import torch

from config.settings import (
    ASR_MODEL_NAME,
    LONGFORM_THRESHOLD_SEC,
    LONGFORM_WARN_SEC,
    TARGET_SAMPLE_RATE,
)
from src.utils import setup_logger

logger = setup_logger("ASREngine")


class ASREngine:
    """
    Обертка над GigaAM.

    Режим выбирается автоматически по длительности буфера:
      - до LONGFORM_THRESHOLD_SEC: обычный transcribe;
      - дольше: transcribe_longform (режет по паузам собственным VAD).
    Longform требует HF_TOKEN в окружении, FFmpeg shared-сборку (torchcodec)
    и принятые условия моделей Pyannote на Hugging Face.
    """

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
        if not os.getenv("HF_TOKEN"):
            logger.warning("HF_TOKEN не задан: transcribe_longform может не запуститься.")

    def _load_model(self):
        """Загружает модель GigaAM."""
        logger.info(f"Загрузка модели GigaAM '{ASR_MODEL_NAME}' на устройстве {self.device}...")
        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            self.model = gigaam.load_model(ASR_MODEL_NAME)
            logger.info("Модель GigaAM успешно загружена.")
        finally:
            ssl._create_default_https_context = original_context

    @staticmethod
    def _prepare(audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Проверка формы и гарантированный ресемплинг к TARGET_SAMPLE_RATE."""
        if audio_data.ndim != 1:
            raise ValueError(f"Ожидается mono, shape={audio_data.shape}")
        audio_data = audio_data.astype(np.float32, copy=False)
        if sr != TARGET_SAMPLE_RATE:
            audio_data = soxr.resample(audio_data, sr, TARGET_SAMPLE_RATE, quality="VHQ")
        return audio_data

    def transcribe_segment(self, audio_data: np.ndarray, sr: int = TARGET_SAMPLE_RATE,
                           output_path: Path = None) -> dict:
        """
        Транскрибирует один сегмент любой длины.

        audio_data: numpy mono float32 с частотой sr (будет приведена к 16 кГц).

        Возвращает:
            {
              "text": str,
              "words": [{"word", "start", "end"}],  # время от начала сегмента
              "mode": "regular" | "longform" | "empty",
              "error": None | str,
            }
        """
        audio_data = self._prepare(audio_data, sr)
        if len(audio_data) == 0:
            return {"text": "", "words": [], "mode": "empty", "error": None}

        duration = len(audio_data) / TARGET_SAMPLE_RATE
        mode = "longform" if duration > LONGFORM_THRESHOLD_SEC else "regular"
        if duration > LONGFORM_WARN_SEC:
            logger.warning(f"Очень длинный сегмент: {duration:.1f} с (режим {mode}).")

        file_path = str(output_path) if output_path else None
        if not file_path:
            fd, file_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        try:
            sf.write(file_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
            if mode == "longform":
                result = self.model.transcribe_longform(file_path, word_timestamps=True)
            else:
                result = self.model.transcribe(file_path, word_timestamps=True)
            parsed = self._parse_result(result)
            parsed.update({"mode": mode, "error": None})
            return parsed
        except Exception as e:
            logger.error(f"Ошибка транскрипции сегмента ({mode}, {duration:.1f} с): {e}")
            return {"text": "", "words": [], "mode": mode, "error": f"{type(e).__name__}: {e}"}
        finally:
            if not output_path and os.path.exists(file_path):
                os.remove(file_path)

    def _parse_result(self, result) -> dict:
        """
        Парсер результата GigaAM. Подходит и для TranscriptionResult (transcribe),
        и для LongformTranscriptionResult (transcribe_longform): у обоих есть
        .text и .words со временем относительно начала переданного аудио.
        """
        text = ""
        words = []

        if hasattr(result, "text"):
            text = result.text
        elif isinstance(result, str):
            text = result

        if hasattr(result, "words") and result.words:
            for w in result.words:
                words.append({
                    "word": getattr(w, "text", str(w)),
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0)),
                })

        return {"text": (text or "").strip(), "words": words}
