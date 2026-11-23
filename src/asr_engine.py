import os
import ssl
import tempfile
from pathlib import Path

import gigaam
import numpy as np
import soundfile as sf
import soxr
import torch

from config.settings import ASR_MODEL_NAME, TARGET_SAMPLE_RATE
from src.utils import setup_logger

logger = setup_logger("ASREngine")


class ASREngine:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

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
        Транскрибирует один короткий сегмент (до ~25-30 с).
        audio_data: numpy mono float32 с частотой sr (будет приведена к 16 кГц).
        Возвращает {"text": str, "words": [{"word", "start", "end"}]}, время слов
        относительно начала сегмента.
        """
        audio_data = self._prepare(audio_data, sr)
        if len(audio_data) == 0:
            return {"text": "", "words": []}

        file_path = str(output_path) if output_path else None
        if not file_path:
            fd, file_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        try:
            sf.write(file_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
            result = self.model.transcribe(file_path, word_timestamps=True)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Ошибка транскрипции сегмента: {e}")
            return {"text": "", "words": []}
        finally:
            if not output_path and os.path.exists(file_path):
                os.remove(file_path)

    def transcribe_long(self, audio_data: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> dict:
        """Longform-транскрипция длинного массива."""
        audio_data = self._prepare(audio_data, sr)
        fd, file_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(file_path, audio_data, TARGET_SAMPLE_RATE, subtype="PCM_16")
            result = self.model.transcribe_longform(file_path)
            return self._parse_result(result)
        except Exception as e:
            logger.error(f"Ошибка longform транскрипции: {e}")
            return {"text": "", "words": []}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def _parse_result(self, result) -> dict:
        """Универсальный парсер результата GigaAM."""
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
