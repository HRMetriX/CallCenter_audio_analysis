import os
import torch
import gigaam
import soundfile as sf
import tempfile
import numpy as np
from pathlib import Path
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
        # Временно отключаем SSL для загрузки, если нужно (как в sandbox)
        import ssl
        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context

        try:
            self.model = gigaam.load_model(ASR_MODEL_NAME)
            logger.info("Модель GigaAM успешно загружена.")
        finally:
            ssl._create_default_https_context = original_context

    def transcribe_segment(self, audio_data: np.ndarray, output_path: Path = None) -> dict:
        """
        Транскрибирует один сегмент аудио.
        audio_data: numpy array (float32, mono, 16kHz)
        Возвращает словарь с текстом и словами.
        """
        import numpy as np

        # Создаем временный файл, если путь не передан
        temp_file = None
        file_path = str(output_path) if output_path else None

        if not file_path:
            temp_fd, file_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)

        try:
            # Сохраняем сегмент во временный WAV
            sf.write(file_path, audio_data, TARGET_SAMPLE_RATE)

            # Вызываем ASR с word_timestamps
            result = self.model.transcribe(file_path, word_timestamps=True)

            # Парсим результат (адаптировано под структуру GigaAM v3)
            text = ""
            words = []

            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, str):
                text = result

            if hasattr(result, 'words') and result.words:
                for w in result.words:
                    words.append({
                        "word": getattr(w, 'text', str(w)),
                        "start": float(getattr(w, 'start', 0.0)),
                        "end": float(getattr(w, 'end', 0.0))
                    })

            return {"text": text.strip(), "words": words}

        except Exception as e:
            logger.error(f"Ошибка транскрипции сегмента: {e}")
            return {"text": "", "words": []}
        finally:
            # Удаляем временный файл
            if not output_path and os.path.exists(file_path):
                os.remove(file_path)