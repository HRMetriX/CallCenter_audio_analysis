import subprocess
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import soundfile as sf
from io import BytesIO
from config.settings import TARGET_SAMPLE_RATE, VAD_THRESHOLD, MIN_SPEECH_DURATION_MS, MIN_SILENCE_DURATION_MS, \
    SPEECH_PAD_MS
from src.utils import setup_logger

logger = setup_logger("AudioProcessor")


class AudioProcessor:
    def __init__(self):
        self.vad_model = None
        self._load_vad()

    def _load_vad(self):
        """Загружает модель Silero VAD."""
        logger.info("Загрузка модели Silero VAD...")
        self.vad_model = load_silero_vad()
        logger.info("Модель Silero VAD загружена.")

    def decode_and_split(self, file_path: str) -> dict[int, np.ndarray]:
        """
        Декодирует аудиофайл, ресамплит до 16кГц и разделяет на каналы.
        Возвращает словарь {channel_index: numpy_array}.
        """
        logger.info(f"Декодирование и разделение каналов: {file_path}")

        # Получаем информацию о количестве каналов через ffprobe
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            "-select_streams", "a:0",
            file_path
        ]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            channels = int(result.stdout.strip())
        except Exception as e:
            logger.error(f"Ошибка при определении количества каналов: {e}")
            raise

        channels_data = {}
        for ch in range(channels):
            # Используем фильтр pan для выбора конкретного канала (c0=c0 - левый, c0=c1 - правый)
            cmd = [
                "ffmpeg", "-v", "error",
                "-i", file_path,
                "-af", f"pan=mono|c0=c{ch}",
                "-f", "f32le",
                "-ac", "1",
                "-ar", str(TARGET_SAMPLE_RATE),
                "-"
            ]
            try:
                process = subprocess.run(cmd, capture_output=True, check=True)
                raw_audio = process.stdout
                audio_np = np.frombuffer(raw_audio, dtype=np.float32)
                channels_data[ch] = audio_np
                logger.debug(f"Канал {ch} декодирован: {len(audio_np)} сэмплов.")
            except Exception as e:
                logger.error(f"Ошибка декодирования канала {ch}: {e}")
                # Выведем stderr для отладки, если ошибка осталась
                if hasattr(e, 'stderr'):
                    logger.error(f"ffmpeg stderr: {e.stderr.decode('utf-8', errors='replace')}")
                raise

        return channels_data

    def get_speech_segments(self, audio: np.ndarray) -> list[dict]:
        """
        Определяет сегменты речи с помощью Silero VAD.
        Возвращает список словарей {'start': float, 'end': float}.
        """
        tensor = torch.from_numpy(audio.astype(np.float32))
        timestamps = get_speech_timestamps(
            tensor,
            self.vad_model,
            return_seconds=True,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        logger.debug(f"Найдено {len(timestamps)} речевых сегментов.")
        return timestamps

    def extract_segment_audio(self, full_audio: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
        """
        Извлекает фрагмент аудио из полного массива по времени.
        Не создает файлов на диске.
        """
        sr = TARGET_SAMPLE_RATE
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        # Защита от выхода за границы массива
        if start_sample < 0: start_sample = 0
        if end_sample > len(full_audio): end_sample = len(full_audio)

        return full_audio[start_sample:end_sample]