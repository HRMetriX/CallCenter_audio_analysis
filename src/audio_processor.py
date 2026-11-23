import json
import subprocess

import numpy as np
import soxr

from config.settings import TARGET_SAMPLE_RATE
from src.utils import setup_logger

logger = setup_logger("AudioProcessor")


class AudioProcessor:
    """
    Предобработка аудио.

    ИНВАРИАНТ: весь пайплайн работает на TARGET_SAMPLE_RATE (16 кГц).
    Исходная частота из ffprobe используется только для декодирования и логов.
    """

    def analyze_and_fix(self, file_path: str) -> dict:
        """Декодирует файл один раз и приводит все каналы к TARGET_SAMPLE_RATE."""
        logger.info(f"Анализ и подготовка файла: {file_path}")

        meta = self._get_meta(file_path)
        n_ch = meta["audio"]["channels"]
        src_sr = meta["audio"]["sample_rate_hz"]

        raw_channels = self._decode_raw(file_path, n_ch, src_sr)

        channels_data = {}
        for i, ch in enumerate(raw_channels):
            if src_sr != TARGET_SAMPLE_RATE:
                ch = soxr.resample(ch, src_sr, TARGET_SAMPLE_RATE, quality="VHQ")
            channels_data[i] = np.ascontiguousarray(ch, dtype=np.float32)

        logger.info(
            f"Каналов: {n_ch}, исходный SR: {src_sr} Гц -> {TARGET_SAMPLE_RATE} Гц, "
            f"длительность: {len(channels_data[0]) / TARGET_SAMPLE_RATE:.1f} с"
        )

        return {
            "fixes": [],
            "channels_data": channels_data,
            "sr": TARGET_SAMPLE_RATE,
            "meta": {"audio": {**meta["audio"], "src_sample_rate_hz": src_sr}},
        }

    def _get_meta(self, file_path: str) -> dict:
        """Метаданные через ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
        if not audio_streams:
            raise RuntimeError("No audio stream found")
        a = audio_streams[0]
        return {
            "audio": {
                "channels": int(a.get("channels", 0)),
                "sample_rate_hz": int(a.get("sample_rate", 0)),
            }
        }

    def _decode_raw(self, file_path: str, channels: int, sr: int) -> list:
        """Декодирует все каналы в нативном SR без ресемплинга ffmpeg."""
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", file_path,
            "-f", "f32le",
            "-ac", str(channels),
            "-ar", str(sr),
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=300)
        audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if channels > 1:
            audio = audio.reshape(-1, channels)
            return [audio[:, i] for i in range(channels)]
        return [audio]

    @staticmethod
    def extract_segment_audio(full_audio: np.ndarray, start_sec: float, end_sec: float,
                              sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
        """Вырезает фрагмент по времени (с защитой от выхода за границы)."""
        start = max(0, int(start_sec * sr))
        end = min(len(full_audio), int(end_sec * sr))
        if start >= end:
            return np.array([], dtype=np.float32)
        return full_audio[start:end]

    @staticmethod
    def normalize_segment(x: np.ndarray, target_dbfs: float = -23.0,
                          max_gain_db: float = 30.0) -> np.ndarray:
        """
        Локальная нормализация сегмента с ограничением усиления.
        Вытягивает тихого оператора, не раздувая шум до бесконечности.
        """
        if len(x) == 0:
            return x
        rms = np.sqrt(np.mean(x ** 2) + 1e-12)
        gain = min(10 ** (target_dbfs / 20) / rms, 10 ** (max_gain_db / 20))
        return np.clip(x * gain, -1.0, 1.0).astype(np.float32)
