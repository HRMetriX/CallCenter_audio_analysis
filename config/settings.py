from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_IN = PROJECT_ROOT / "data" / "in"
DATA_OUT = PROJECT_ROOT / "data" / "out"
LOGS_DIR = PROJECT_ROOT / "logs"
PROCESSED_REGISTRY = PROJECT_ROOT / "processed_files.json"

# Параметры аудио
TARGET_SAMPLE_RATE = 16000

# Параметры VAD (Silero)
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MIN_SILENCE_DURATION_MS = 300
SPEECH_PAD_MS = 200

# Параметры ASR (GigaAM)
ASR_MODEL_NAME = "v3_e2e_rnnt"

# Логирование
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5