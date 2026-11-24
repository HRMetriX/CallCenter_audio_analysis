from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_IN = PROJECT_ROOT / "data" / "in"
DATA_OUT = PROJECT_ROOT / "data" / "out"
LOGS_DIR = PROJECT_ROOT / "logs"
PROCESSED_REGISTRY = PROJECT_ROOT / "processed_files.json"

# Параметры аудио: весь пайплайн работает на этой частоте
TARGET_SAMPLE_RATE = 16000

# Параметры ASR (GigaAM)
ASR_MODEL_NAME = "v3_e2e_rnnt"
# Обычный transcribe принимает до ~25 с. Дольше идём через transcribe_longform.
LONGFORM_THRESHOLD_SEC = 24.5
# Предупреждение в логе, если один сегмент длиннее этого (longform будет медленным)
LONGFORM_WARN_SEC = 120.0

# Логирование
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5
