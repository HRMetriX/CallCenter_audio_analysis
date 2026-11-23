import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import DATA_OUT, PROCESSED_REGISTRY, TARGET_SAMPLE_RATE
from src.asr_engine import ASREngine
from src.audio_processor import AudioProcessor
from src.diarizer import SpeakerDiarizer
from src.utils import (
    load_processed_registry,
    log_execution_time,
    save_processed_registry,
    setup_logger,
)

logger = setup_logger("Pipeline")

SR = TARGET_SAMPLE_RATE
PAD_SEC = 0.25          # паддинг вокруг сегмента при вырезке
MERGE_GAP_SEC = 0.6     # склеивать сегменты одного канала с паузой меньше этого
MAX_SEG_SEC = 28.0      # максимум для одного вызова ASR
MIN_ASR_SEC = 0.25      # речь короче этого в ASR не отправляется
SHORT_SOUND = "[короткий звук]"
CHANNEL_ROLES = {0: "operator", 1: "client"}


def merge_segments(segs: List[Tuple[float, float]], gap: float = MERGE_GAP_SEC,
                   max_len: float = MAX_SEG_SEC) -> List[Tuple[float, float]]:
    """Склеивает близкие сегменты и режет слишком длинные на куски <= max_len."""
    merged: List[List[float]] = []
    for s, e in sorted(segs):
        if merged and s - merged[-1][1] < gap and e - merged[-1][0] <= max_len:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    out: List[Tuple[float, float]] = []
    for s, e in merged:
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


SPEAKER_LABELS = {"operator": "оператор", "client": "клиент"}


def merge_turns(utterances: List[Dict]) -> List[Dict]:
    """
    Склеивает подряд идущие реплики одного спикера в одну (формат "диалог").
    Реплики-заглушки [короткий звук] в читаемый диалог не попадают и
    не разрывают реплику собеседника.
    """
    turns: List[Dict] = []
    for u in sorted(utterances, key=lambda x: x["start"]):
        if "short_sound" in u.get("flags", []) or not u["text"].strip():
            continue
        if turns and turns[-1]["speaker"] == u["speaker"]:
            turns[-1]["text"] += " " + u["text"].strip()
            turns[-1]["end"] = max(turns[-1]["end"], u["end"])
        else:
            turns.append({
                "speaker": u["speaker"],
                "start": u["start"],
                "end": u["end"],
                "text": u["text"].strip(),
            })
    return turns


class CallCenterPipeline:
    def __init__(self, random_count: Optional[int] = None):
        self.audio_proc = AudioProcessor()
        self.asr_eng = ASREngine()
        self.diarizer = SpeakerDiarizer()
        self.processed_files = load_processed_registry(PROCESSED_REGISTRY)
        self.random_count = random_count

    def _save_dialogue_json(self, dialogue: List[Dict], out_path: Path):
        data = {
            "source_file": out_path.stem.replace("_dialogue", ""),
            "dialogue": dialogue,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_readable_json(self, source_file: str, duration_sec: float,
                            utterances: List[Dict], out_path: Path):
        """Сохраняет диалог в читаемом виде: "оператор: ...", "клиент: ..."."""
        turns = merge_turns(utterances)
        data = {
            "source_file": source_file,
            "duration_sec": round(duration_sec, 3),
            "turns": turns,
            "lines": [
                f"{SPEAKER_LABELS.get(t['speaker'], t['speaker'])}: {t['text']}"
                for t in turns
            ],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_transcript_json(self, full_data: Dict, out_path: Path):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)

    def _process_channel(self, ch_idx: int, audio, total_sec: float) -> List[Dict]:
        """Диаризация + посегментная ASR одного канала. Возвращает список реплик."""
        role = CHANNEL_ROLES.get(ch_idx, f"channel_{ch_idx}")
        logger.info(f"Диаризация канала {ch_idx} ({role})...")

        raw = self.diarizer.diarize_channel(audio, ch_idx)
        spans = merge_segments([(x["start"], x["end"]) for x in raw])
        logger.info(f"Канал {ch_idx}: {len(raw)} сегментов Pyannote -> {len(spans)} после слияния.")

        utterances: List[Dict] = []
        for s, e in spans:
            ps, pe = max(0.0, s - PAD_SEC), min(total_sec, e + PAD_SEC)
            seg = self.audio_proc.extract_segment_audio(audio, ps, pe, SR)

            # Контроль согласованности SR: длина буфера должна соответствовать времени
            if abs(len(seg) / SR - (pe - ps)) > 0.05:
                raise RuntimeError(
                    f"Рассинхрон SR: буфер {len(seg) / SR:.3f} с, ожидалось {pe - ps:.3f} с"
                )

            flags: List[str] = []
            words: List[Dict] = []

            if e - s < MIN_ASR_SEC:
                text = SHORT_SOUND
                flags.append("short_sound")
            else:
                seg = self.audio_proc.normalize_segment(seg)
                asr_res = self.asr_eng.transcribe_segment(seg, sr=SR)
                text = asr_res["text"].strip()
                words = [
                    {
                        "word": w["word"],
                        "start": round(ps + w["start"], 3),
                        "end": round(ps + w["end"], 3),
                    }
                    for w in asr_res["words"]
                ]

            if not text:
                continue

            utterances.append({
                "speaker": role,
                "channel": ch_idx,
                "start": round(s, 3),
                "end": round(e, 3),
                "duration": round(e - s, 3),
                "text": text,
                "words": words,
                "flags": flags,
            })
        return utterances

    @log_execution_time(logger)
    def process_file(self, file_path: Path) -> bool:
        """Обрабатывает один аудиофайл: декодирование -> диаризация -> ASR -> JSON."""
        if file_path.name in self.processed_files:
            logger.info(f"Файл {file_path.name} уже обработан. Пропуск.")
            return False

        logger.info(f"Начало обработки: {file_path.name}")
        start_time = time.time()

        try:
            analysis = self.audio_proc.analyze_and_fix(str(file_path))
            channels_data = analysis["channels_data"]
            total_sec = len(channels_data[0]) / SR

            full_transcript = {
                "source_file": file_path.name,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_fixes": analysis["fixes"],
                "src_sample_rate_hz": analysis["meta"]["audio"]["src_sample_rate_hz"],
                "pipeline_sample_rate_hz": SR,
                "duration_sec": total_sec,
                "channels": {},
            }

            all_utterances: List[Dict] = []

            for ch_idx, audio in channels_data.items():
                utts = self._process_channel(ch_idx, audio, total_sec)
                role = CHANNEL_ROLES.get(ch_idx, f"channel_{ch_idx}")

                speech_texts = [u["text"] for u in utts if "short_sound" not in u["flags"]]
                channel_words = [w for u in utts for w in u["words"]]
                full_transcript["channels"][role] = {
                    "speech_sec": round(sum(u["duration"] for u in utts), 3),
                    "segments_count": len(utts),
                    "short_sounds": sum("short_sound" in u["flags"] for u in utts),
                    "full_text": " ".join(speech_texts),
                    "words": channel_words,
                }
                all_utterances.extend(utts)

            all_utterances.sort(key=lambda u: u["start"])

            stem = file_path.stem
            out_dir = DATA_OUT / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            self._save_transcript_json(full_transcript, out_dir / f"{stem}_transcript.json")
            self._save_dialogue_json(all_utterances, out_dir / f"{stem}_dialogue.json")
            self._save_readable_json(
                file_path.name, total_sec, all_utterances, out_dir / f"{stem}_readable.json"
            )

            self.processed_files.add(file_path.name)
            save_processed_registry(PROCESSED_REGISTRY, self.processed_files)

            logger.info(f"✅ Успешно: {file_path.name} за {time.time() - start_time:.2f} сек.")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {file_path.name}: {e}", exc_info=True)
            return False

    def run_batch(self, input_dir: Path):
        """Пакетная обработка с поддержкой случайной выборки."""
        files = sorted(input_dir.glob("*.ogg"))
        if not files:
            logger.warning(f"В папке {input_dir} не найдено .ogg файлов.")
            return

        if self.random_count and self.random_count > 0:
            if self.random_count >= len(files):
                logger.info(f"Запрос {self.random_count} файлов, всего доступно {len(files)}. Обработаем все.")
            else:
                files = random.sample(files, self.random_count)
                logger.info(f"🎲 Выбрано {len(files)} случайных файлов из {input_dir}")

        logger.info(f"Найдено {len(files)} файлов для обработки.")
        success_count = 0
        total_duration = 0.0
        total_processing_time = 0.0

        for i, f in enumerate(files):
            t_start = time.time()
            if self.process_file(f):
                success_count += 1
                t_elapsed = time.time() - t_start
                try:
                    tr_path = DATA_OUT / f.stem / f"{f.stem}_transcript.json"
                    if tr_path.exists():
                        with open(tr_path, "r", encoding="utf-8") as jf:
                            dur = json.load(jf).get("duration_sec", 0)
                        total_duration += dur
                        total_processing_time += t_elapsed
                        speed = dur / t_elapsed if t_elapsed > 0 else 0
                        logger.info(f"[{i + 1}/{len(files)}] {f.name} | {dur:.1f}с | {t_elapsed:.1f}с | {speed:.2f}x")
                except Exception:
                    pass

        avg_speed = total_duration / total_processing_time if total_processing_time > 0 else 0
        logger.info("=" * 60)
        logger.info("🏁 ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
        logger.info(f"Успешно: {success_count}/{len(files)}")
        logger.info(f"Суммарная длительность: {total_duration:.1f} сек")
        logger.info(f"Средняя скорость: {avg_speed:.2f}x realtime")
        logger.info("=" * 60)