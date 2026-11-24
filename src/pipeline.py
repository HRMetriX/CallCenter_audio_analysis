import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import (
    DATA_OUT,
    LONGFORM_THRESHOLD_SEC,
    PIPELINE_VERSION,
    TARGET_SAMPLE_RATE,
)
from src.asr_engine import ASREngine
from src.audio_processor import AudioProcessor
from src.diarizer import SpeakerDiarizer
from src.utils import (
    FailureLog,
    ProcessedRegistry,
    atomic_write_json,
    log_execution_time,
    setup_logger,
)

logger = setup_logger("Pipeline")

SR = TARGET_SAMPLE_RATE
PAD_SEC = 0.25          # паддинг вокруг сегмента при вырезке
MERGE_GAP_SEC = 0.6     # склеивать сегменты одного канала с паузой меньше этого
MAX_MERGE_SEC = 23.0    # склейка соседей не создаёт спан длиннее этого (чтобы остаться в обычном режиме ASR)
MIN_ASR_SEC = 0.25      # речь короче этого в ASR не отправляется
SHORT_SOUND = "[короткий звук]"
CHANNEL_ROLES = {0: "operator", 1: "client"}
SPEAKER_LABELS = {"operator": "оператор", "client": "клиент"}


def merge_segments(segs: List[Tuple[float, float]], gap: float = MERGE_GAP_SEC,
                   max_merge: float = MAX_MERGE_SEC) -> List[Tuple[float, float]]:
    """
    Склеивает близкие сегменты (пауза < gap), пока объединённый спан не превысит max_merge.
    Сегменты, которые уже длиннее max_merge сами по себе, не режутся: они уйдут
    в ASR целиком (и будут обработаны через transcribe_longform).
    """
    merged: List[List[float]] = []
    for s, e in sorted(segs):
        if merged and s - merged[-1][1] < gap and e - merged[-1][0] <= max_merge:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merge_turns(utterances: List[Dict]) -> List[Dict]:
    """
    Склеивает подряд идущие реплики одного спикера в одну (формат "диалог").
    Реплики-заглушки [короткий звук] и пустые реплики (в том числе с ошибкой ASR)
    в читаемый диалог не попадают и не разрывают реплику собеседника.
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
    def __init__(self, registry: ProcessedRegistry, failures: FailureLog):
        """
        Отбор файлов (что уже обработано, --force, --retry-errors, --random)
        делает вызывающий код (main.py); пайплайн обрабатывает то, что ему передали.
        """
        self.registry = registry
        self.failures = failures
        self.audio_proc = AudioProcessor()
        self.asr_eng = ASREngine()
        self.diarizer = SpeakerDiarizer()

    # --- сохранение результатов (все записи атомарные) -------------------
    @staticmethod
    def _save_dialogue_json(dialogue: List[Dict], out_path: Path):
        atomic_write_json(out_path, {
            "source_file": out_path.stem.replace("_dialogue", ""),
            "dialogue": dialogue,
        })

    @staticmethod
    def _save_readable_json(source_file: str, duration_sec: float,
                            utterances: List[Dict], out_path: Path):
        """Сохраняет диалог в читаемом виде: "оператор: ...", "клиент: ..."."""
        turns = merge_turns(utterances)
        atomic_write_json(out_path, {
            "source_file": source_file,
            "duration_sec": round(duration_sec, 3),
            "turns": turns,
            "lines": [
                f"{SPEAKER_LABELS.get(t['speaker'], t['speaker'])}: {t['text']}"
                for t in turns
            ],
        })

    @staticmethod
    def _save_transcript_json(full_data: Dict, out_path: Path):
        atomic_write_json(out_path, full_data)

    # --- обработка канала -----------------------------------------------
    def _process_channel(self, ch_idx: int, audio, total_sec: float) -> List[Dict]:
        """Диаризация + посегментная ASR одного канала. Возвращает список реплик."""
        role = CHANNEL_ROLES.get(ch_idx, f"channel_{ch_idx}")
        logger.info(f"Диаризация канала {ch_idx} ({role})...")

        raw = self.diarizer.diarize_channel(audio, ch_idx)
        spans = merge_segments([(x["start"], x["end"]) for x in raw])
        long_spans = sum(1 for s, e in spans if (e - s) + 2 * PAD_SEC > LONGFORM_THRESHOLD_SEC)
        logger.info(
            f"Канал {ch_idx}: {len(raw)} сегментов Pyannote -> {len(spans)} после слияния "
            f"(длинных, пойдут в longform: {long_spans})."
        )

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
                if asr_res["mode"] == "longform":
                    flags.append("longform")
                if asr_res["error"]:
                    flags.append("asr_error")

            # Пустой результат без ошибки пропускаем, а упавший сегмент оставляем в
            # dialogue.json с флагом asr_error, чтобы потери были видны.
            if not text and "asr_error" not in flags:
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

    # --- обработка файла ------------------------------------------------
    @log_execution_time(logger)
    def process_file(self, file_path: Path) -> Optional[Dict]:
        """
        Обрабатывает один файл: декодирование -> диаризация -> ASR -> JSON.
        Возвращает метаданные результата при успехе и None при исключении
        (исключение пишется в failed_files.json, файл будет повторён при следующем запуске).
        """
        stem = file_path.stem
        logger.info(f"Начало обработки: {file_path.name}")
        start_time = time.time()

        try:
            analysis = self.audio_proc.analyze_and_fix(str(file_path))
            channels_data = analysis["channels_data"]
            total_sec = len(channels_data[0]) / SR

            full_transcript = {
                "source_file": file_path.name,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": PIPELINE_VERSION,
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

                speech_texts = [
                    u["text"] for u in utts
                    if "short_sound" not in u["flags"] and u["text"]
                ]
                channel_words = [w for u in utts for w in u["words"]]
                full_transcript["channels"][role] = {
                    "speech_sec": round(sum(u["duration"] for u in utts), 3),
                    "segments_count": len(utts),
                    "short_sounds": sum("short_sound" in u["flags"] for u in utts),
                    "longform_segments": sum("longform" in u["flags"] for u in utts),
                    "asr_errors": sum("asr_error" in u["flags"] for u in utts),
                    "full_text": " ".join(speech_texts),
                    "words": channel_words,
                }
                all_utterances.extend(utts)

            all_utterances.sort(key=lambda u: u["start"])

            out_dir = DATA_OUT / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            tr_path = out_dir / f"{stem}_transcript.json"

            # Порядок записи важен: transcript.json - маркер "файл готов", поэтому
            # старый удаляем до записи, а новый пишем последним. Если процесс оборвётся
            # на середине, файл не будет считаться обработанным и повторится при перезапуске.
            tr_path.unlink(missing_ok=True)
            self._save_dialogue_json(all_utterances, out_dir / f"{stem}_dialogue.json")
            self._save_readable_json(
                file_path.name, total_sec, all_utterances, out_dir / f"{stem}_readable.json"
            )
            self._save_transcript_json(full_transcript, tr_path)

            channels = full_transcript["channels"].values()
            total_errors = sum(c["asr_errors"] for c in channels)
            meta = {
                "source_file": file_path.name,
                "processed_at": full_transcript["processed_at"],
                "duration_sec": round(total_sec, 3),
                "processing_sec": round(time.time() - start_time, 1),
                "asr_errors": total_errors,
                "longform_segments": sum(c["longform_segments"] for c in channels),
                "pipeline_version": PIPELINE_VERSION,
            }
            self.registry.mark_done(stem, meta)
            self.failures.clear(stem)

            if total_errors:
                logger.warning(f"⚠️ {file_path.name}: сегментов с ошибкой ASR: {total_errors} "
                               f"(можно повторить: --retry-errors)")
            logger.info(f"✅ Успешно: {file_path.name} за {meta['processing_sec']:.2f} сек.")
            return meta

        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {file_path.name}: {e}", exc_info=True)
            self.failures.record(stem, f"{type(e).__name__}: {e}")
            return None

    def run_batch(self, files: List[Path]) -> Dict:
        """Обрабатывает переданный список файлов и печатает итоговую статистику."""
        total = len(files)
        ok = 0
        failed: List[str] = []
        with_errors = 0
        total_duration = 0.0
        total_processing = 0.0
        batch_start = time.time()

        for i, f in enumerate(files, 1):
            t_start = time.time()
            meta = self.process_file(f)
            elapsed = time.time() - t_start
            if meta is None:
                failed.append(f.name)
                continue
            ok += 1
            if meta["asr_errors"]:
                with_errors += 1
            dur = meta["duration_sec"]
            total_duration += dur
            total_processing += elapsed
            speed = dur / elapsed if elapsed > 0 else 0
            logger.info(f"[{i}/{total}] {f.name} | {dur:.1f}с | {elapsed:.1f}с | {speed:.2f}x")

        avg_speed = total_duration / total_processing if total_processing > 0 else 0
        logger.info("=" * 60)
        logger.info("🏁 ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
        logger.info(f"Успешно: {ok}/{total}")
        logger.info(f"С ошибками ASR в сегментах: {with_errors} (повтор: --retry-errors)")
        logger.info(f"Упало с исключением: {len(failed)} (см. failed_files.json)")
        for name in failed[:20]:
            logger.info(f"   - {name}")
        logger.info(f"Общее время: {time.time() - batch_start:.1f} сек")
        logger.info(f"Суммарная длительность аудио: {total_duration:.1f} сек")
        logger.info(f"Средняя скорость: {avg_speed:.2f}x realtime")
        logger.info("=" * 60)
        return {"ok": ok, "failed": failed, "with_errors": with_errors}
