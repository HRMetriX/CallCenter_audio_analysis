# sandbox/sandbox_gigaam.py
"""
Тестовый прогон GigaAM-v3 на записи колл-центра.

Конвейер:
  <in>.ogg (8 кГц stereo vorbis)
    → ffmpeg: split L/R + resample → 16 кГц mono WAV     (artifact: <stem>_ch0.wav, _ch1.wav)
    → Silero VAD → речь-сегменты по каналам               (artifact: <stem>_vad.json)
    → нарезка сегментов в WAV                             (artifact: <stem>_segments/*.wav)
    → GigaAM v3_e2e_rnnt (word_timestamps=True)           (artifact: <stem>_transcript.json)
    → текст по каналам                                    (artifact: <stem>_text.json)

Запуск:
  poetry run python sandbox/sandbox_gigaam.py data/in/demo_1.ogg
"""
from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import sys
import time
from pathlib import Path

ASR_MODEL = "v3_e2e_rnnt"  # ← ОБНОВЛЕНО: v3 с пунктуацией и нормализацией
TARGET_SR = 16000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = PROJECT_ROOT / "data" / "out"


def fail(msg: str) -> None:
    print(f"ОШИБКА: {msg}", file=sys.stderr)
    sys.exit(1)


def run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}")


def split_channels(in_path: Path, stem: str, out_dir: Path) -> dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ch in (0, 1):
        out = out_dir / f"{stem}_ch{ch}.wav"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(in_path),
            "-af", f"pan=mono|c0=c{ch}",
            "-ar", str(TARGET_SR), "-ac", "1",
            str(out),
        ]
        run_ffmpeg(cmd)
        print(f"  [ffmpeg] channel {ch} → {out.name}")
        paths[ch] = out
    return paths


def vad_channel(wav_path: Path, vad_model) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    tensor = torch.from_numpy(audio.astype(np.float32))
    return get_speech_timestamps(
        tensor, vad_model, return_seconds=True,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
    )


def save_segments(wav_path: Path, stamps: list[dict], seg_dir: Path,
                  ch: int, stem: str) -> list[Path]:
    import soundfile as sf
    seg_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    paths = []
    for i, st in enumerate(stamps):
        s = int(st["start"] * sr)
        e = int(st["end"] * sr)
        out = seg_dir / f"{stem}_ch{ch}_seg_{i:03d}.wav"
        sf.write(str(out), audio[s:e], sr)
        paths.append(out)
    return paths


def parse_asr_result(res) -> tuple[str, list[dict]]:
    """Парсим результат ASR. GigaAM v3 возвращает объект с text и words."""
    text, words = "", []

    if res is None:
        return text, words

    # Если результат - строка (старые версии)
    if isinstance(res, str):
        text = res.strip()
        return text, words

    # GigaAM v3 возвращает объект с атрибутами
    text = getattr(res, "text", "") or ""

    # Проверяем наличие word-level таймстампов
    if hasattr(res, "words") and res.words:
        for w in res.words:
            words.append({
                "word": getattr(w, "text", str(w)),
                "start": float(getattr(w, "start", 0.0)),
                "end": float(getattr(w, "end", 0.0)),
            })

    # Если текста нет, но есть слова - собираем из слов
    if not text and words:
        text = " ".join(w["word"] for w in words)

    return text.strip(), words


def transcribe_segments(seg_paths: list[Path], asr_model) -> list[dict]:
    """Транскрибируем сегменты через GigaAM v3 с word_timestamps."""
    out = []
    for i, p in enumerate(seg_paths):
        t0 = time.time()
        try:
            # ← ВОЗВРАЩАЕМ word_timestamps=True для v3 моделей
            res = asr_model.transcribe(str(p), word_timestamps=True)
            text, words = parse_asr_result(res)
        except Exception as e:
            text, words = "", []
            print(f"    ASR seg {i + 1}/{len(seg_paths)} ОШИБКА: {e}")

        dt = time.time() - t0
        preview = text.replace("\n", " ")[:60]
        print(f"    ASR seg {i + 1}/{len(seg_paths)} [{dt:.1f}s, {len(words)} слов] {preview!r}")
        out.append({"text": text, "words": words})
    return out


def process_channel(ch: int, ch_wav: Path, stem: str, out_dir: Path,
                    vad_model, asr_model) -> dict:
    import soundfile as sf
    print(f"\n=== Канал {ch} ({ch_wav.name}) ===")
    t0 = time.time()
    info = sf.info(str(ch_wav))
    dur = info.frames / info.samplerate
    stamps = vad_channel(ch_wav, vad_model)
    speech_sec = sum(s["end"] - s["start"] for s in stamps)
    print(f"  VAD: {len(stamps)} сегм, {speech_sec:.1f} c речи / {dur:.1f} c "
          f"({speech_sec / dur * 100:.0f}%)")

    seg_dir = out_dir / f"{stem}_segments"
    seg_paths = save_segments(ch_wav, stamps, seg_dir, ch, stem)
    print(f"  Нарезано {len(seg_paths)} WAV в {seg_dir.relative_to(out_dir)}/")

    asr_results = transcribe_segments(seg_paths, asr_model)

    segments = []
    for idx, (st, seg_wav, asr) in enumerate(zip(stamps, seg_paths, asr_results)):
        seg_start = float(st["start"])

        # Используем word-level таймстампы из GigaAM v3
        if asr["words"]:
            words_abs = [
                {"word": w["word"],
                 "start": round(seg_start + w["start"], 3),
                 "end": round(seg_start + w["end"], 3)}
                for w in asr["words"]
            ]
        else:
            # Заглушка, если слов нет
            words_abs = [{
                "word": asr["text"],
                "start": round(seg_start, 3),
                "end": round(float(st["end"]), 3)
            }] if asr["text"] else []

        segments.append({
            "index": idx,
            "start": round(seg_start, 3),
            "end": round(float(st["end"]), 3),
            "wav": seg_wav.name,
            "text": asr["text"],
            "words": words_abs,
        })

    full_text = " ".join(s["text"] for s in segments if s["text"]).strip()
    print(f"  Канал {ch} готов за {time.time() - t0:.1f} c: "
          f"{len(full_text)} симв., {len(segments)} сегм.")

    return {
        "audio_wav": ch_wav.name,
        "duration_sec": round(dur, 3),
        "speech_sec": round(speech_sec, 3),
        "segments_count": len(segments),
        "full_text": full_text,
        "segments": segments,
        "vad": [{"start": round(s["start"], 3), "end": round(s["end"], 3)} for s in stamps],
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python sandbox_gigaam.py <audio_file>", file=sys.stderr)
        return 2
    in_path = Path(argv[1])
    if not in_path.exists():
        print(f"Файл не найден: {in_path}", file=sys.stderr)
        return 1

    stem = in_path.stem
    out_dir = DATA_OUT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # чистим старые сегменты от предыдущих прогонов
    old_seg = out_dir / f"{stem}_segments"
    if old_seg.exists():
        shutil.rmtree(old_seg)

    print(f"=== GigaAM-v3 test: {in_path} ===")
    print(f"=== Output dir: {out_dir} ===\n")

    # [1] split + resample
    print("[1/4] ffmpeg: split channels + resample 16 kHz")
    ch_wavs = split_channels(in_path, stem, out_dir)

    # [2] load models
    print("\n[2/4] Загрузка моделей "
          "(первый запуск скачает GigaAM-v3 ~1 ГБ и Silero VAD)...")

    # ⚠️ ВРЕМЕННО ОТКЛЮЧАЕМ ПРОВЕРКУ SSL СЕРТИФИКАТОВ ⚠️
    print("  [SSL] Отключаем проверку SSL-сертификатов для загрузки моделей...")
    original_ssl_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    print("  [SSL] ✓ Проверка SSL отключена")

    try:
        import torch
        from silero_vad import load_silero_vad
        import gigaam
    except ImportError as e:
        fail(f"импорт не удался: {e}. Выполните: "
             f"poetry add gigaam silero-vad soundfile numpy torchaudio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  torch {torch.__version__} | device: {device}")
    t_load = time.time()
    vad_model = load_silero_vad()
    asr_model = gigaam.load_model(ASR_MODEL)
    print(f"  GigaAM model '{ASR_MODEL}' загружена за {time.time() - t_load:.1f} c")

    # ✅ ВОЗВРАЩАЕМ ОБРАТНО ПРОВЕРКУ SSL СЕРТИФИКАТОВ ✅
    ssl._create_default_https_context = original_ssl_context
    print("  [SSL] ✓ Проверка SSL-сертификатов восстановлена")

    # [3] process channels
    print("\n[3/4] Обработка каналов")
    channels = {}
    for ch in (0, 1):
        channels[f"channel_{ch}"] = process_channel(
            ch, ch_wavs[ch], stem, out_dir, vad_model, asr_model)

    # [4] artifacts
    print("\n[4/4] Сохранение артефактов")
    transcript = {
        "file": in_path.name,
        "model": f"gigaam/{ASR_MODEL}",
        "vad": "silero-vad",
        "target_sample_rate": TARGET_SR,
        "channels": channels,
    }
    tr_path = out_dir / f"{stem}_transcript.json"
    tr_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"  {tr_path.name}")

    text_path = out_dir / f"{stem}_text.json"
    text_path.write_text(json.dumps(
        {k: v["full_text"] for k, v in channels.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {text_path.name}")

    vad_path = out_dir / f"{stem}_vad.json"
    vad_path.write_text(json.dumps(
        {k: v["vad"] for k, v in channels.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {vad_path.name}")

    print("\n=== Готово ===")
    for k, v in channels.items():
        print(f"  {k}: {v['segments_count']} сегм, "
              f"{v['speech_sec']:.0f} c речи, {len(v['full_text'])} симв.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))