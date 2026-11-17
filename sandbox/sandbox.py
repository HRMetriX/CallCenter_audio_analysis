# sandbox/sandbox.py
"""
Проба аудиозаписи колл-центра перед выбором инструментов ASR/диаризации.

Зависимости: Python 3.10+, ffmpeg+ffprobe в PATH, numpy.
Запуск:
    poetry run python sandbox/sandbox.py data/in/your_call.ogg
Скрипт ничего не записывает на диск — только читает и печатает отчёт.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], binary: bool = False):
    p = subprocess.run(cmd, capture_output=True, text=not binary, timeout=600)
    return p.returncode, p.stdout, p.stderr


def ffprobe_meta(path: Path) -> dict:
    rc, out, err = _run([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ])
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()}")
    data = json.loads(out)
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    if not audio_streams:
        raise RuntimeError("Аудиопоток не найден.")
    a = audio_streams[0]
    return {
        "container": fmt.get("format_name"),
        "duration_sec": float(fmt.get("duration", a.get("duration", 0)) or 0),
        "bitrate_bps": int(fmt.get("bit_rate", 0) or 0),
        "audio_streams_total": len(audio_streams),
        "audio": {
            "codec": a.get("codec_name"),
            "channels": int(a.get("channels", 0)),
            "channel_layout": a.get("channel_layout"),
            "sample_rate_hz": int(a.get("sample_rate", 0)),
            "bits_per_sample": int(a.get("bits_per_raw_sample", a.get("bits_per_sample", 0)) or 0),
            "avg_bitrate_bps": int(a.get("bit_rate", 0) or 0),
        },
    }


def decode_to_numpy(path: Path, channels: int, sample_rate: int):
    import numpy as np
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-",
    ]
    rc, raw, err = _run(cmd, binary=True)
    if rc != 0 or not raw:
        raise RuntimeError(f"ffmpeg decode failed: {err.decode('utf-8', 'replace').strip()}")
    audio = np.frombuffer(raw, dtype="<f4")
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio


def analyze_signal(audio, channels: int, sample_rate: int) -> dict:
    import numpy as np
    eps = 1e-12
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    n = audio.shape[0]
    stats: dict = {"frames": int(n), "seconds": round(n / sample_rate, 3)}

    def ch_stats(x) -> dict:
        x64 = x.astype(np.float64)
        rms = math.sqrt(float(np.mean(x64 ** 2)) + eps)
        peak = float(np.max(np.abs(x64))) + eps
        dc = float(np.mean(x64))
        silent = float(np.mean(np.abs(x64) < 0.01))  # порог ~ -40 dBFS
        return {
            "rms_dbfs": round(20 * math.log10(rms), 2),
            "peak_dbfs": round(20 * math.log10(peak), 2),
            "dc_offset": round(dc, 6),
            "silence_ratio": round(silent, 4),
            "crest_factor_db": round(20 * math.log10(peak / rms), 2),
        }

    stats["per_channel"] = [ch_stats(audio[:, c]) for c in range(channels)]
    if channels == 2:
        l = audio[:, 0].astype(np.float64)
        r = audio[:, 1].astype(np.float64)
        corr = float(np.corrcoef(l, r)[0, 1])
        stats["stereo_correlation"] = round(corr, 4)
        diff_rms = math.sqrt(float(np.mean((l - r) ** 2)) + eps)
        stats["diff_rms_dbfs"] = round(20 * math.log10(diff_rms), 2)
    return stats


def diagnose(meta: dict, sig: dict) -> list[str]:
    notes: list[str] = []
    a = meta["audio"]
    sr, ch = a["sample_rate_hz"], a["channels"]

    # 1. Полоса
    if sr <= 8000:
        notes.append(
            f"sample_rate={sr} Гц → ТЕЛЕФОННАЯ узкополосная запись. Whisper large-v3 (нативно 16 кГц) "
            "потеряет 5..15% WER vs широкополосная речь. Обязателен ресамплинг до 16 кГц. "
            "Рассмотрите NeMo stt_ru_conformer_ctc_large (обучен на телефонном русском) как альтернативу/контроль."
        )
    elif sr < 16000:
        notes.append(f"sample_rate={sr} Гц → узкая полоса. Ресамплинг до 16 кГц.")
    elif sr == 16000:
        notes.append(f"sample_rate={sr} Гц → широкополосная, оптимально для Whisper.")
    else:
        notes.append(
            f"sample_rate={sr} Гц → полная полоса. Whisper всё равно даунсемплит до 16 кГц; "
            "можно даунсемплить заранее для экономии I/O без потери точности."
        )

    # 2. Каналы → стратегия диаризации
    if ch == 1:
        notes.append(
            "mono → разделить спикеров по каналам НЕВОЗМОЖНО. "
            "Диаризация только через pyannote.audio (speaker-diarization-3.1)."
        )
    elif ch == 2:
        corr = sig.get("stereo_correlation")
        if corr is None:
            notes.append("Стерео, корреляция не вычислена.")
        elif abs(corr) < 0.3:
            notes.append(
                f"stereo, correlation={corr} → НИЗКАЯ. Почти наверняка ДВУХКАНАЛЬНАЯ запись "
                "(оператор в одном канале, клиент в другом). Диаризация = разделение каналов, "
                "pyannote НЕ нужен. ASR по каждому каналу отдельно — это и самый точный, и самый дешёвый путь."
            )
        elif abs(corr) > 0.7:
            notes.append(
                f"stereo, correlation={corr} → ВЫСОКАЯ. Это стерео-микс одного контента "
                "(оба спикера в обоих каналах). Диаризация только через pyannote.audio."
            )
        else:
            notes.append(
                f"stereo, correlation={corr} → ПРОМЕЖУТОЧНАЯ. Послушайте запись; "
                "вероятно потребуется pyannote.audio."
            )
        l, r = sig["per_channel"]
        lr_diff = abs(l["rms_dbfs"] - r["rms_dbfs"])
        if lr_diff > 20:
            notes.append(
                f"Разница уровней каналов {lr_diff:.0f} дБ → один канал почти пустой. "
                "Возможно записана только одна нога диалога или брак. Проверьте вручную."
            )
    else:
        notes.append(f"channels={ch} → нестандартная схема для колл-центра; проверьте вручную.")

    # 3. Тишина / плотность речи
    sil = max(pc["silence_ratio"] for pc in sig["per_channel"])
    if sil > 0.5:
        notes.append(
            f"silence_ratio={sil:.2f} > 0.5 → много тишины. Обязателен VAD (Silero VAD) перед ASR, "
            "иначе Whisper галлюцинирует на паузах («Спасибо за просмотр», «Подпишитесь» и т.п.)."
        )
    else:
        notes.append(f"silence_ratio={sil:.2f} → нормальная плотность. VAD всё равно рекомендуется.")

    notes.append(
        f"Контейнер={meta.get('container')}, кодек={a['codec']}. ffmpeg декодирует прозрачно; "
        "на конвейер подаём 16 кГц WAV/PCM (mono или stereo — по решению выше)."
    )
    return notes


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python sandbox.py <audio_file>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return 1

    print(f"=== Проба: {path} ===\n")
    meta = ffprobe_meta(path)
    print("[Метаданные ffprobe]")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    a = meta["audio"]
    audio = decode_to_numpy(path, a["channels"], a["sample_rate_hz"])
    sig = analyze_signal(audio, a["channels"], a["sample_rate_hz"])
    print("\n[Сигнальные характеристики]")
    print(json.dumps(sig, indent=2, ensure_ascii=False))

    print("\n[Диагноз и рекомендации по выбору инструментов]")
    for i, note in enumerate(diagnose(meta, sig), 1):
        print(f"  {i}. {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))