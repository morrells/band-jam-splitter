#!/usr/bin/env python3
"""Split a rehearsal WAV into likely full-song MP3 tracks.

This script uses simple audio heuristics to detect sections that are likely songs:
- higher loudness than baseline (RMS dB)
- higher harmonic-vs-percussive ratio
- moderate/high onset strength (musical activity)
- minimum contiguous duration

Output files are named track01.mp3, track02.mp3, ... in the chosen output folder.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass
class Segment:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def sec_to_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same")


def build_song_mask(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    smooth_sec: float,
    score_threshold: float,
    loudness_floor_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-10

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + eps, ref=np.max)

    y_harm, y_perc = librosa.effects.hpss(y)
    rms_h = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_length)[0]
    rms_p = librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0]
    harm_ratio = (rms_h + eps) / (rms_h + rms_p + eps)

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset = onset[: len(rms)] if len(onset) >= len(rms) else np.pad(onset, (0, len(rms) - len(onset)))

    def robust_norm(v: np.ndarray) -> np.ndarray:
        p10, p90 = np.percentile(v, [10, 90])
        if p90 - p10 < 1e-6:
            return np.zeros_like(v)
        return np.clip((v - p10) / (p90 - p10), 0.0, 1.0)

    loudness_norm = robust_norm(rms_db)
    harm_norm = robust_norm(harm_ratio)
    onset_norm = robust_norm(onset)

    score = 0.50 * loudness_norm + 0.35 * harm_norm + 0.15 * onset_norm

    frames_per_sec = sr / float(hop_length)
    smooth_win = max(1, int(round(smooth_sec * frames_per_sec)))
    smooth_score = moving_average(score, smooth_win)

    mask = (smooth_score >= score_threshold) & (rms_db >= loudness_floor_db)
    times = librosa.frames_to_time(np.arange(len(mask)), sr=sr, hop_length=hop_length)
    return mask, times


def mask_to_segments(
    mask: np.ndarray,
    times: np.ndarray,
    min_song_sec: float,
    max_bridge_gap_sec: float,
    pad_sec: float,
    audio_duration_sec: float,
) -> list[Segment]:
    if not np.any(mask):
        return []

    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    raw: list[Segment] = []
    for s_idx, e_idx in zip(starts, ends):
        start = float(times[min(s_idx, len(times) - 1)])
        end = float(times[min(max(e_idx - 1, 0), len(times) - 1)])
        if end > start:
            raw.append(Segment(start, end))

    if not raw:
        return []

    merged: list[Segment] = [raw[0]]
    for seg in raw[1:]:
        prev = merged[-1]
        gap = seg.start_sec - prev.end_sec
        if gap <= max_bridge_gap_sec:
            merged[-1] = Segment(prev.start_sec, max(prev.end_sec, seg.end_sec))
        else:
            merged.append(seg)

    kept = [s for s in merged if s.duration_sec >= min_song_sec]

    padded: list[Segment] = []
    for seg in kept:
        s = max(0.0, seg.start_sec - pad_sec)
        e = min(audio_duration_sec, seg.end_sec + pad_sec)
        if e > s:
            padded.append(Segment(s, e))
    return padded


def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def export_mp3_segments(
    y: np.ndarray,
    sr: int,
    segments: list[Segment],
    out_dir: Path,
    bitrate: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(segments, start=1):
        s = int(math.floor(seg.start_sec * sr))
        e = int(math.ceil(seg.end_sec * sr))
        chunk = y[s:e]
        if chunk.size == 0:
            continue

        out_path = out_dir / f"track{idx:02d}.mp3"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)

        try:
            sf.write(str(tmp_wav), chunk, sr)
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_wav),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                str(out_path),
            ]
            subprocess.run(cmd, check=True)
            print(f"Wrote {out_path} ({sec_to_hms(seg.duration_sec)})")
        finally:
            if tmp_wav.exists():
                tmp_wav.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split rehearsal WAV into likely song tracks.")
    p.add_argument("input_wav", type=Path, help="Input .wav rehearsal recording")
    p.add_argument("--out-dir", type=Path, default=Path("tracks"), help="Output directory (default: tracks)")
    p.add_argument("--sr", type=int, default=22050, help="Analysis sample rate (default: 22050)")
    p.add_argument("--frame-length", type=int, default=4096, help="Analysis frame length")
    p.add_argument("--hop-length", type=int, default=1024, help="Analysis hop length")
    p.add_argument("--smooth-sec", type=float, default=4.0, help="Score smoothing window in seconds")
    p.add_argument("--score-threshold", type=float, default=0.56, help="Music score threshold (0-1)")
    p.add_argument("--loudness-floor-db", type=float, default=-42.0, help="Minimum RMS dB relative to peak")
    p.add_argument("--min-song-sec", type=float, default=120.0, help="Minimum segment length to keep")
    p.add_argument("--max-bridge-gap-sec", type=float, default=25.0, help="Merge adjacent segments if gap <= this")
    p.add_argument("--pad-sec", type=float, default=4.0, help="Pad each segment start/end by this many seconds")
    p.add_argument("--bitrate", default="192k", help="MP3 bitrate, e.g. 128k/192k/256k")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_wav.exists():
        raise SystemExit(f"Input file not found: {args.input_wav}")
    if args.input_wav.suffix.lower() != ".wav":
        raise SystemExit("Input must be a .wav file")
    if not has_ffmpeg():
        raise SystemExit("ffmpeg is required for MP3 export. Install ffmpeg and retry.")

    print(f"Loading {args.input_wav} ...")
    y, sr = librosa.load(str(args.input_wav), sr=args.sr, mono=True)
    duration = len(y) / float(sr)
    print(f"Loaded {sec_to_hms(duration)} at {sr} Hz")

    mask, times = build_song_mask(
        y=y,
        sr=sr,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        smooth_sec=args.smooth_sec,
        score_threshold=args.score_threshold,
        loudness_floor_db=args.loudness_floor_db,
    )

    segments = mask_to_segments(
        mask=mask,
        times=times,
        min_song_sec=args.min_song_sec,
        max_bridge_gap_sec=args.max_bridge_gap_sec,
        pad_sec=args.pad_sec,
        audio_duration_sec=duration,
    )

    if not segments:
        print("No likely songs found with current thresholds.")
        print("Try lowering --score-threshold and/or --min-song-sec.")
        return 2

    print("Detected song sections:")
    for i, s in enumerate(segments, start=1):
        print(
            f"  {i:02d}. {sec_to_hms(s.start_sec)} -> {sec_to_hms(s.end_sec)} "
            f"({sec_to_hms(s.duration_sec)})"
        )

    export_mp3_segments(y=y, sr=sr, segments=segments, out_dir=args.out_dir, bitrate=args.bitrate)
    print(f"Done. Exported {len(segments)} tracks to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
