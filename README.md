# Rehearsal Track Splitter

Split a long rehearsal `.wav` into likely **song** segments and export each as:

- `track01.mp3`
- `track02.mp3`
- ...

The script tries to filter out talking/noodling using audio heuristics (loudness, harmonic content, and musical onset activity) plus minimum-duration filtering.

## Requirements

- Python 3.10+
- `ffmpeg` in PATH (required for MP3 export)

Install Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python separate_rehearsal_tracks.py rehearsal.wav --out-dir tracks
```

Outputs are created in `tracks/` as `track01.mp3`, `track02.mp3`, etc.

## Useful tuning flags

- `--min-song-sec 120` minimum segment duration to keep
- `--score-threshold 0.56` higher = stricter song detection
- `--max-bridge-gap-sec 25` merges nearby song chunks across short pauses
- `--pad-sec 4` adds context at segment boundaries

Example for stricter filtering:

```bash
python separate_rehearsal_tracks.py rehearsal.wav \
  --out-dir tracks \
  --score-threshold 0.62 \
  --min-song-sec 150
```

Example for catching shorter songs/jams:

```bash
python separate_rehearsal_tracks.py rehearsal.wav \
  --out-dir tracks \
  --score-threshold 0.50 \
  --min-song-sec 75
```

## Notes

- This is heuristic-based, not source separation. It separates **sections in time** from one rehearsal recording.
- If no tracks are found, lower `--score-threshold` and/or `--min-song-sec`.
