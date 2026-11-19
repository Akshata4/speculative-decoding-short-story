#!/usr/bin/env bash
# Build a narrated draft video from slides/png/*.png + video/script.md (macOS only:
# uses the built-in `say` text-to-speech, plus ffmpeg).
#
#   bash scripts/make_video.sh [voice] [words-per-minute]
#
# For the final submission, record your own voice instead (see README); this
# draft is useful for timing rehearsal or as a fallback.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOICE="${1:-Samantha}"
RATE="${2:-165}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# split script.md into one text file per slide
awk -v dir="$WORK" '
  /^## Slide [0-9]+/ { n = $3; f = sprintf("%s/s%02d.txt", dir, n); next }
  f { print > f }
' "$ROOT/video/script.md"

: > "$WORK/list.txt"
for png in "$ROOT"/slides/png/slide_*.png; do
  n="$(basename "$png" .png | sed 's/slide_//')"
  txt="$WORK/s$n.txt"
  say -v "$VOICE" -r "$RATE" -o "$WORK/a$n.aiff" -f "$txt"
  # 0.6 s of silence before and 1.0 s after each slide's narration
  ffmpeg -loglevel error -y -loop 1 -i "$png" -i "$WORK/a$n.aiff" \
    -af "adelay=600|600,apad=pad_dur=1" -shortest \
    -vf "scale=1920:1080,format=yuv420p" -r 30 -c:v libx264 -tune stillimage \
    -c:a aac -b:a 160k "$WORK/v$n.mp4"
  echo "file '$WORK/v$n.mp4'" >> "$WORK/list.txt"
  echo "slide $n done"
done

mkdir -p "$ROOT/video"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy \
  "$ROOT/video/speculative-decoding-narrated-draft.mp4"
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 \
  "$ROOT/video/speculative-decoding-narrated-draft.mp4" | awk '{printf "duration: %d min %d s\n", $1/60, $1%60}'
