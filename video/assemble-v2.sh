#!/usr/bin/env bash
# Assemble Gatekeeper demo video v2: v1 segments, but S3 is rebuilt from the
# real CDP screencast frames (record/record.mjs) — frames held for their
# recorded durations, with the unchanged S3 narration.
set -euo pipefail
cd "$(dirname "$0")"
FPS=30
SEGDIR=segments
OUTDIR=out
mkdir -p "$SEGDIR" "$OUTDIR"

# --- S3 from screencast frames ---
python3 - <<'PY'
import json, pathlib
meta = json.load(open("record/frames-meta.json"))
times = meta["times"]
frames = sorted(pathlib.Path("record/frames").glob("frame-*.jpg"))
assert len(frames) == len(times), (len(frames), len(times))
TOTAL = 29.5  # matches v1 S3 segment (28.752s narration + 0.7s pad)
lines = []
for i, (f, t) in enumerate(zip(frames, times)):
    nxt = times[i + 1] if i + 1 < len(times) else TOTAL
    dur = max(nxt - t, 0.05)
    lines.append(f"file '{f.resolve()}'")
    lines.append(f"duration {dur:.3f}")
# concat demuxer: repeat the last file once without a duration
lines.append(f"file '{frames[-1].resolve()}'")
pathlib.Path("record/frames-list.txt").write_text("\n".join(lines) + "\n")
print(f"S3: {len(frames)} screencast frames -> {TOTAL}s")
PY

ffmpeg -y -v error -f concat -safe 0 -i record/frames-list.txt -i n3.mp3 \
  -filter_complex "[0:v]fps=$FPS,scale=1920:1080,format=yuv420p[v];[1:a]apad=whole_dur=29.5[a]" \
  -map "[v]" -map "[a]" -t 29.5 -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k \
  "$SEGDIR/seg3-live.mp4"

# --- v2 concat: v1 segments 1,2,4,5,6 + new live S3 ---
printf "file '%s'\n" seg1.mp4 seg2.mp4 seg3-live.mp4 seg4.mp4 seg5.mp4 seg6.mp4 > "$SEGDIR/list-v2.txt"
ffmpeg -y -v error -f concat -safe 0 -i "$SEGDIR/list-v2.txt" -c copy "$OUTDIR/gatekeeper-demo-v2.mp4"
TOTAL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTDIR/gatekeeper-demo-v2.mp4")
echo "FINAL: $OUTDIR/gatekeeper-demo-v2.mp4  duration ${TOTAL}s"
