#!/usr/bin/env bash
# Assemble the Gatekeeper demo video v1: slides + per-slide narration.
set -euo pipefail
cd "$(dirname "$0")"
FPS=30
SEGDIR=segments
mkdir -p "$SEGDIR" out

for i in 1 2 3 4 5 6; do
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "n$i.mp3")
  SEG=$(python3 -c "print(max(float('$DUR') + 0.7, 3.0))")
  FRAMES=$(python3 -c "print(int(float('$SEG') * $FPS))")
  echo "segment $i: audio ${DUR}s -> ${SEG}s (${FRAMES} frames)"
  ffmpeg -y -v error -loop 1 -framerate $FPS -i "slides/s$i.png" -i "n$i.mp3" \
    -filter_complex "[0:v]zoompan=z='min(1+0.12*on/$FRAMES,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=$FRAMES:s=1920x1080:fps=$FPS[v];[1:a]apad=whole_dur=$SEG[a]" \
    -map "[v]" -map "[a]" -t "$SEG" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k \
    "$SEGDIR/seg$i.mp4"
done

printf "file '%s'\n" seg1.mp4 seg2.mp4 seg3.mp4 seg4.mp4 seg5.mp4 seg6.mp4 > "$SEGDIR/list.txt"
ffmpeg -y -v error -f concat -safe 0 -i "$SEGDIR/list.txt" -c copy out/gatekeeper-demo-v1.mp4
TOTAL=$(ffprobe -v error -show_entries format=duration -of csv=p=0 out/gatekeeper-demo-v1.mp4)
echo "FINAL: out/gatekeeper-demo-v1.mp4  duration ${TOTAL}s"
