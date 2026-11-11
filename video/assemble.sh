#!/usr/bin/env bash
# Build the shared slide segments (1, 2, 4, 5, 6) used by assemble-v2.sh.
# v1 (all slides, incl. static S3) is retired; S3 in v2 is the live recording.
set -euo pipefail
cd "$(dirname "$0")"
FPS=30
SEGDIR=segments
mkdir -p "$SEGDIR" out

# S3 is no longer a static slide: v2 rebuilds it from the live screen
# recording (see assemble-v2.sh), so only segments 1, 2, 4, 5, 6 are built here.
for i in 1 2 4 5 6; do
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "n$i.mp3")
  SEG=$(python3 -c "print(max(float('$DUR') + 0.7, 3.0))")
  FRAMES=$(python3 -c "print(int(float('$SEG') * $FPS))")
  echo "segment $i: audio ${DUR}s -> ${SEG}s (${FRAMES} frames)"
  ffmpeg -y -v error -loop 1 -framerate $FPS -i "slides/s$i.png" -i "n$i.mp3" \
    -filter_complex "[0:v]zoompan=z='min(1+0.12*on/$FRAMES,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=$FRAMES:s=1920x1080:fps=$FPS[v];[1:a]apad=whole_dur=$SEG[a]" \
    -map "[v]" -map "[a]" -t "$SEG" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k \
    "$SEGDIR/seg$i.mp4"
done

echo "Segments ready in $SEGDIR/ — run assemble-v2.sh for the final video."
