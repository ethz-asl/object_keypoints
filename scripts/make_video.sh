#!/bin/bash
set -e

ffmpeg -i "$1/%06d.jpg" -r 60 -y -c:v libx264 -vf scale=1280:360 -crf 25 "$1/out.mp4"

