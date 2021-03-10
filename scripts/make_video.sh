#!/bin/bash
set -e

ffmpeg -i '/tmp/frames/%06d.jpg' -r 60 -y -c:v libx264 -vf scale=1280:360 -crf 25 /tmp/frames/out.mp4

