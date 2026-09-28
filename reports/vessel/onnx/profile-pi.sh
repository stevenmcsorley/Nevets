#!/bin/sh
set -eu
cd /home/ubuntu/nbench
IMAGE=$(docker inspect channel-watch-nevets --format '{{.Config.Image}}')
: > profile-results.jsonl
for model in batch.fp32.onnx batch.onnx batch.fullint8.onnx; do
  for threads in 1 2; do
    for batch in 1 4; do
      for format in prose compact; do
        docker run --rm --cpus=2 --memory=900m --network=none -v "$PWD:/bench" -w /bench --entrypoint node \
          -e MODEL="$model" -e THREADS="$threads" -e BATCH="$batch" -e FORMAT="$format" "$IMAGE" bench-profile.mjs >> profile-results.jsonl
      done
    done
  done
done
