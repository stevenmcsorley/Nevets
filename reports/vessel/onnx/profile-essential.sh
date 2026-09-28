#!/bin/sh
set -eu
cd /home/ubuntu/nbench
IMAGE=$(docker inspect channel-watch-nevets --format '{{.Config.Image}}')
for model in batch.fp32.onnx batch.fullint8.onnx; do
  for batch in 1 2 4; do
    docker run --rm --cpus=2 --memory=900m --network=none -v "$PWD:/bench" -w /bench --entrypoint node \
      -e MODEL="$model" -e THREADS=2 -e BATCH="$batch" -e FORMAT=essential "$IMAGE" bench-profile.mjs >> profile-essential.jsonl
  done
done
