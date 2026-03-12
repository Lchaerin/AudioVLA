#!/usr/bin/env bash
# Generate simulated binaural episode data for Audio-VLA training
set -e

OUTPUT_DIR="${1:-data/sim_episodes}"
NUM_EPISODES="${2:-5000}"
AUDIO_DIR="${3:-}"  # optional: path to AudioSet clip directory

echo "Generating $NUM_EPISODES simulation episodes → $OUTPUT_DIR"

AUDIO_ARG=""
if [ -n "$AUDIO_DIR" ]; then
    AUDIO_ARG="--audio_dir $AUDIO_DIR"
fi

python data/sim_generator/episode_collector.py \
    --output_dir "$OUTPUT_DIR" \
    --num_episodes "$NUM_EPISODES" \
    --audio_sr 24000 \
    --audio_duration 5.0 \
    --img_size 512 \
    $AUDIO_ARG

echo "Done. Episodes saved to $OUTPUT_DIR"
