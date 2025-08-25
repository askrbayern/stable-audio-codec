#!/usr/bin/env bash
set -e

if [ $# -lt 5 ]; then
    echo "Usage: $0 <config_name> <lambda> <wandb_project> <wandb_run_name> <continue|new|activation>"
    exit 1
fi

CONFIG_NAME=$1
LAMBDA=$2
WANDB_PROJECT=$3
WANDB_RUN_NAME=$4
RESUME_MODE=$5

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Split CONFIG_NAME by "-" and construct path
IFS='-' read -ra CONFIG_PARTS <<< "$CONFIG_NAME"
CONFIG_PATH=""
for part in "${CONFIG_PARTS[@]:0:${#CONFIG_PARTS[@]}-1}"; do
    CONFIG_PATH="${CONFIG_PATH}${part}/"
done
CONFIG_FILE="${CONFIG_PARTS[-1]}_config.json"
MODEL_CONFIG=${BASE_DIR}/stable_audio_tools/configs/model_configs/test/${CONFIG_PATH}${CONFIG_FILE}

# Update lm_weight in config file
if [ "$LAMBDA" != "-1" ]; then
    python -c "
import json
with open('$MODEL_CONFIG', 'r') as f:
    config = json.load(f)
config['model']['lm_weight'] = float($LAMBDA)
with open('$MODEL_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
"
fi

DATA_DIR=/itet-stor/feigao/disco-computing/music_datasets/jamendo/raw_30s
# SMALL_DATA_DIR=/itet-stor/feigao/disco-computing/music_datasets/jamendo/raw_30s/00
INPUT_DIR=/itet-stor/feigao/home/stable-audio-tools/test_folder/
OUTPUT_DIR=/itet-stor/feigao/net_scratch/outputs/${WANDB_PROJECT}/${WANDB_RUN_NAME}
CHECKPOINT_PATH=${OUTPUT_DIR}/checkpoints/last.ckpt

NUM_GPUS=2
BATCH_SIZE=4
NUM_WORKERS=8
MAX_EPOCHS=300

echo "Model config: $MODEL_CONFIG"
echo "Lambda (lm_weight): $LAMBDA"
echo "Project: $WANDB_PROJECT"
echo "Run name: $WANDB_RUN_NAME"
echo "Resume mode: $RESUME_MODE"

if [ "$RESUME_MODE" = "continue" ]; then
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "Checkpoint not found at $CHECKPOINT_PATH. Cannot continue."
        exit 1
    fi
    python train_start.py \
      --model-config $MODEL_CONFIG \
      --data-dir $DATA_DIR \
      --input-dir $INPUT_DIR \
      --output-dir $OUTPUT_DIR \
      --batch-size $BATCH_SIZE \
      --num-workers $NUM_WORKERS \
      --max-epochs $MAX_EPOCHS \
      --wandb-project $WANDB_PROJECT \
      --wandb-run-name $WANDB_RUN_NAME \
      --accelerator gpu \
      --devices $NUM_GPUS \
      --ckpt-path $CHECKPOINT_PATH
elif [ "$RESUME_MODE" = "new" ]; then
    python train_start.py \
      --model-config $MODEL_CONFIG \
      --data-dir $DATA_DIR \
      --input-dir $INPUT_DIR \
      --output-dir $OUTPUT_DIR \
      --batch-size $BATCH_SIZE \
      --num-workers $NUM_WORKERS \
      --max-epochs $MAX_EPOCHS \
      --wandb-project $WANDB_PROJECT \
      --wandb-run-name $WANDB_RUN_NAME \
      --accelerator gpu \
      --devices $NUM_GPUS
elif [ "$RESUME_MODE" = "inspect" ]; then
    echo "Entering inspect mode: extracting latents"
    # Run inspect in train_start.py
    python train_start.py \
      --model-config $MODEL_CONFIG \
      --data-dir $DATA_DIR \
      --input-dir $DATA_DIR \
      --output-dir $OUTPUT_DIR \
      --batch-size 1 \
      --num-workers 1 \
      --max-epochs 1 \
      --accelerator cpu \
      --devices 0 \
      --inspect \
      --inspect-count 5 \
      --ckpt-path $CHECKPOINT_PATH
    exit 0
elif [ "$RESUME_MODE" = "activation" ]; then
    echo "Activation mode: train 1 epoch, eval, save, reload, eval again"
    python train_start_activation.py \
      --model-config $MODEL_CONFIG \
      --data-dir $DATA_DIR \
      --input-dir $INPUT_DIR \
      --output-dir $OUTPUT_DIR \
      --batch-size $BATCH_SIZE \
      --num-workers $NUM_WORKERS \
      --max-epochs 5 \
      --wandb-project $WANDB_PROJECT \
      --wandb-run-name $WANDB_RUN_NAME \
      --accelerator gpu \
      --devices $NUM_GPUS

else
    echo "Unknown resume mode: $RESUME_MODE. Use 'continue' or 'new'."
    exit 1
fi