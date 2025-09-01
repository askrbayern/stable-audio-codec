#!/bin/bash

export AUDIT_BALANCE=1
export AUDIT_BALANCE_BATCHES=20
export LOG_DATASET_TIMINGS=1

set -e

# Usage: train_server_dc.sh <wandb_project> <wandb_run_name> <continue|new>
if [ $# -lt 3 ]; then
  echo "Usage: $0 <wandb_project> <wandb_run_name> <continue|new>"
  exit 1
fi


WANDB_PROJECT=$1
WANDB_RUN_NAME=$2
RESUME_MODE=$3
RECON_DIR=/itet-stor/feigao/net_scratch/datasets/recon
OUTPUT_DIR=/itet-stor/feigao/net_scratch/outputs/${WANDB_PROJECT}/${WANDB_RUN_NAME}


MODEL_CONFIG=/itet-stor/feigao/home/stable-audio-tools/stable_audio_tools/configs/model_configs/test/fsq/size32768/sf128/regularLR/fsq2048_config.json
# DATA_CONFIG=/itet-stor/feigao/home/stable-audio-tools/stable_audio_tools/configs/dataset_configs/test_training.json
# DATA_CONFIG=/itet-stor/feigao/home/stable-audio-tools/stable_audio_tools/configs/dataset_configs/full_training_tikgpu10.json
DATA_CONFIG=/itet-stor/feigao/home/stable-audio-tools/stable_audio_tools/configs/dataset_configs/full_training.json
CHECKPOINT_PATH=${OUTPUT_DIR}/checkpoints/last.ckpt

BATCH_SIZE=30
NUM_WORKERS=8
MAX_EPOCHS=2000
NUM_GPUS=8
PRECISION="bf16-mixed"
CKPT_EVERY_EPOCHS=2
RECON_EVERY_EPOCHS=5


mkdir -p "$OUTPUT_DIR"

echo "Model config: $MODEL_CONFIG"
echo "Data config:  $DATA_CONFIG"
echo "Project:      $WANDB_PROJECT"
echo "Run name:     $WANDB_RUN_NAME"
echo "Resume mode:  $RESUME_MODE"
echo "Recon dir:    $RECON_DIR"
echo "Output dir:   $OUTPUT_DIR"



if [ "$RESUME_MODE" = "continue" ]; then
  if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found at $CHECKPOINT_PATH. Cannot continue."
    exit 1
  fi
  python train_start_dc.py \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --recon-dir "$RECON_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --max-epochs $MAX_EPOCHS \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --accelerator gpu \
    --devices $NUM_GPUS \
    --precision "$PRECISION" \
    --checkpoint-every-epochs $CKPT_EVERY_EPOCHS \
    --recon-every-epochs $RECON_EVERY_EPOCHS \
    --ckpt-path "$CHECKPOINT_PATH"
elif [ "$RESUME_MODE" = "new" ]; then
  python train_start_dc.py \
    --model-config "$MODEL_CONFIG" \
    --data-config "$DATA_CONFIG" \
    --recon-dir "$RECON_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --max-epochs $MAX_EPOCHS \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$WANDB_RUN_NAME" \
    --accelerator gpu \
    --devices $NUM_GPUS \
    --precision "$PRECISION" \
    --checkpoint-every-epochs $CKPT_EVERY_EPOCHS \
    --recon-every-epochs $RECON_EVERY_EPOCHS
else
  echo "Unknown mode: $RESUME_MODE (use 'continue' or 'new')"
  exit 1
fi
