#!/bin/bash
#SBATCH --job-name=stable-audio-train-dc
#SBATCH --output=/itet-stor/feigao/net_scratch/logs/A100_%j.out
#SBATCH --error=/itet-stor/feigao/net_scratch/logs/A100_%j.err
#SBATCH --nodelist=tikgpu10
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=35
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL

set -e
# SBATCH --nodelist=tikgpu10
# SBATCH --gres=gpu:geforce_rtx_3090:2
# export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
# squeue -w tikgpu10 -O jobid,username,state,timeused,numcpus,gres:50,reasonlist -S +i

export PATH="$HOME/mambaforge/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /itet-stor/feigao/net_scratch/fg-envs/vec-env

export WANDB_API_KEY=
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export WANDB_SERVICE_WAIT=1200

cd /itet-stor/feigao/home/stable-audio-tools

if [ $# -ne 3 ]; then
  echo "Usage: $0 <wandb_project> <wandb_run_name> <continue|new>"
  exit 1
fi

WANDB_PROJECT=$1
WANDB_RUN_NAME=$2
RESUME_MODE=$3

srun bash train_server_dc.sh "$WANDB_PROJECT" "$WANDB_RUN_NAME" "$RESUME_MODE"

