#!/bin/bash
#SBATCH --job-name=stable-audio-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:geforce_rtx_3090:2
#SBATCH --nodelist=tikgpu07
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -e
# SBATCH --gres=gpu:1
# SBATCH --nodelist=tikgpu[03-09]
# SBATCH --gres=gpu:geforce_rtx_3090:2
# export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
# conda activate /itet-stor/feigao/net_scratch/fg-envs/vec-env
export PATH="$HOME/mambaforge/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /itet-stor/feigao/net_scratch/fg-envs/vec-env

export WANDB_API_KEY=
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export WANDB_SERVICE_WAIT=1200

cd /itet-stor/feigao/home/stable-audio-tools

# Check if arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <config_name> <lambda> <wandb_project> <wandb_run_name> <continue|new>"
    exit 1
fi

CONFIG_NAME=$1
LAMBDA=$2
WANDB_PROJECT=$3
WANDB_RUN_NAME=$4
RESUME_MODE=$5

srun bash train_server.sh "$CONFIG_NAME" "$LAMBDA" "$WANDB_PROJECT" "$WANDB_RUN_NAME" "$RESUME_MODE"


# sinfo -o "%.15N %.6t %.4c %.8z %.6m %.8d %.6w %.8f %20E"



# sbatch train_server.sbatch original-vae128 0 comparison vae-128_size-65536_run-1 continue


# sbatch train_server.sbatch fsq-size16384-sf256-regularLR-fsq256 1 new-weight-check fsq-256_weights-change new






# sbatch train_server.sbatch fsq-size16384-sf256-regularLR-fsq64 1 new-weight-check fsq-64_weights-change_LM-LR-regular-5 continue
# sbatch train_server.sbatch fsq-size16384-sf256-regularLR-fsq128 1 new-weight-check fsq-128_weights-change_LM-LR-regular-5 continue