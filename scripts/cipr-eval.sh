#!/bin/bash

#SBATCH --partition prod_long
#SBATCH --job-name=mcmot-finetune
#SBATCH -t 3-00:00:00

#SBATCH --nodelist cipr-gpu[18]
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

export MASTER_PORT=$SLURM_JOB_ID
export WORLD_SIZE=1

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

ENV_PATH="sinet39" #"/net/mlfs01/export/users/dpatel/bin/miniconda3/envs/mcmot39/envs/mcmot39"
export PATH=/home/ml/dpatel/miniconda3/bin:$PATH
source activate $ENV_PATH

cd /net/mlfs01/export/users/dpatel/mcmot

SCENE=$1

LOG_FILE="$SCENE-$LOG_PREFIX-`date`.log"

echo "Saving output logs to ${LOG_FILE}"

./scripts/evaluate_checkponts_trackbox.sh runs/office/finetune-continue-b32-600iter-100epochs-frz_dtrTrue-res_unused3-res_tgtFalse-clplinear-R50-cipr-gpu06_May10_155045  office_pred_boxes_new_metric > "$LOG_FILE" 2>&1
# ./scripts/evaluate_checkponts_trackbox.sh runs/retail/finetune-continue-b32-600iter-100epochs-frz_dtrTrue-res_unused3-res_tgtFalse-clplinear-R50-cipr-gpu09_May22_003524 retail_pred_boxes_new_metric > "$LOG_FILE" 2>&1
# ./scripts/evaluate_checkponts_trackbox.sh runs/lobby/finetune-continue-b32-600iter-100epochs-frz_dtrTrue-res_unused3-res_tgtFalse-clplinear-R50-cipr-gpu17_May10_172408 lobby_pred_boxes_new_metric > "$LOG_FILE" 2>&1
# ./scripts/evaluate_checkponts_trackbox.sh runs/industry/finetune-continue-b32-600iter-100epochs-frz_dtrTrue-res_unused3-res_tgtFalse-clplinear-R50-cipr-gpu16_May10_152947 industry_pred_boxes_new_metric > "$LOG_FILE" 2>&1 
