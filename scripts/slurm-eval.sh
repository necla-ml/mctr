#!/bin/bash

#SBATCH --partition legacy_preempt
#SBATCH --job-name=EVAL-MCTR
#SBATCH -t 2-00:00:00

## SBATCH --nodelist cipr-gpu[17]

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1

ENV_PATH="sinet39" #"/net/mlfs01/export/users/dpatel/bin/miniconda3/envs/mcmot39/envs/mcmot39"
export PATH=/home/ml/dpatel/miniconda3/bin:$PATH
source activate $ENV_PATH

cd /net/mlfs01/export/users/dpatel/mcmot

SETTING="retail-epoch99-reset1-`date`"
LOG_FILE="$SETTING.log"
CHECKPOINT_PATH="runs/retail/finetune-continue-b32-600iter-100epochs-frz_dtrTrue-res_unused3-res_tgtFalse-clplinear-R50-cipr-gpu09_May22_003524/checkpoints/PAIRWISE.pth"

echo "Saving output logs to ${LOG_FILE}"
python scripts/trackeval_mmptrack_reset1.py $CHECKPOINT_PATH > "$LOG_FILE" 2>&1 

# python scripts/trackeval_trackbox_mmptrack.py $CHECKPOINT_PATH > "$LOG_FILE" 2>&1 