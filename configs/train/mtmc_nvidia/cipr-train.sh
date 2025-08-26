#!/bin/bash

#SBATCH --partition prod_preempt
#SBATCH --job-name=mcmt-finetune
#SBATCH -t 2-00:00:00

#SBATCH --nodelist cipr-gpu[16]

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:4

export MASTER_PORT=$SLURM_JOB_ID #12340
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source activate <conda_env>

cd /net/mlfs01/export/users/dpatel/mcmot

TASK="train"
CONFIG_PATH="configs/train/mtmc_nvidia/pairwise.yaml"
SCENE=$1

if [ $SCENE == "" ]; then
  echo "no setting name was given"
  exit
fi

LOGDIR="runs/$SCENE"
LOG_PREFIX="$2"
INITIALIZE="$3"

if [ ${LOG_PREFIX} == "" ]; then
    LOG_PREFIX="exp"
fi

LOG_FILE="$SCENE-$LOG_PREFIX-`date`.log"

echo "Saving output logs to ${LOG_FILE}"
srun python main_pairwise.py $TASK \
--LOGDIR $LOGDIR \
--INITIALIZE $INITIALIZE \
--CFG $CONFIG_PATH DATASET.LOCATIONS.train=$SCENE \
--LOG-PREFIX $LOG_PREFIX  > "$LOG_FILE" 2>&1 
