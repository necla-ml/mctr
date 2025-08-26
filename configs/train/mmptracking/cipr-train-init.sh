#!/bin/bash

#SBATCH --partition <slurm_partition>
#SBATCH --job-name=mcmot-init
#SBATCH -t 3-00:00:00

#SBATCH --nodelist cipr-gpu[17]
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:4

export MASTER_PORT=$SLURM_JOB_ID #12340
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

source activate mcmot39

cd mcmot

TASK="train"
CONFIG_PATH="configs/train/pairwise/pairwise_init.yaml"
SCENE=$1

if [ $SCENE == "" ]; then
  echo "no scene name was given"
  exit
fi

LOGDIR="runs/$SCENE"
LOG_PREFIX="${2}_init"

if [ ${LOG_PREFIX} == "" ]; then
    LOG_PREFIX="exp"
fi

LOG_FILE="$SCENE-$LOG_PREFIX-`date`.log"

echo "Saving output logs to ${LOG_FILE}"
srun python main_pairwise.py $TASK \
--LOGDIR $LOGDIR \
--CFG $CONFIG_PATH DATASET.LOCATIONS.train.64am=$SCENE DATASET.LOCATIONS.train.63am=$SCENE DATASET.LOCATIONS.validation.64pm=$SCENE \
--LOG-PREFIX $LOG_PREFIX  > "$LOG_FILE" 2>&1 
