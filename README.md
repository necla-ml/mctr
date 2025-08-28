# MCTR: Multi Camera Tracking Transformer
[![arXiv](https://img.shields.io/badge/arXiv-2408.13243-b31b1b.svg)](https://arxiv.org/pdf/2408.13243)

This repository provides training and evaluation code for `MCTR` using MMPTracking and MTMC_NVIDIA datasets. 

## Clone Repo
```sh
git clone <repo_address>
make pull # pull submodules
```

## Create Env 
```sh
mamba env create -f mcmot39
mamba activate mcmot39
```

## Training 
MCTR training is performed in 2 steps: 
1. First step uses `pairwise_init.yaml` config 
2. Second step uses `pairwise.yaml` 

We used slurm for training jobs and corresponding scripts are available with the config files. 

## Evaluation
Evaluation script: `scripts/trackeval_mmptrack.py` & `scripts/trackeval_trackbox_mmptrack.py` performs the inference using the pretrained checkpoint. 
The `trackeval_trackbox_mmptrack.py` script uses the bounding box predictions from the trackbox head. 
It generates prediction and groundtruth files which are then used by `trackeval` library to run the MOT metrics. 
Cross camera metrics are output to the stdout when the script completes

## Other
The `scripts` directory contains other useful sanity checks and visualization scripts that may be helpful for debugging.
