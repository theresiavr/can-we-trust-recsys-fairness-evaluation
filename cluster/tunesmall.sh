#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=23000M
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00


hostname
nvidia-smi


echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"

py=anaconda3/envs/jointeval/bin/python3.10
prg=jointeval/RecBole/run_hyper_update.py

model=$1
param=jointeval/RecBole/hyperchoice/$model.hyper

cd jointeval

echo $model

tunedata() {
    echo $1
    $py $prg \
                --dataset=$1 \
                --model=$model \
                --params_file=$param \
                --gpu_id=${CUDA_VISIBLE_DEVICES:0:1}
                }

tunedata "new_Amazon-lb" 
tunedata "new_Lastfm" 
tunedata "new_QK-video" 


