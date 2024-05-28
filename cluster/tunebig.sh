#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=84000M
#SBATCH --gres=gpu:1
#SBATCH --time=15-00:00:00


hostname
nvidia-smi



py=anaconda3/envs/jointeval/bin/python3.10
prg=jointeval/RecBole/run_hyper_update.py


cd jointeval


tunedata() {
    echo $1 $2
    param=jointeval/RecBole/hyperchoice/$2.hyper
    $py $prg \
                --dataset=$1 \
                --model=$2 \
                --params_file=$param \
                --gpu_id=${CUDA_VISIBLE_DEVICES:0:1}
                }


tunedata "new_ML-10M" "ItemKNN" 
tunedata "new_ML-10M" "NCL" 
tunedata "new_ML-10M" "BPR" 
tunedata "new_ML-10M" "MultiVAE"


