#!/bin/bash

export job_name="forgetting_mlp__depth_$1__reg_$2__optimizer_$3__dropout_$4"
export output_folder="./results/$job_name"
mkdir -p $output_folder

#SBATCH --job-name=$job_name
#SBATCH --output=$output_folder/output_%a.log
#SBATCH --error=$output_folder/error_%a.log
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=128GB
#SBATCH --ntasks=1

module load anaconda3

export conda_env_name="py311-forgetting-mlp"
if ! conda env list | grep -q "^$conda_env_name\s"; then
    conda create -n $conda_env_name python=3.11
fi 
conda activate $conda_env_name
python -V
ulimit -v unlimited

export CUDA_VISIBLE_DEVICES=""  # Force CPU-only
export depth=$1
export regularizer=$2
export optimizer=$3
export dropout=$4


pip install -r requirements.txt
echo "Memory before Python: $(free -h)" >> $output_folder/debug.log
python forgetting_mlp.py