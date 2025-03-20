#!/bin/bash

export job_name="forgetting_mlp__depth_$1__reg_$2__optimizer_$3__dropout_$4"
export output_folder="./results/$job_name"
mkdir -p $output_folder

#SBATCH --job-name=$job_name
#SBATCH --output=$output_folder/output_%a.log
#SBATCH --error=$output_folder/error_%a.log
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB          # Increased to 64GB to avoid OOM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --pty

# Start memory monitoring in the background
echo "Starting memory usage monitoring..." > $output_folder/memory_usage.log
(
  while true; do
    sstat -j $SLURM_JOB_ID --format="AveRSS,MaxRSS" >> $output_folder/memory_usage.log 2>/dev/null
    sleep 1  # Log every 1 second for finer granularity
  done
) &

# Store the PID of the monitoring process
MONITOR_PID=$!

module load anaconda3

export conda_env_name="py311-forgetting-mlp"
if ! conda env list | grep -q "^$conda_env_name\s"; then
    conda create -n $conda_env_name python=3.11
fi
conda activate $conda_env_name
python -V

export depth=$1
export regularizer=$2
export optimizer=$3
export dropout=$4

pip install -r requirements.txt -q
python forgetting_mlp.py

# Stop the memory monitoring after the script finishes or fails
kill $MONITOR_PID
echo "Memory monitoring stopped" >> $output_folder/memory_usage.log