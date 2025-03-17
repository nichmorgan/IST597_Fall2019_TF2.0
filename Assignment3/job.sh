#!/bin/bash
#SBATCH --job-name=forgetting_mlp__depth_$1__reg_$2__optimizer_$3__dropout_$4
#SBATCH --output=output_%a.log
#SBATCH --error=error_%a.log
#SBATCH --time=1-00:00:00

module load anaconda3

export depth=$1
export regularizer=$2
export optimizer=$3
export dropout=$4

pipenv run python forgetting_mlp.py