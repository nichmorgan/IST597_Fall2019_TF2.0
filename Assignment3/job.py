import subprocess
import time
from tqdm import tqdm
from itertools import product
import os

depths = [2, 3, 4]
regularizers = ["NLL", "L1", "L2", "L1+L2"]
optimizers = ["adam", "sgd", "rmsprop"]
dropouts = [False, True]

combinations = tqdm(product(depths, regularizers, optimizers, dropouts), desc="Creating jobs", unit="job")

job_ids = []
for depth, regularizer, optimizer, dropout in combinations:
    dropout = str(dropout).lower()
    cmd = f"sbatch ./job.sh {depth} {regularizer} {optimizer} {dropout}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split()[-1]  # Extract job ID, e.g., "Submitted batch job 12345"
    job_ids.append(job_id)
    print(f"Submitted job {job_id} for depth={depth}, regularizer={regularizer}, optimizer={optimizer}, dropout={dropout}")

# Monitor progress
total_jobs = len(job_ids)
completed = 0
with tqdm(total=total_jobs, desc="Job Progress", unit="job") as pbar:
    while completed < total_jobs:
        time.sleep(60)  # Check every minute
        completed = sum(1 for job_id in job_ids if subprocess.run(f"squeue -j {job_id}", shell=True, capture_output=True, text=True).returncode != 0)
        pbar.update(completed - pbar.n)