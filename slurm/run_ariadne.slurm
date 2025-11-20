#!/bin/bash
#SBATCH --job-name=ariadne_array
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-163%25     # 164 stars, max 25 running at once
#SBATCH --time=12:00:00      # total wall time allowed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4    # match threads used in your fitter
#SBATCH --mem=32G

module load miniconda
module load Python
conda init
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stinc312

# Run your script with the task index
python3 /home/ktp9/stellar-inclinations/stellar-inclinations/code/auto_ariadne.py $SLURM_ARRAY_TASK_ID

