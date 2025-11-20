#!/bin/bash
#SBATCH --job-name=spinspotter       # Job name
#SBATCH --output=logs/spinspotter_%A_%a.out   # Standard output
#SBATCH --error=logs/spinspotter_%A_%a.err    # Standard error
#SBATCH --array=0-163                 # Array indices (adjust to number of stars minus 1)
#SBATCH --time=05:00:00               # Max runtime
#SBATCH --mem=8G                       # Memory per job
#SBATCH --cpus-per-task=2             # CPUs per job
#SBATCH --partition=day               # Partition / queue

# Load modules or activate conda environment
module load miniconda
source activate stinc312   # Replace with your environment name

# Make sure logs directory exists
mkdir -p logs_ss

# Run the Python script with the SLURM array index
python auto_spinspotter.py $SLURM_ARRAY_TASK_ID

