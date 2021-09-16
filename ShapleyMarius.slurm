#!/bin/bash
#SBATCH -J LDA_Marius
#SBATCH --array=0-241                    # how many tasks in the array
#SBATCH -c 6                             # one CPU core per task
#SBATCH -t 1-0:00:00
#SBATCH -o ShapleyEstimation-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marius.zoican@rotman.utoronto.ca
#SBATCH -p standard
#SBATCH --mem=65000

# Load software
# module load anaconda/3.5.3
# source activate ldaenv

# Run python script with a command line argument
srun --unbuffered python build_shapley.py $SLURM_ARRAY_TASK_ID
