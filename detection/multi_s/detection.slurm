#!/bin/sh
#SBATCH --account=share-ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=mbs
#SBATCH --time=0-04:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:a100:1             # Setting the number of GPUs to 1
#SBATCH --mem=512G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=./detection_results/output.txt      # Specifying 'stdout'
#SBATCH --error=./detection_results/output.err        # Specifying 'stderr'

#SBATCH --mail-user=sofiefe@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge
module load Anaconda3/2024.02-1
source activate misogyny_env


python detection_multi_simple.py
#python detection_multi_trans_simple.py
