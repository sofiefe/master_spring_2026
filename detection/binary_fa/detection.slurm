#!/bin/sh
#SBATCH --account=share-ie-idi       # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name=binary
#SBATCH --time=0-04:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
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

# Running your python file
module load Anaconda3/2024.02-1
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#pip install -r requirements.txt
#pip install transformers[torch]
#pip install 'accelerate>=0.26.0'

python detection_binary_fa.py
python detection_binary_trans_fa.py

