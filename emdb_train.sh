#!/bin/bash
#SBATCH --job-name=MyPythonJob       # Job name, displayed in squeue
#SBATCH --output=slurm-%j.out        # Standard output file, %j will be replaced by job ID
#SBATCH --error=slurm-%j.err         # Standard error file
#SBATCH --nodes=1                    # Number of nodes requested
#SBATCH --ntasks=1                   # Number of tasks per node (usually 1 main process)
#SBATCH --cpus-per-task=16            # Number of CPU cores allocated per task
#SBATCH --mem=48G                    # Amount of memory allocated per node
#SBATCH --time=48:00:00               # Maximum job runtime (HH:MM:SS)
#SBATCH --gres=gpu:1              # Request 1 GPU (if GPU is needed)
#SBATCH --partition=batch            # Submit to the 'batch' partition (check your sinfo output)


source /home/louzihan/anaconda3/etc/profile.d/conda.sh
conda activate hugs-test


DATASETS=(
    "P0_08_outdoor_remove_jacket"
    "P1_16_outdoor_warmup"
    "P2_23_outdoor_hug_tree"
    "P3_31_outdoor_workout"
    "P4_34_mvs_a"
    "P5_39_mvs_b"
    "P6_48_outdoor_walk_downhill"
    "P7_61_outdoor_sit_lie_walk"
    "P8_62_mvs_a"
    "P9_72_outdoor_jump"
)

for DATA_NAME in "${DATASETS[@]}"; do
    echo "start train: $DATA_NAME"
        srun python -u emdb_train.py "$DATA_NAME"
done