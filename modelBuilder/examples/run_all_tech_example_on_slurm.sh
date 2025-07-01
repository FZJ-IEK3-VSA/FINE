#!/bin/bash

#SBATCH --job-name=Test_MB
#SBATCH --output="./logs/Test_MB-%A-%a-%x.out"
#SBATCH --error="./logs/Test_MB-%A-%a-%x.err"
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB ## we nee to config slurm for that!
#SBATCH --exclude=cn[1-40] #optimus
# #SBATCH --array=[0-6]

#### JOB LOGIC ### #TODO can env be activated only if base env is active?
echo "start script now"

python all_tech_example.py #$SLURM_ARRAY_TASK_ID

# how to use:

# 1) activate environment manually in terminal (first deactivate environments --> source deactivate, then source activate <environment-name>)
# 2) cd PATH_TO_"examples" 
# 3) sbatch PATH_TO_THIS_FILE
# 4) check squeue with squeue --me