#!/bin/bash
#SBATCH --job-name=gen-long-var
#SBATCH --output=/home/igraham/.tmp/%A-%a.out
#SBATCH --time=7-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-50
#SBATCH --mem=100M
#SBATCH --nodes=1
        
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    # test case when 
    NUM=1
else
    NUM=${SLURM_ARRAY_TASK_ID}
fi

python gen_long_data.py ${NUM}