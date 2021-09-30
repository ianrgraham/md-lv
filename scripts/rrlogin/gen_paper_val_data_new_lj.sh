#!/bin/bash
#SBATCH --job-name=gen-var
#SBATCH --output=/home/igraham/.tmp/%A-%a.out
#SBATCH --time=3-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-10
#SBATCH --mem=100M
#SBATCH --nodes=1
        
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    # test case when 
    NUM=1
else
    NUM=${SLURM_ARRAY_TASK_ID}
fi

python gen_paper_val_data_new_lj.py ${NUM}