#!/bin/bash
#SBATCH --job-name=gen-var
#SBATCH --output=/home/igraham/.tmp/%A-%a.out
#SBATCH --time=2-00:00:00
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --array=1-200
#SBATCH --mem=100M
#SBATCH --nodes=1
        
if [[ -z "${SLURM_ARRAY_TASK_ID}" ]]; then
    # test case when 
    NUM=1
else
    NUM=${SLURM_ARRAY_TASK_ID}
fi

python gen_paper_data.py ${NUM}