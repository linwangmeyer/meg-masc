#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --time=01-00:20:00  #requested time 1 day and 20 minutes
#SBATCH -p batch    #running on "batch" partition/queue
#SBATCH -N 1    #1 nodes
#SBATCH	-n 2   #2 tasks total
#SBATCH -c 1	#1 cpu cores per task
#SBATCH --mem=8g  #requesting 8GB of RAM total
#SBATCH --output=myjob.%j.out  #saving standard output to file
#SBATCH --error=myjob.%j.err   #saving standard error to file
##SBATCH --mail-type=ALL    #email optitions
##SBATCH --mail-user=your_utln@tufts.edu

#load anaconda module
module load /cluster/tufts/hpc/tools/anaconda/202111

#initialize the shell to use conda
conda init bash

#activate conda environment, if you use locally installed conda env, make sure use the full path to the env
conda activate /cluster/tufts/kuperberglab/lwang11/condaenv/mne

python 02_filter_epochs.py #make sure the results and data generated from your scripts are saved to files

echo "Job completed!"

conda deactivate

