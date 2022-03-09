#!/bin/bash
# Reza Torbati
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=10000
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=slurmOut/delete_%J_stdout.txt
#SBATCH --error=slurmOut/delete_%J_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=reza.j.torbati-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/symbiotic/dont_archive/rtorbati/HonorsThesisOSCER
#
#################################################

. /home/fagg/tf_setup.sh
conda activate tf

python3 experiment.py -pklDir /ourdisk/hpc/symbiotic/auto_archive_notyet/tape_2copies/datasets/group/sippc3/kinematics -reduce 4 -exp r -epochs 5000 -patience 250 -kernel_sizes 15 25 25 25 15 15 15 15 15 15 5 5 5 5 5 -filters 25 15 15 20 15 15 25 20 20 30 15 15 10 10 10 -kernel_strides 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -pool_sizes 1 1 1 10 1 1 5 1 1 5 1 8 1 1 1 -pool_strides 1 1 1 5 1 1 3 1 1 5 1 1 1 1 1 -dense 30 24 24 -dropout .35 -l2 0.0001



