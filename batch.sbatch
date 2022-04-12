#!/bin/bash
# Reza Torbati
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=5500
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=slurmOut/folds_%j_%a_stdout.txt
#SBATCH --error=slurmOut/folds_%j_%a_stderr.txt
#SBATCH --time=00:25:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=reza.j.torbati-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/symbiotic/dont_archive/rtorbati/HonorsThesisOSCER
#SBATCH --array=0-5
#################################################

. /home/fagg/tf_setup.sh
conda activate tf


python3 experiment.py -pklDir /ourdisk/hpc/symbiotic/auto_archive_notyet/tape_2copies/datasets/group/sippc3/kinematics -exp clean2 -foldsPath folds -rot $SLURM_ARRAY_TASK_ID -reduce 4 -epochs 5000 -patience 200 -kernel_sizes 5 7 7 7 7 7 7 7 7 7 7 7 7 7 7 6 5 5 -filters 15 11 11 11 20 18 18 18 25 22 22 22 30 28 28 35 50 90 -pool_sizes 1 1 1 1 3 1 1 1 3 1 1 1 3 1 1 1 1 1 -pool_strides 1 1 1 1 2 1 1 1 2 1 1 1 2 1 1 1 1 1 -dense 90 90 18 -dropout .1 -sDropout .1 -l2 0.0001


