#!/bin/bash
# Reza Torbati
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
# memory in MB
#SBATCH --mem=10000
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=slurmOut/s_%J_stdout.txt
#SBATCH --error=slurmOut/s_%J_stderr.txt
#SBATCH --time=10:00:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=reza.j.torbati-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/symbiotic/dont_archive/rtorbati/HonorsThesisOSCER
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate
# Change this line to start an instance of your experiment

python3 experiment.py -pklDir /ourdisk/hpc/symbiotic/auto_archive_notyet/tape_2copies/datasets/group/sippc3/kinematics -exp 31dn -epochs 6000 -patience 3000 -kernel_sizes 20 25 25 10 5 5 3 2 2 2 2 1 -filters 35 20 20 20 25 30 30 35 20 15 10 10 -kernel_strides 1 1 1 1 2 1 1 1 1 1 1 1 -pool_sizes 1 1 1 5 1 5 1 3 1 1 1 1 -pool_strides 1 1 1 3 1 5 1 2 1 1 1 1 -dense 40 25 25 -dropout .4 -l2 0

