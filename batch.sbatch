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
#SBATCH --output=slurmOut/init_%J_stdout.txt
#SBATCH --error=slurmOut/init_%J_stderr.txt
#SBATCH --time=03:00:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=reza.j.torbati-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/symbiotic/dont_archive/rtorbati/HonorsThesisOSCER
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate
# Change this line to start an instance of your experiment

python3 experiment.py -pklDir /ourdisk/hpc/symbiotic/auto_archive_notyet/tape_2copies/datasets/group/sippc3/kinematics -epochs 10000 -patience 1000 -batchSize 15 -stepsPerEpoch 12 -kernel_sizes 10 5 5 3 -filters 100 120 140 150 -kernel_strides 5 1 3 1 -pool_sizes 1 3 1 3 -pool_strides 1 2 1 2 -dense 100 25 -dropout .1

