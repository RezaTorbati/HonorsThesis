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
#SBATCH --output=slurmOut/test_%J_stdout.txt
#SBATCH --error=slurmOut/test_%J_stderr.txt
#SBATCH --time=01:00:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=reza.j.torbati-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/symbiotic/dont_archive/rtorbati
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate
# Change this line to start an instance of your experiment

python3 experiment.py

