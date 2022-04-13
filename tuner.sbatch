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
#SBATCH --output=slurmOut/tuner200REAL_%a_stdout.txt
#SBATCH --error=slurmOut/tuner200REAL_%a_stderr.txt
#SBATCH --time=01:30:00
#SBATCH --job-name=MasteryOfPropulsion
#SBATCH --mail-user=reza.j.torbati-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/symbiotic/dont_archive/rtorbati/HonorsThesisOSCER
#SBATCH --array=0-5
#################################################

. /home/fagg/tf_setup.sh
conda activate tf


python3 tunerExperiment.py -pklDir /ourdisk/hpc/symbiotic/auto_archive_notyet/tape_2copies/datasets/group/sippc3/kinematics -exp tuner200 -foldsPath folds -rot $SLURM_ARRAY_TASK_ID -epochs 5000 -patience 200 -trials 200 -tune 0


