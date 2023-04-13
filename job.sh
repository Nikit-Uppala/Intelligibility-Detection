#!/bin/bash
#SBATCH -A research
#SBATCH -c 1
#SBATCH -o "job.txt"
#SBATCH --mail-type="prince.tomar@students.iiit.ac.in"

# Entrypoint
source ~/.bashrc
echo "Time at entrypoint: " `date`
echo "Working directory: ${PWD}"

echo "Intelligibility Threshold"
python "threshold_gaussian.py"
# Exit
echo "Time at exit: " `date`