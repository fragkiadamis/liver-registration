#!/bin/ksh

#$ -o /work/icmub/af469853/liver-baseline-registration/logs/localnet_1.out

#$ -N LocalNet_1

echo "Submitting deep learning registration."

PROJECT_DIR="${WORKDIR}/liver-baseline-registration"

module load "pytorch/1.11.0/cuda/11.3.1/gpu"
python ${PROJECT_DIR}/localnet.py -i data/localnet -f 1 -o data/results -n localnet

echo "Job submitted."
