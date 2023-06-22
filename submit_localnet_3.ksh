#!/bin/ksh

#$ -o /work/icmub/af469853/liver-baseline-registration/logs/localnet3.out

#$ -N LocalNet_3

echo "Submitting deep learning registration."

PROJECT_DIR="${WORKDIR}/liver-baseline-registration"

module load "pytorch/1.11.0/cuda/11.3.1/gpu"
python ${PROJECT_DIR}/localnet.py -i data/localnet -f 3 -o data/models -n localnet

echo "Job submitted."
