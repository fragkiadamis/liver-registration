#!/bin/ksh

#$ -o /work/icmub/af469853/liver-baseline-registration/logs/globalnet1.out

#$ -N GlobalNet_1

echo "Submitting deep learning registration."

PROJECT_DIR="${WORKDIR}/liver-baseline-registration"

module load "pytorch/1.11.0/cuda/11.3.1/gpu"
python ${PROJECT_DIR}/globalnet.py -i data/globalnet -f 1 -o data/models -n globalnet

echo "Job submitted."
