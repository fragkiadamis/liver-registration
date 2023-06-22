#!/bin/ksh

#$ -o /work/icmub/af469853/liver-baseline-registration/logs/localnet.out

#$ -N LocalNet

echo "Submitting deep learning registration."

PROJECT_DIR="${WORKDIR}/liver-baseline-registration"

module load "pytorch/1.11.0/cuda/11.3.1/gpu"
python ${PROJECT_DIR}/global_local_nets.py -i data/localnet -f 4 -o data/models -n localnet

echo "Job submitted."
