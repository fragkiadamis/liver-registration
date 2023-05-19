#!/bin/ksh

#$ -q 'gpu*&!*07"

#$ -o /work/icmub/af469853/liver-baseline-registration/logs/dl_registration.out

#$ -N LocalNet

echo "Submitting deep learning registration."

PROJECT_DIR="${WORKDIR}/liver-baseline-registration"

module load "pytorch/1.11.0/cuda/11.3.1/gpu"
python ${PROJECT_DIR}deep_registration.py

echo "Job submitted."
