#!/bin/ksh

#$ -q batch

echo "Starting registration job submission."

# Setup the project's paths.
PROJECT_DIR="${WORKDIR}/liver-baseline-registration"
INPUT_DIR="${PROJECT_DIR}/data/nifty_rs_iso_space"
OUTPUT_DIR="${PROJECT_DIR}/data/elastix"
PIPELINE="${PROJECT_DIR}/pipelines/exp3.json"

# For each patient submit a registration job.
for PATIENT_DIR in $INPUT_DIR/*
	do qsub job.ksh "$PROJECT_DIR/registration.py" "${PATIENT_DIR}" "${OUTPUT_DIR}" "${PATIENT_DIR##*/}" "${PIPELINE}"
done

echo "Registration job submission ended!"
