#!/bin/ksh

#$ -q batch

echo "Starting registration job submission."

# Setup the project's paths.
PROJECT_DIR="${WORKDIR}/liver-baseline-registration"
INPUT_DIR="${PROJECT_DIR}/data/nifty_rs_iso_space"
OUTPUT_DIR="${PROJECT_DIR}/data/elastix"
LOG_DIR="${PROJECT_DIR}/logs"
PIPELINE="${PROJECT_DIR}/pipelines/exp3.json"

for PATIENT_DIR in $INPUT_DIR/*
	do qsub job.ksh $PROJECT_DIR/registration.py "-i ${PATIENT_DIR}" "-o ${OUTPUT_DIR}/${PATIENT_DIR##*/}" "-p ${PIPELINE}" "${LOG_DIR}"/${PATIENT_DIR##*/}".out"
done

echo "Registration job submission ended!"

