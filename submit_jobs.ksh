#!/bin/ksh

#$ -q batch

echo "Starting registration job submission."

# Setup the project's paths.
PROJECT_DIR="${WORKDIR}/liver-baseline-registration"
INPUT_DIR="${PROJECT_DIR}/data/mri_spect_nii_iso"
OUTPUT_DIR="${PROJECT_DIR}/data/elastix"
PIPELINE="${PROJECT_DIR}/pipelines/baseline_bounding_box_ct_seg.json"

# For each patient submit a registration job.
for PATIENT_DIR in $INPUT_DIR/*
	do qsub bl_registration.ksh "$PROJECT_DIR/registration.py" "${INPUT_DIR}" "${OUTPUT_DIR}" "${PATIENT_DIR##*/}" "${PIPELINE}"
done

echo "Registration job submission ended!"
