#!/bin/ksh

#$ -q batch

echo "Starting registration job submission."

# Setup the project's paths.
PROJECT_DIR="${WORKDIR}/liver-baseline-registration"
VENV="${PROJECT_DIR}/venv/bin/activate"
INPUT_DIR="${PROJECT_DIR}/data/nii_iso"
PIPELINE="${PROJECT_DIR}/pipelines/baseline_unet3d_masks.json"

# For each patient submit a registration job.
for PATIENT_DIR in $INPUT_DIR/*
do
    # Get Patient directory and patient ID
    PATIENT=${PATIENT_DIR##*/}
    PATIENT_ID=$(echo $PATIENT | sed 's/[A-Za-z._]//g')

    # Create the file to be qsubed.
    {
        echo "#!/bin/ksh"
        echo "#$ -q batch"
        echo "#$ -o ${PROJECT_DIR}/logs/elx_registration.out"
        echo "#$ -N p$PATIENT_ID"
        echo "source ${VENV}"
        echo "python ${PROJECT_DIR}/registration.py -i ${INPUT_DIR} -o results -p ${PATIENT} -pl ${PIPELINE}"
    } >> reg_job.ksh

    # Submit the file in the queue and delete it after
    qsub reg_job.ksh
    rm reg_job.ksh
done

echo "Registration job submission ended!"

QSTAT_HEADER=2                                      # Two lines is the header of the qstat output
QSTAT_LINES=$(qstat | wc -l)                        # Take the total lines of the qstat output
TOTAL_JOBS=$(( ${QSTAT_LINES}-${QSTAT_HEADER} ))    # Take the number of lines minus the number of the header lines

# Log qstat and the number of the jobs submitted.
qstat ; echo $TOTAL_JOBS
