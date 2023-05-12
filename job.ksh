#!/bin/ksh

#$ -q batch

#$ -o /work/icmub/af469853/liver-baseline-registration/logs/registration_output.out

#$ -N $4

# Setup the project's paths.
PROJECT_DIR="${WORKDIR}/liver-baseline-registration"

# Enable the virtual environment.
source $PROJECT_DIR/venv/bin/activate

python $1 -i $2 -o $3 -p $4 -pl $5
