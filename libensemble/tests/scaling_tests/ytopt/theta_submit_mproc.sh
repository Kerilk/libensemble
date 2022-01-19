#!/bin/bash -x
#COBALT -t 00:30:00
#COBALT -n 4
#COBALT -q debug-cache-quad
#COBALT -A <project code>

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available.

# Name of calling script
export EXE=run_ytopt_xsbench.py

# Communication Method
export COMMS="--comms local"

# Number of workers.
export NWORKERS="--nworkers 4"

# Name of Conda environment
export CONDA_ENV_NAME=<conda_env_name>

export PMI_NO_FORK=1 # Required for python kills on Theta

# Unload Theta modules that may interfere with job monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate $CONDA_ENV_NAME

# Launch libE
#python $EXE $NUM_WORKERS $LIBE_WALLCLOCK > out.txt 2>&1
#python $EXE $NUM_WORKERS > out.txt 2>&1
python $EXE $COMMS $NWORKERS > out.txt 2>&1
