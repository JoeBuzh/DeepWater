#!/bin/bash

thisTime=$1 

export PROJ_HOME="/public/home/buzh/water/DeepWater"

export MPIRUN="/public/software/intel/impi/4.1.0.024/intel64/bin/mpirun"
export PBATCH="$PROJ_HOME/pbatch"
export MPI_HOST="$PROJ_HOME/host_file"
export LOGS_HOME="$PROJ_HOME/logs/logs_forecast"
# export PYTHON="/public/home/buzh/test_env2/miniconda3/bin/python"
export PYTHON="/public/home/buzh/env/miniconda3/bin/python"

if [ ! -d ${LOGS_HOME} ];then
  mkdir -p ${LOGS_HOME}
fi

cd $PROJ_HOME/scripts
python task2file.py $thisTime

cd $PROJ_HOME/task
# mpirun
if [ -f task_$thisTime.txt ];then
  echo 'task file exist ...'
#   mpirun -f ${MPI_HOST} -np 24 ${PBATCH} ./task_$thisTime.txt
  mpirun -np 5 ${PBATCH} ./task_$thisTime.txt
fi
