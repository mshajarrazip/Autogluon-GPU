#!/bin/bash

#-------------------------------------------------------------------------------
#
#   Environment variable set up
#
#-------------------------------------------------------------------------------

# Assign GPUs to use
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# Autogluon is multi-threaded by default - so we need to set the 
# CPU affinity with taskset to avoid hogging up all the resources
CPU_affinity_list="1-8"

taskset -c $CPU_affinity_list python basic-autogluon.py