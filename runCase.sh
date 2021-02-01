#!/bin/bash

mpirun -n 1 python3 ./src/run_case.py -case $1 -log_view