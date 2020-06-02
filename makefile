# *-* makefile -*-
MPIEXEC=${HOME}/Tesis/petsc/arch-linux-c-debug/bin/mpiexec -n ${nproc} 
PYTHON=python3
CASE=run_case

$(CASE):
	${MPIEXEC} ${PYTHON} src/${CASE}.py