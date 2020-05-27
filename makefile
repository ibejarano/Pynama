# *-* makefile -*-
MPIEXEC=${HOME}/tesis/petsc/arch-linux2-c-debug/bin/mpiexec -n 1
PYTHON=python3
CASE=uniform

$(CASE):
	${MPIEXEC} ${PYTHON} src/${CASE}.py -yaml src/cases/${CASE}.yaml
