# *-* makefile -*-
MPIEXEC=${HOME}/Tesis/petsc/arch-linux-c-debug/bin/mpirun -n ${nproc} 
PYTHON=python3
CASE=run_case

$(CASE):
	${MPIEXEC} ${PYTHON} src/${CASE}.py -case ibm-dynamic -ksp_type preonly -pc_type lu
