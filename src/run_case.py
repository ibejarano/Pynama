import matplotlib.pyplot as plt
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
import logging
import yaml 

OptDB = petsc4py.PETSc.Options()
case = OptDB.getString('case', False)

if case == 'taylor-green':
    from cases.taylor_green import TaylorGreen as FemProblem
    print("imported")

def generateChart(viscousTime):
    hAx = list()
    vAx = list()
    for i, ngl in enumerate(range(2,6)):
        fem = TaylorGreen(ngl=ngl)
        fem.setUpSolver()
        vAx.append(fem.getKLEError(times=viscousTime))

    hAx = [2, 3, 4, 5]
    marker = [',','v','>','<','1','2','3','4','s','p','*','h','+']
    vAxArray = np.array(vAx)
    plt.figure(figsize=(10,10))
    for i in range(vAxArray.shape[1]):
        plt.semilogy(hAx, vAxArray[:,i],'k'+marker[i]+'-', basey=10,label=r'$ \tau = $' + str(viscousTime[i]) ,linewidth =0.5)

    plt.legend()
    plt.xlabel(r'$N*$')
    plt.ylabel(r'$||Error||_{\infty}$')
    plt.grid(True)
    plt.show()

def generateParaviewer():
    fem = FemProblem(ngl=3)
    fem.setUpSolver()
    fem.solveKLETests()

def timeSolving():
    fem = FemProblem(ngl=3)
    fem.setUp()
    fem.setUpSolver()
    fem.setUpTimeSolver()
    fem.startSolver()
    fem.viewer.writeXmf('Taylor-Green')

def main():
    case = OptDB.getString('case', False)
    log = OptDB.getString('log', 'INFO')
    runTests = OptDB.getString('test', False)

    logging.basicConfig(level=log.upper() )
    logger = logging.getLogger("")

    try:
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)

    except:
        logger.info(f"Case '{case}' Not Found")

    if runTests:
        logger.info(f"Running {runTests} TESTS:  {yamlData['name']} ")
        generateParaviewer()
    else:
        logger.info(f"Running problem:  {yamlData['name']}")
        timeSolving()

    # self.logger.info(yamlData)
    # self.caseName = "taylor-green"

if __name__ == "__main__":
    main()