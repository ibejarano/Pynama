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
    name = 'taylor-green'
    from cases.taylor_green import TaylorGreen as FemProblem
elif case == 'uniform':
    name = 'uniform'
    from cases.uniform import UniformFlow as FemProblem
elif case == 'ibm-static':
    name = 'ibm-static'
    from cases.immersed_boundary import ImmersedBoundaryStatic as FemProblem
elif case == 'senoidal':
    name = 'senoidal'
    from cases.senoidal import Senoidal as FemProblem
elif case == 'custom-func':
    raise Exception("class not found")
elif case == 'cavity':
    name = 'cavity'
    from cases.cavity import Cavity as FemProblem
else:
    print("Case not defined unabled to import")
    exit()

def generateChart(viscousTime=[0.001,0.002,0.01,0.02,0.03,0.09,0.5]):
    hAx = list()
    vAx = list()
    totalNgl = 13
    for i, ngl in enumerate(range(3,totalNgl,2)):
        fem = FemProblem(ngl=ngl)
        fem.setUp()
        fem.setUpSolver()
        vAx.append(fem.getKLEError(times=viscousTime))

    hAx = list(range(3,totalNgl,2))
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
    fem = FemProblem()
    fem.setUp()
    fem.setUpSolver()
    fem.solveKLETests()

def timeSolving(name):
    fem = FemProblem()
    fem.setUp()
    fem.setUpSolver()
    if not fem.comm.rank:
        fem.logger.info("Solving problem...")
    fem.timer.tic()
    fem.startSolver()
    fem.viewer.writeXmf(name)
    if not fem.comm.rank:
        fem.logger.info(f"Solver Finished in {fem.timer.toc()} seconds")

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

    if runTests == 'kle':
        generateParaviewer()
    elif runTests == 'chart':
        generateChart()
    else:
        timeSolving(name)

    # self.logger.info(yamlData)
    # self.caseName = "taylor-green"

if __name__ == "__main__":
    main()