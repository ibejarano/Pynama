import matplotlib.pyplot as plt
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
import logging
import yaml 

OptDB = petsc4py.PETSc.Options()
case = OptDB.getString('case', False)

customFunctions = ['taylor-green', 'senoidal', 'flat-plate']

if case in customFunctions:
    from cases.custom_func import CustomFuncCase as FemProblem
elif case == 'uniform':
    from cases.uniform import UniformFlow as FemProblem
elif case == 'ibm-static':
    from cases.immersed_boundary import ImmersedBoundaryStatic as FemProblem
elif case == 'ibm-dynamic':
    from cases.immersed_boundary import ImmersedBoundaryDynamic as FemProblem
elif case == 'cavity':
    from cases.cavity import Cavity as FemProblem
else:
    print("Case not defined unabled to import")
    exit()

def generateChart(viscousTime=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    # viscousTime = [0.01 , 0.05 , 0.1, 0.15]
    hAx = list()
    vAx = list()
    totalNgl = 21
    for i, ngl in enumerate(range(3,totalNgl,1)):
        fem = FemProblem(ngl=ngl)
        fem.setUp()
        fem.setUpSolver()
        vAx.append(fem.getKLEError(viscousTimes=viscousTime))
        del fem
        hAx.append((ngl-1)*2)

    #hAx = list(range(3,totalNgl,2))
    marker = [',','v','>','<','1','2','3','4','s','p','*','h','+']
    vAxArray = np.array(vAx)
    plt.figure(figsize=(10,10))
    for i in range(vAxArray.shape[1]):
        plt.loglog(hAx, vAxArray[:,i],'k'+marker[i]+'-', basey=10,label=r'$ \tau = $' + str(viscousTime[i]) ,linewidth =0.5)
        
    plt.legend()
    plt.xlabel(r'$N*$')
    plt.ylabel(r'$||Error||_{\infty}$')
    plt.grid(True)
    plt.show()

def generateChartOperators():
    hAx = [list(),list()]
    vAx = [[list(), list(), list()],[list(), list(), list()]]
    names = ["convective", "diffusive", "curl"]
    totalNgl = 7
    dim = 3
    for x, elem in enumerate(range(2,5,2)):
        for i, ngl in enumerate(range(3,totalNgl,1)):
            fem = FemProblem(ngl=ngl, nelem=[elem]*dim)
            fem.setUp()
            fem.setUpSolver()
            errorConv, errorDiff, errorCurl = fem.OperatorsTests()
            vAx[x][0].append(errorConv)
            vAx[x][1].append(errorDiff)
            vAx[x][2].append(errorCurl)
            hAx[x].append((ngl-1)*elem)
    #hAx = list(range(2,totalNgl,1))
    marker = ['h','v','>','<','1','2','3','4','s','p','*','h','+']
    for i in range(3):
        plt.figure(figsize=(10,10))
        plt.loglog(hAx[0], vAx[0][i],'k'+marker[i]+'-', basey=10,linewidth =0.5, color="b", label="Nel=2x2")
        plt.loglog(hAx[1], vAx[1][i],'k'+marker[i]+'-', basey=10,linewidth =0.5, color="r", label="Nel=4x4")
        plt.xlabel(r'$N*$')
        plt.legend()
        plt.ylabel(r'$||Error||_{\infty}$')
        plt.grid(True)
        plt.savefig(f"error-{names[i]}")
        plt.clf()

def generateParaviewer():
    fem = FemProblem()
    fem.setUp()
    fem.setUpSolver()
    fem.solveKLETests()

def timeSolving():
    fem = FemProblem()
    fem.setUp()
    fem.setUpSolver()
    if not fem.comm.rank:
        fem.logger.info("Solving problem...")
    fem.timer.tic()
    fem.startSolver()
    if not fem.comm.rank:
        fem.logger.info(f"Solver Finished in {fem.timer.toc()} seconds")
        fem.logger.info(f"Total time: {fem.timerTotal.toc()} seconds")

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
    elif runTests == 'operators':
        generateChartOperators()
    else:
        timeSolving()

    # self.logger.info(yamlData)
    # self.caseName = "taylor-green"

if __name__ == "__main__":
    main()