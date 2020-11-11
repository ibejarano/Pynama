import matplotlib.pyplot as plt
import numpy as np
import sys
import petsc4py
petsc4py.init(sys.argv)
import logging
import yaml 
import csv

OptDB = petsc4py.PETSc.Options()
case = OptDB.getString('case', False)

customFunctions = ['taylor-green','taylor-green2d-3d', 'taylor-green3dN', 'senoidal', 'flat-plate']

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

MARKERS = ['o','*','>','p','+','1','2','3','4','s','+']

def generateChart(config, viscousTime=[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    viscousTime = [0,2,4,6,8,10]
    # viscousTime = [0.01 , 0.04 , 0.08, 0.12, 0.16]
    #viscousTime = [0.01, 0.02 , 0.05 , 0.1 , 0.15]
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    #viscousTime = [0,0.2,0.4,0.6,0.8,0.9]
    hAx = list()
    vAx = list()
    totalNgl = 11
    plt.figure(figsize=(10,10))
    plt.legend()
    plt.xlabel(r'$N*$')
    plt.ylabel(r'$||\mathit{Error}||_{2}$')
    plt.grid(True)
    plt.tight_layout(pad=3)
    for i, ngl in enumerate(range(3,totalNgl,1)):
        fem = FemProblem(config, ngl=ngl, nelem=[2,2,2])
        fem.setUp()
        fem.setUpSolver()
        vAx.append(fem.getKLEError(viscousTimes=viscousTime))
        del fem
        hAx.append((ngl-1)*2)
        vAxArray = np.array(vAx)

    for i in range(vAxArray.shape[1]):
        plt.loglog(hAx, vAxArray[:,i],'k'+MARKERS[i]+'-', markersize=9, basey=10,label=r'$ t= $' + str(viscousTime[i]) ,linewidth =0.9)
        
    plt.legend()
    hAx = list()
    vAx = list()
    viscousTime = [viscousTime[0], viscousTime[-1]]
    for i, ngl in enumerate(range(3,totalNgl,1)):
        fem = FemProblem(config, ngl=3, nelem=[ngl-1, ngl-1, ngl-1])
        fem.setUp()
        fem.setUpSolver()
        vAx.append(fem.getKLEError(viscousTimes=viscousTime))
        del fem
        hAx.append((ngl-1)*2)

        vAxArray = np.array(vAx)

    for i in range(vAxArray.shape[1]):
        plt.loglog(hAx, vAxArray[:,i],'b'+'-'*(i+1), basey=10,label=fr"$t = {viscousTime[i]} - Q_2 $", linewidth =1.25)
    plt.legend()
    plt.savefig("test-kle-t-gande.png")
    # plt.show()

def generateChartOperators(config):
    Nelem = [list(),list()]
    totalNodes = []
    errors = [[list(), list(), list()],[list(), list(), list()]]
    errors_h = [list(), list(), list()]
    names = ["Convectivo", "Difusivo", "Rotor"]
    #totalNgl = 
    dim = len(config.get("domain").get("nelem"))
    for x, elem in enumerate(range(2,5,2)):
        for i, ngl in enumerate(range(3,int(20/elem+elem-2),1)):
            fem = FemProblem(config, ngl=ngl, nelem=[elem]*dim)
            fem.setUp()
            fem.setUpSolver()
            errorConv, errorDiff, errorCurl = fem.OperatorsTests()
            errors[x][0].append(errorConv)
            errors[x][1].append(errorDiff)
            errors[x][2].append(errorCurl)
            Nelem[x].append((ngl-1)*elem)
            totalNodes.append(((ngl-1) * elem) + 1)

    totalNodes = sorted(list(set(totalNodes)))
    Nelem_h = list()
    print(totalNodes)
    for n in totalNodes:
        nelem = int((n - 1)/2)
        fem = FemProblem(config, ngl=3, nelem=[nelem]*dim)
        fem.setUp()
        fem.setUpSolver()
        errorConv, errorDiff, errorCurl = fem.OperatorsTests()
        errors_h[0].append(errorConv)
        errors_h[1].append(errorDiff)
        errors_h[2].append(errorCurl)
        Nelem_h.append(((3-1)* nelem ) )

    with open(f"out-operators-test-{config['name']}.yaml", "w") as f:
            data = dict()
            data["mesh-2x2"] = {"N": Nelem[0], "error-curl": errors[0][2], "error-diff": errors[0][1], "error-conv": errors[0][0]}
            data["mesh-4x4"] = {"N": Nelem[1], "error-curl": errors[1][2], "error-diff": errors[1][1], "error-conv": errors[1][0]}
            data["mesh-href"] = {"N": Nelem_h, "error-curl": errors_h[2], "error-diff": errors_h[1], "error-conv": errors_h[0]}
            f.write(yaml.dump(data))

    for i in range(3):
        plt.figure(figsize=(10,10))
        plt.loglog(Nelem[0], errors[0][i],marker='o', markersize=3 ,basey=10,linewidth =0.75, color="b", label=r"$n_{elem} = 2$ - refinamiento p")
        plt.loglog(Nelem[1], errors[1][i],marker='o', markersize=3, basey=10,linewidth =0.75, color="r", label=r"$n_{elem} = 4$ - refinamiento p")
        plt.loglog(Nelem_h, errors_h[i],marker='o', markersize=3, basey=10,linewidth =0.75, color="k", label=r"$Q_2$ - refinamiento h")
        plt.xlabel(r'$N*$')
        plt.legend()
        plt.ylabel(r'$||Error||_{2}$')
        plt.grid(True)
        plt.savefig(f"error-{names[i]}-1t")
        plt.clf()

def generateParaviewer(config):
    fem = FemProblem(config)
    fem.setUp()
    fem.setUpSolver()
    fem.solveKLETests()

def generateChartKLE(config):
    fem = FemProblem(config,chart=True)
    fem.setUp()
    fem.setUpSolver()
    if not fem.comm.rank:
        fem.logger.info("Solving problem...")
    fem.timer.tic()
    fem.startSolver()
    if not fem.comm.rank:
        fem.logger.info(f"Solver Finished in {fem.timer.toc()} seconds")
        fem.logger.info(f"Total time: {fem.timerTotal.toc()} seconds")
    fem.getChartKLE()

def generateChartKLEs(config):
    nelems=[6,8,10,20]
    ngl=3
    labels=[]
    dim = len(config.get("domain").get("nelem"))
    file="velerror.csv"
    for i in nelems:
        nelem=[i]*dim
        labels.append(r"$n_{elem}$="+str(nelem)+"-$N_{GL}$="+str(ngl))
        print(labels)
        fem = FemProblem(config,ngl=3, nelem=nelem,chart=True)
        fem.setUp()
        fem.setUpSolver()
        if not fem.comm.rank:
            fem.logger.info("Solving problem...")
        fem.timer.tic()
        fem.startSolver()
        if not fem.comm.rank:
            fem.logger.info(f"Solver Finished in {fem.timer.toc()} seconds")
            fem.logger.info(f"Total time: {fem.timerTotal.toc()} seconds")
        fem.getErrorCsv(file)
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})
    plt.xlabel(r'tiempo')
    plt.ylabel(r'$||Error_{vel}||_{\infty}$')
    colors=["b","r","violet","g","grey","b","r","violet","g","black","lightblue","orange","pink"]
    mark= ["","","","","",'o','>',"|","x",".",""]
    with open(file,"r") as f:
        n=0
        file_csv= csv.reader(f)
        for i, line in enumerate(file_csv):
            if i%2==0:
                time=[float(num) for num in line if num !=""  ]
                print(i)
            else:
                error=[float(num)for num in line if num !="" ]
                plt.plot(time, error ,marker=mark[n], markersize=3 ,color=colors[n],label=labels[n])
                n+=1
    #plt.xlim(-0.001,5)
    plt.legend()    
    plt.title(r'Máximo error de la velocidad en el tiempo')
    plt.savefig(f"Error-Velocidad-mallas-{case}-{nelems[0]}-{nelems[-1]}-{ngl}")
    plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 22})
    plt.xlabel(r'tiempo')
    plt.ylabel(r'$||Error_{vel}||_{\infty}$')
    colors=["b","r","violet","g","grey","b","r","violet","g","black","lightblue","orange","pink"]
    mark= ["","","","","",'o','>',"|","x",".",""]
    with open(file,"r") as f:
        n=0
        file_csv= csv.reader(f)
        for i, line in enumerate(file_csv):
            if i%2==0:
                time=[float(num) for num in line if num !=""  ]
                print(i)
            else:
                error=[float(num)for num in line if num !="" ]
                plt.plot(time, error ,marker=mark[n], markersize=3 ,color=colors[n],label=labels[n])
                n+=1
    plt.xlim(-0.001,4)
    plt.legend()    
    plt.title(r'Máximo error de la velocidad en el tiempo')
    plt.savefig(f"Error-Velocidad-mallas-{case}-{nelems[0]}-{nelems[-1]}-ngl-{ngl}-tiempo4")
    
        


def timeSolving(config):
    fem = FemProblem(config)
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
    logger = logging.getLogger("Init")
    try:
        with open(f'src/cases/{case}.yaml') as f:
            yamlData = yaml.load(f, Loader=yaml.Loader)
    except:
        logger.info(f"Case '{case}' Not Found")

    if runTests == 'kle':
        generateParaviewer(yamlData)
    elif runTests == 'chart':
        generateChart(yamlData)
    elif runTests == 'operators':
        generateChartOperators(yamlData)
    elif runTests == 'chartkle':
        generateChartKLE(yamlData)
    elif runTests == 'chartkles':
        generateChartKLEs(yamlData)
    else:
        timeSolving(yamlData)

if __name__ == "__main__":
    main()