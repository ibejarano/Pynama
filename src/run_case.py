from cases.taylor_green import TaylorGreen
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np

viscousTime = [0.01, 0.05, 0.1 ,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
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