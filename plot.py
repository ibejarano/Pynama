import yaml
import numpy as np
from scipy import fftpack
# from src.viewer.plotter import Plotter
import pandas as pd
from math import sqrt
# from matplotlib.pyplot import plt

def createPlotData(name, data):
    a = {"name": name,"data": data}
    return a

def removeString(val):
    a = val.split()
    return float(a[1])

def generateWithYaml(name):
    with open(f'{name}.yaml') as f:
        yamlData = yaml.load(f, Loader=yaml.Loader)
    return yamlData

def generateWithText(textName):
    df_raw = pd.read_table(f"{textName}.txt", delimiter="|", names=["-","Time" , "cd", "cl"])
    a = df_raw.drop("-", axis=1).dropna()
    a["Time"] = a["Time"].apply(removeString)
    a["cd"] = a["cd"].apply(removeString)
    a["cl"] = a["cl"].apply(removeString)
    print(a["cl"][:].max())
    cds = createPlotData(r"$C_D$", a["cd"])
    clifts = createPlotData(r"$C_L$", a["cl"])
    return a["Time"], [cds, clifts]

def generateFrecuencies(t, ft):
    f= 0.26
    sins = {"name": r"$sin$" ,"data": 0.32 * np.sin( 2 * np.pi * np.array(t) * f)}
    f_s = len(t)
    X = fftpack.fft(np.array(ft))
    Y = fftpack.fft(sins['data'])
    freqs = fftpack.fftfreq(len(X)) * f_s
    freq_data =  createPlotData(r"$f$" ,  np.abs(X))
    freq_sin = createPlotData(r"$f_sin$", np.abs(Y))
    return freqs, [freq_sin, freq_data]

def plotFrecuencies(name):
    plt = Plotter("C Lift", "Frecuency")
    
    try:
        t, ft = generateWithYaml(name)
        x , ys = generateFrecuencies(t, ft[1]['data'])
    except:
        t, ft = generateWithText(name)
        print("text found")
        x , ys = generateFrecuencies(t, ft[1]['data'])

    plt.updatePlot( x, ys, xlim=(-30,30))

def plot(name):
    plt = Plotter('tiempo [seg]', " ")
    try:
        x, ys = generateWithYaml(name)
    except:
        x, ys= generateWithText(name)
    
    plt.updatePlot(x, ys, xlim=(0.1, 80), ylim=(-0.5,1.6))


def threeGrid(r):
    """supports only three cell grids"""
    r = abs(r)
    if r <=  0.5:
        return (1 + sqrt(-3*r**2 + 1))/3
    elif r <= 1.5:
        return (5 - 3*r - sqrt(-3*(1-r)**2 + 1))/6
    else:
        return 0

if __name__ == "__main__":
    # plot("salida-re80")
    # a = np.linspace(-1.6,1.6,50)
    # t = np.fromiter((threeGrid(xi) for xi in a), a.dtype)
    # T = createPlotData("v", t)
    # plt.updatePlot(a, [T])
    t, data= generateWithYaml("ibm-sidebyside-d15-200")
    plt = Plotter("r", "d", t , data)
    
    # print(data)