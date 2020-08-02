import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

MARKERS = ['o','v','>','<','1','2','3','4','s','p','*','h','+']

class BasePlotter:
    def __init__(self):
        fig, ax1 = plt.subplots()
        plt.figure(figsize=(10,10))
        fig.tight_layout()
        self.ax = ax1
        self.fig = fig
        self.plt = plt
class DualAxesPlotter:
    availableTypes = ["simple","dual", "semilog", "log"]
    def __init__(self, varName1, varName2):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel(varName1, color='red')
        ax1.tick_params(axis='y', labelcolor=color)
        color = 'tab:blue'
        ax2.set_ylabel(varName2, color='blue')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig
        self.plt = plt

    def updatePlot(self, x ,var1, var2, realTimePlot=False):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_ylim(0,2)
        self.ax2.set_ylim(-1,1)
        self.ax1.plot(x, var1, color="red")
        self.ax2.plot(x, var2, color="blue")
        if realTimePlot:
            self.plt.pause(0.0001)
        self.plt.show()

    def savePlot(self, name):
        self.plt.savefig(name+".png")

class Plotter:
    color = ["red", "blue", "black"]
    def __init__(self, xlabel: str, ylabel: str):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        fig.tight_layout()
        self.ax = ax1
        self.fig = fig
        self.plt = plt

    def updatePlot(self, x : list , variables: dict,step, save=False ,**kwargs):
        self.ax.set(**kwargs)
        self.ax.grid()
        for i, var in enumerate(variables):
            style = f"k-{MARKERS[step]}"
            self.ax.plot(x, var['data'], style, label=var['name'], markersize=3, linewidth=0.5 )

    def scatter(self, x, y, name):
        self.ax.scatter(x, y)

    def show(self):
        self.plt.show()


class ErrorPlotter(BasePlotter):
    def __init__(self):
        super().__init__()
        self.ax.set_xlabel(r'$N*$')
        self.ax.set_ylabel(r'$||Error||_{2}$')
        self.plt.grid(True)

    def updateLogLogPlot(self,x , y , labels , legend=False):
        # for i in range(y.shape[1]):
            # self.ax.loglog(x, y[:,i],'k'+MARKERS[i]+'-', basey=10,label=fr"$\tau$ = ${labels[i]}$" ,linewidth =0.75)
        self.ax.plot([0,1,10], [1,22,23])
        self.plt.show()
        print("plotie")
        if legend:
            self.plt.legend()

    def savePlot(self, name):
        self.plt.savefig(f"{name}.png")