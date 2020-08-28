# import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14

MARKERS = ['o','v','>','<','1','2','3','4','s','p','*','h','+']

# class BasePlotter:
#     def __init__(self):
#         fig, ax1 = plt.subplots()
#         plt.figure(figsize=(10,10))
#         fig.tight_layout()
#         self.ax = ax1
#         self.fig = fig
#         self.plt = plt
# class DualAxesPlotter:
#     availableTypes = ["simple","dual", "semilog", "log"]
#     def __init__(self, varName1, varName2):
#         fig, ax1 = plt.subplots()
#         ax2 = ax1.twinx()
#         color = 'tab:red'
#         ax1.set_xlabel('time (s)')
#         ax1.set_ylabel(varName1, color='red')
#         ax1.tick_params(axis='y', labelcolor=color)
#         color = 'tab:blue'
#         ax2.set_ylabel(varName2, color='blue')
#         ax2.tick_params(axis='y', labelcolor=color)
#         fig.tight_layout()
#         self.ax1 = ax1
#         self.ax2 = ax2
#         self.fig = fig
#         self.plt = plt

#     def updatePlot(self, x ,var1, var2, realTimePlot=False):
#         self.ax1.clear()
#         self.ax2.clear()
#         self.ax1.set_ylim(0,2)
#         self.ax2.set_ylim(-1,1)
#         self.ax1.plot(x, var1, color="red")
#         self.ax2.plot(x, var2, color="blue")
#         if realTimePlot:
#             self.plt.pause(0.0001)
#         self.plt.show()

#     def savePlot(self, name):
#         self.plt.savefig(name+".png")

class Plotter:
    color = ["red", "blue", "black"]
    def __init__(self, xlabel: str, ylabel: str , t: list, var: list):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        fig.tight_layout()
        self.ax = ax1
        self.fig = fig
        self.plt = plt
        self.ax.grid()
        for coef in var:
            self.ax.plot(t[100:] , coef['data'][100:])
        self.plt.show()