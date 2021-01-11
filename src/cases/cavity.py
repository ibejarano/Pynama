from cases.base_problem import NoSlipFreeSlip
import numpy as np

class Cavity(NoSlipFreeSlip):
    def computeInitialCondition(self, startTime):
        self.vort.set(0.0)