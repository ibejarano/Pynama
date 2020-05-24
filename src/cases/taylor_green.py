from cases.base_problem import BaseProblem

class TaylorGreen(BaseProblem):
    def __init__(self, comm):
        super().__init__(comm)

    def buildMatrices(self):
        # create emtpy mats
        # build mats
        raise NotImplementedError

    def setUpBoundaryFunction(self, callback):
        """
        callback: function that describes the problem in the borders
            callback( coords: numpy array )
        # No depende de la cara, sino de la coordenada
        # TODO: How to apply this to every PETSc Vec?
        """
        raise NotImplementedError

    def setUpSolver(self, parameter_list):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError