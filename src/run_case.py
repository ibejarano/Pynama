from cases.taylor_green import TaylorGreen


fem = TaylorGreen(ngl=3)
fem.setUpSolver()
fem.solveKLE()