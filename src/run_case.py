from cases.taylor_green import TaylorGreen


fem = TaylorGreen()
fem.setUpSolver()
fem.solve()
