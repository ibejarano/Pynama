import logging
from mpi4py import MPI

class Base(object):
    """
    ex Abstracty
    """
    def __init__(self, dim):
        # self.comm = comm
        self.dim = dim
        self.dim_w = 1 if dim == 2 else 3
        self.dim_s = 3 if dim == 2 else 6
        if comm != None:
            self.logger = logging.getLogger(("[{}] Class".format(comm.rank)))
        else:
            self.logger = logging.getLogger(("Class"))