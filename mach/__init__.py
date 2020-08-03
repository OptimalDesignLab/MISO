__version__ = '0.0.1'

from mpi4py import MPI

from .pyMach import MachSolver, Mesh, Vector

try: 
    import openmdao
except ImportError as err:  
    openmdao = None

    if MPI.COMM_WORLD.rank == 0:
        print('Warning: OpenMDAO dependency is not installed. omMach wrapper will not be active')

if openmdao is not None: 
    from .omMach import omMach