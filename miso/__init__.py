__version__ = '0.0.1'

from mpi4py import MPI

from .pyMISO import MISOSolver, Mesh, Vector

try: 
    import openmdao
except ImportError as err:  
    openmdao = None

    if MPI.COMM_WORLD.rank == 0:
        print('Warning: OpenMDAO dependency is not installed. omMISO wrapper will not be active.')

if openmdao is not None: 
    from .omMISO import omMISOState, omMISOFunctional
    from .omMeshMovement import omMeshMove

try:
    import pyCAPS
except ImportError as err:
    pyCAPS = None

    if MPI.COMM_WORLD.rank == 0:
        print('Warning: pyCAPS dependency is not installed. omEGADS wrapper will not be active.')

if pyCAPS is not None: 
    from .omEGADS import omEGADS