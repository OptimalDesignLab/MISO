__version__ = '0.0.1'

from mpi4py import MPI

from .pyMach import MachSolver, MeshWarper

try: 
    import openmdao
except ImportError as err:  
    openmdao = None

    if MPI.COMM_WORLD.rank == 0:
        print('Warning: OpenMDAO dependency is not installed. omMach wrapper will not be active.')

if openmdao is not None: 
    from .omMach import omMachState, omMachFunctional
    from .mesh_warper import MachMeshWarper

    try:
        import mphys
    except ImportError as err:
        mphys = None

        if MPI.COMM_WORLD.rank == 0:
            print('Warning: MPhys dependency is not installed. MPhys wrapper will not be active.')

    if mphys is not None:
        # from .omMach import omMachState, omMachFunctional
        pass
