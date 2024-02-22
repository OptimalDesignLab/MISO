__version__ = '0.0.1'

from mpi4py import MPI

from .pyMISO import MISOSolver, PDESolver, MeshWarper

try: 
    import openmdao
except ImportError as err:  
    openmdao = None

    if MPI.COMM_WORLD.rank == 0:
        print('Warning: OpenMDAO dependency is not installed. omMISO wrapper will not be active.')

if openmdao is not None: 
    from .omMISO import omMISOState, omMISOFunctional
    from .mesh_warper import MISOMeshWarper
    from .miso_state import MISOMesh, MISOState
    from .miso_output import MISOFunctional

    try:
        import mphys
    except ImportError as err:
        mphys = None

        if MPI.COMM_WORLD.rank == 0:
            print('Warning: MPhys dependency is not installed. MPhys wrapper will not be active.')

    if mphys is not None:
        from .mphys.miso_builder import MISOBuilder, MISOMeshGroup
