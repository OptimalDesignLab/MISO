from miso import omEGADS, omMeshMove, Mesh, Vector
import openmdao.api as om

from miso import MISOSolver

options = {
    'mesh': {
        'file': 'data/testOMMeshMovement/cyl.smb',
        'model-file': 'data/testOMMeshMovement/cyl.egads'
    },
    'print-options': True,
    'space-dis': {
        'degree': 1,
        'basis-type': 'H1'
    },
    'time-dis': {
        'steady': True,
        'steady-abstol': 1e-12,
        'steady-restol': 1e-10,
        'ode-solver': 'PTC',
        't-final': 100,
        'dt': 1e12,
        'cfl': 1.0,
        'res-exp': 2.0
    },
    'nonlin-solver': {
        'type': 'newton',
        'printlevel': 3,
        'maxiter': 50,
        'reltol': 1e-10,
        'abstol': 1e-12
    },
    'lin-solver': {
        'type': 'hyprepcg',
        'printlevel': -1,
        'maxiter': 100,
        'abstol': 1e-14,
        'reltol': 1e-14
    },
    'lin-prec': {
        'type': 'hypreboomeramg',
        'printlevel': -1
    },
    'saveresults': False,
    'problem-opts': {
        'uniform-stiff': {
            'lambda': 1,
            'mu': 1
        }
    }
}

if __name__ == "__main__":
    problem = om.Problem()
    model = problem.model
    ivc = om.IndepVarComp()

    ivc.add_output('length', 2.0) # length of cylinder
    ivc.add_output('radius', 0.5) # radius of cylinder

    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('surf_mesh_move', omEGADS(csm_file='data/testOMMeshMovement/cyl',
                                                  model_file='data/testOMMeshMovement/cyl.egads',
                                                  mesh_file='data/testOMMeshMovement/cyl.smb',
                                                  tess_file='data/testOMMeshMovement/cyl.eto'))
    
    meshMoveSolver = MISOSolver("MeshMovement", options, problem.comm)
    model.add_subsystem('vol_mesh_move', omMeshMove(solver=meshMoveSolver))
    
    model.connect('des_vars.length', 'surf_mesh_move.length')
    model.connect('des_vars.radius', 'surf_mesh_move.radius')

    model.connect('surf_mesh_move.surf_mesh_disp', 'vol_mesh_move.surf_mesh_disp')

    problem.setup()
    problem.run_model()
