from mach import omEGADS, omMeshMove, omMachState, Mesh, Vector
import openmdao.api as om

from mach import MachSolver

mesh_options = {
    'mesh': {
        'file': 'motor.smb',
        'model-file': 'motor.egads'
    },
    'print-options': True,
    'space-dis': {
        'degree': 1,
        'basis-type': 'H1'
    },
    'time-dis': {
        'steady': True,
        'steady-abstol': 1e-8,
        'steady-restol': 1e-8,
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
        'reltol': 1e-8,
        'abstol': 1e-8
    },
    'lin-solver': {
        'type': 'hyprepcg',
        'printlevel': -1,
        'maxiter': 100,
        'abstol': 1e-10,
        'reltol': 1e-10
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

em_options = {
    "silent": False,
    "print-options": False,
    "mesh": {
        "file": "motor.smb",
        "model-file": "motor.egads"
    },
    "space-dis": {
        "basis-type": "nedelec",
        "degree": 1
    },
    "time-dis": {
        "steady": True,
        "steady-abstol": 1e-8,
        "steady-reltol": 1e-8,
        "ode-solver": "PTC",
        "t-final": 100,
        "dt": 1e12,
        "max-iter": 5
    },
    "lin-solver": {
        "type": "hypregmres",
        "printlevel": 2,
        "maxiter": 150,
        "abstol": 1e-10,
        "reltol": 1e-10
    },
    "lin-prec": {
        "type": "hypreams",
        "printlevel": 0
    },
    "nonlin-solver": {
        "type": "newton",
        "printlevel": 3,
        "maxiter": 10,
        "reltol": 1e-8,
        "abstol": 1e-8
    },
    "components": {
        "farfields": {
            "material": "air",
            "linear": True,
            "attrs": [1, 2]
        },
        "stator": {
            "attr": 3,
            "material": "steel",
            "linear": True
        },
        "rotor": {
            "attr": 4,
            "material": "steel",
            "linear": True
        },
        "airgap": {
            "attr": 5,
            "material": "air",
            "linear": True
        },
        "magnets": {
            "material": "NdFeB",
            "linear": True,
            "attrs": [35, 40, 58, 60]
        },
        "windings": {
            "material": "copperwire",
            "linear": True,
            "attrs": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                      36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                      51, 52, 53, 54, 55, 56, 57, 59, 61]
        }
    },
    "problem-opts": {
        "fill-factor": 1.0,
        "current-density": 100000,
        "current" : {
            "Phase-A": [19, 32 ,13, 41, 28, 46, 21, 37, 
                  55, 53, 34, 59, 48, 9, 30, 51, 43, 7, 8, 20],
            "Phase-B": [54, 11, 18, 45, 33, 29, 23, 27,
                  12, 50, 24, 52, 44, 26, 17, 42],
            "Phase-C": [56, 38, 10, 39, 47, 16, 57, 61,
                  25, 36, 49, 6, 31, 22, 15, 14]
        },
        "magnets": {
            "north": [40, 58],
            "south": [35, 60]
        },
    },
    "outputs": {
        "co-energy": {}
    },
    "external-fields": {
        "mesh-coords": {}
    }
}

thermal_options = {
    "print-options": False,
    "mesh": {
        "file": "motor.smb",
        "model-file": "motor.egads"
    },
    "space-dis": {
        "basis-type": "H1",
        "degree": 1
    },
    "steady": False,
    "time-dis": {
        "ode-solver": "MIDPOINT",
        "const-cfl": True,
        "cfl": 1.0,
        "dt": 0.1,
        "t-final": 1.0
    },
    "lin-prec": {
        "type": "hypreboomeramg"
    },
    "lin-solver": {
        "reltol": 1e-10,
        "abstol": 1e-10,
        "printlevel": -1,
        "maxiter": 100
    },
    "nonlin-solver": {
        "type": "newton",
        "printlevel": 3,
        "maxiter": 5,
        "reltol": 1e-8,
        "abstol": 1e-8
    },
    "components": {
        "farfields": {
            "material": "air",
            "linear": True,
            "attrs": [1, 2]
        },
        "stator": {
            "attr": 3,
            "material": "steel",
            "linear": True
        },
        "rotor": {
            "attr": 4,
            "material": "steel",
            "linear": True
        },
        "airgap": {
            "attr": 5,
            "material": "air",
            "linear": True
        },
        "magnets": {
            "material": "NdFeB",
            "linear": True,
            "attrs": [35, 40, 58, 60]
        },
        "windings": {
            "material": "copperwire",
            "linear": True,
            "attrs": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                      36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                      51, 52, 53, 54, 55, 56, 57, 59, 61]
        }
    },
    "problem-opts": {
        "rho-agg": 10,
        "init-temp": 300,
        "fill-factor": 1.0,
        "current-density": 100000,
        "frequency": 1500,
        "current" : {
            "Phase-A": [19, 32 ,13, 41, 28, 46, 21, 37, 
                  55, 53, 34, 59, 48, 9, 30, 51, 43, 7, 8, 20],
            "Phase-B": [54, 11, 18, 45, 33, 29, 23, 27,
                  12, 50, 24, 52, 44, 26, 17, 42],
            "Phase-C": [56, 38, 10, 39, 47, 16, 57, 61,
                  25, 36, 49, 6, 31, 22, 15, 14]
        },
        "magnets": {
            "north": [40, 58],
            "south": [35, 60]
        },
    },
    # "bcs": {
    #     "outflux": [0, 0]
    # },
    "outflux-type": "test",
    "outputs": {
        "temp-agg": {}
    },
    "external-fields": {
        "mvp": {
            "basis-type": "nedelec",
            "degree": 1,
            "num-states": 1
        },
        "mesh-coords": {}
    }
}

if __name__ == "__main__":
    problem = om.Problem()
    model = problem.model
    ivc = om.IndepVarComp()

    ivc.add_output('stator_od', 0.15645)
    ivc.add_output('stator_id', 0.12450)
    ivc.add_output('rotor_od', 0.11370)
    ivc.add_output('rotor_id', 0.11125)
    ivc.add_output('slot_depth', 0.01210)
    ivc.add_output('tooth_width', 0.00430)
    ivc.add_output('magnet_thickness', 0.00440)
    ivc.add_output('heatsink_od', 0.16000)
    ivc.add_output('tooth_tip_thickness', 0.00100)
    ivc.add_output('tooth_tip_angle', 10.00000)
    ivc.add_output('slot_radius', 0.00100)
    ivc.add_output('stack_length', 0.03450)

    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('surf_mesh_move',
                        omEGADS(csm_file='high_fidelity_motor',
                                model_file='motor.egads',
                                mesh_file='motor.smb',
                                tess_file='motor.eto'))

    model.connect('des_vars.stator_od', 'surf_mesh_move.stator_od')
    model.connect('des_vars.stator_id', 'surf_mesh_move.stator_id')
    model.connect('des_vars.rotor_od', 'surf_mesh_move.rotor_od')
    model.connect('des_vars.rotor_id', 'surf_mesh_move.rotor_id')
    model.connect('des_vars.slot_depth', 'surf_mesh_move.slot_depth')
    model.connect('des_vars.tooth_width', 'surf_mesh_move.tooth_width')
    model.connect('des_vars.magnet_thickness', 'surf_mesh_move.magnet_thickness')
    model.connect('des_vars.heatsink_od', 'surf_mesh_move.heatsink_od')
    model.connect('des_vars.tooth_tip_thickness', 'surf_mesh_move.tooth_tip_thickness')
    model.connect('des_vars.tooth_tip_angle', 'surf_mesh_move.tooth_tip_angle')
    model.connect('des_vars.slot_radius', 'surf_mesh_move.slot_radius')
    model.connect('des_vars.stack_length', 'surf_mesh_move.stack_length')
    
    meshMoveSolver = MachSolver("MeshMovement", mesh_options, problem.comm)
    model.add_subsystem('vol_mesh_move', omMeshMove(solver=meshMoveSolver))
    model.connect('surf_mesh_move.surf_mesh_disp', 'vol_mesh_move.surf_mesh_disp')

    emSolver = MachSolver("Magnetostatic", em_options, problem.comm)
    model.add_subsystem('em_solver', omMachState(solver=emSolver, 
                                                 initial_condition=Vector([0.0, 0.0, 0.0])))
    model.connect('vol_mesh_move.vol_mesh_coords', 'em_solver.mesh-coords')

    thermalSolver = MachSolver("Thermal", thermal_options, problem.comm)
    model.add_subsystem('thermal_solver', omMachState(solver=thermalSolver,
                                                      initial_condition=300.0))
    model.connect('vol_mesh_move.vol_mesh_coords', 'thermal_solver.mesh-coords')
    model.connect('em_solver.state', 'thermal_solver.mvp')

    problem.setup()
    problem.run_model()
