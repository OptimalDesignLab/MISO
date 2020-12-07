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
        'steady-reltol': 1e-8,
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
        'keep-bndrys': 'all',
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
        "steady-abstol": 0.0,
        "steady-reltol": 0.0,
        "ode-solver": "PTC",
        "t-final": 100,
        "dt": 1e14,
        "max-iter": 8
    },
    "lin-solver": {
        "type": "hypregmres",
        "printlevel": 2,
        "maxiter": 250,
        "abstol": 0.0,
        "reltol": 1e-8
    },
    "lin-prec": {
        "type": "hypreams",
        "printlevel": 0
    },
    "nonlin-solver": {
        "type": "relaxed-newton",
        "printlevel": 3,
        "maxiter": 5,
        "reltol": 1e-4,
        "abstol": 5e-1,
        "abort": False
    },
    "components": {
        "farfields": {
            "material": "air",
            "linear": True,
            "attrs": [1, 2]
        },
        "stator": {
            "attr": 3,
            "material": "hiperco50",
            "linear": False
        },
        "rotor": {
            "attr": 4,
            "material": "hiperco50",
            "linear": False
        },
        "airgap": {
            "attr": 5,
            "material": "air",
            "linear": True
        },
        "magnets": {
            "material": "Nd2Fe14B",
            "linear": True,
            "attrs": [6, 10, 14, 18, 22, 26, 30, 34, 38, 42,
                      7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                      8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                      9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        },
        "windings": {
            "material": "copperwire",
            "linear": True,
            "attrs": [46, 47, 48, 49,
                      50, 51, 52, 53, 
                      54, 55, 56, 57, 
                      58, 59, 60, 61, 
                      62, 63, 64, 65,
                      66, 67, 68, 69,
                      70, 71, 72, 73,
                      74, 75, 76, 77,
                      78, 79, 80, 81,
                      82, 83, 84, 85,
                      86, 87, 88, 89, 
                      90, 91, 92, 93,
                      94, 95, 96, 97,
                      98, 99, 100, 101,
                      102, 103, 104, 105,
                      106, 107, 108, 109,
                      110, 111, 112, 113,
                      114, 115, 116, 117,
                      118, 119, 120, 121,
                      122, 123, 124, 125,
                      126, 127, 128, 129,
                      130, 131, 132, 133,
                      134, 135, 136, 137, 138, 139,
                      140, 141, 142, 143, 144, 145]
        }
    },
    "problem-opts": {
        "fill-factor": 0.6,
        "current-density": 11e6,
        "current" : {
            "Phase-A": [46, 47, 48, 49,
                      58, 59, 60, 61, 
                      78, 79, 80, 81,
                      90, 91, 92, 93,
                      102, 103, 104, 105,
                      122, 123, 124, 125,
                      134, 135, 136, 137, 138, 139,
                      140, 141, 142, 143, 144, 145],
            "Phase-B": [50, 51, 52, 53,
                      62, 63, 64, 65,
                      70, 71, 72, 73,
                      82, 83, 84, 85,
                      94, 95, 96, 97,
                      106, 107, 108, 109,
                      114, 115, 116, 117,
                      126, 127, 128, 129],
            "Phase-C": [54, 55, 56, 57,
                      66, 67, 68, 69,
                      74, 75, 76, 77,
                      86, 87, 88, 89, 
                      98, 99, 100, 101,
                      110, 111, 112, 113,
                      118, 119, 120, 121,
                      130, 131, 132, 133]
        },
        "magnets": {
            "south": [6, 10, 14, 18, 22, 26, 30, 34, 38, 42],
            "cw": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
            "north": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
            "ccw": [9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        },
    },
    "bcs": {
        "essential": [1, 3]
    },
    "outputs": {
        "co-energy": {}
    },
    "external-fields": {
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
    "time-dis": {
        "steady": True,
        "steady-abstol": 1e-8,
        "steady-reltol": 1e-8,
        "ode-solver": "PTC",
        "dt": 1e12,
        "max-iter": 5
    },
    "lin-prec": {
        "type": "hypreboomeramg"
    },
    "lin-solver": {
        "type": "hyprepcg",
        "reltol": 1e-10,
        "abstol": 1e-8,
        "printlevel": 3,
        "maxiter": 100
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
            "material": "hiperco50",
            "linear": False
        },
        "rotor": {
            "attr": 4,
            "material": "hiperco50",
            "linear": False
        },
        "airgap": {
            "attr": 5,
            "material": "air",
            "linear": True
        },
        "magnets": {
            "material": "Nd2Fe14B",
            "linear": True,
            "attrs": [6, 10, 14, 18, 22, 26, 30, 34, 38, 42,
                      7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
                      8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
                      9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        },
        "windings": {
            "material": "copperwire",
            "linear": True,
            "attrs": [65, 98, 142, 127,
                      75, 111, 64, 101,
                      138, 145, 46, 86, 113, 68,
                      114, 59, 79, 76,
                      85, 62, 78, 82,
                      61, 104, 94, 119,
                      106, 105, 120, 126, 118, 50,
                      72, 132, 129, 51,
                      139, 73, 60, 49,
                      136, 69, 87, 123,
                      110, 108, 48, 47,
                      103, 52, 56, 67,
                      121, 100, 96, 80,
                      131, 54, 70, 83,
                      102, 122, 125, 116,
                      130, 91, 143, 115,
                      137, 95, 133, 57,
                      144, 128, 134, 77,
                      89, 71, 74, 53,
                      107, 63, 81, 55,
                      93, 90, 88, 92,
                      135, 58, 97, 84,
                      109, 66, 117, 124,
                      99, 112, 140, 141]
        }
    },
    "problem-opts": {
        "keep-bndrys": [19, 20, 42, 43, 72, 73, 87, 88],
        "rho-agg": 10,
        "init-temp": 300,
        "fill-factor": 0.6,
        "current-density": 11e6,
        "frequency": 1000,
        "current" : {
            "Phase-A": [46, 47, 48, 49,
                      58, 59, 60, 61, 
                      78, 79, 80, 81,
                      90, 91, 92, 93,
                      102, 103, 104, 105,
                      122, 123, 124, 125,
                      134, 135, 136, 137, 138, 139,
                      140, 141, 142, 143, 144, 145],
            "Phase-B": [50, 51, 52, 53,
                      62, 63, 64, 65,
                      70, 71, 72, 73,
                      82, 83, 84, 85,
                      94, 95, 96, 97,
                      106, 107, 108, 109,
                      114, 115, 116, 117,
                      126, 127, 128, 129],
            "Phase-C": [54, 55, 56, 57,
                      66, 67, 68, 69,
                      74, 75, 76, 77,
                      86, 87, 88, 89, 
                      98, 99, 100, 101,
                      110, 111, 112, 113,
                      118, 119, 120, 121,
                      130, 131, 132, 133]
        },
        "magnets": {
            "north": [6, 10, 14, 18, 22, 26, 30, 34, 38, 42],
            "cw": [7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
            "south": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
            "ccw": [9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
        },
    },
    "bcs": {
      "essential": [1, 3, 19, 20, 42, 43, 72, 73, 87, 88],
    },
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
    }
}

#     "external-fields": {
#         "mesh-coords": {}
#     }
# "mvp": {
#             "basis-type": "nedelec",
#             "degree": 1,
#             "num-states": 1
#         },

# faces for inner rotor yoke: 42, 43, 72, 73, 
#"keep-bndrys-adj-to": [1, 2],

if __name__ == "__main__":
    problem = om.Problem()
    model = problem.model
    # ivc = om.IndepVarComp()

    # ivc.add_output('stator_od', 0.15645)
    # ivc.add_output('stator_id', 0.12450)
    # ivc.add_output('rotor_od', 0.11370)
    # ivc.add_output('rotor_id', 0.11125)
    # ivc.add_output('slot_depth', 0.01210)
    # ivc.add_output('tooth_width', 0.00430)
    # ivc.add_output('magnet_thickness', 0.00440)
    # ivc.add_output('heatsink_od', 0.16000)
    # ivc.add_output('tooth_tip_thickness', 0.00100)
    # ivc.add_output('tooth_tip_angle', 10.00000)
    # ivc.add_output('slot_radius', 0.00100)
    # ivc.add_output('stack_length', 0.03450)

    # model.add_subsystem('des_vars', ivc)
    # model.add_subsystem('surf_mesh_move',
    #                     omEGADS(csm_file='high_fidelity_motor',
    #                             model_file='motor.egads',
    #                             mesh_file='motor.smb',
    #                             tess_file='motor.eto'))

    # model.connect('des_vars.stator_od', 'surf_mesh_move.stator_od')
    # model.connect('des_vars.stator_id', 'surf_mesh_move.stator_id')
    # model.connect('des_vars.rotor_od', 'surf_mesh_move.rotor_od')
    # model.connect('des_vars.rotor_id', 'surf_mesh_move.rotor_id')
    # model.connect('des_vars.slot_depth', 'surf_mesh_move.slot_depth')
    # model.connect('des_vars.tooth_width', 'surf_mesh_move.tooth_width')
    # model.connect('des_vars.magnet_thickness', 'surf_mesh_move.magnet_thickness')
    # model.connect('des_vars.heatsink_od', 'surf_mesh_move.heatsink_od')
    # model.connect('des_vars.tooth_tip_thickness', 'surf_mesh_move.tooth_tip_thickness')
    # model.connect('des_vars.tooth_tip_angle', 'surf_mesh_move.tooth_tip_angle')
    # model.connect('des_vars.slot_radius', 'surf_mesh_move.slot_radius')
    # model.connect('des_vars.stack_length', 'surf_mesh_move.stack_length')
    
    # meshMoveSolver = MachSolver("MeshMovement", mesh_options, problem.comm)
    # model.add_subsystem('vol_mesh_move', omMeshMove(solver=meshMoveSolver))
    # model.connect('surf_mesh_move.surf_mesh_disp', 'vol_mesh_move.surf_mesh_disp')

    # emSolver = MachSolver("Magnetostatic", em_options, problem.comm)
    # model.add_subsystem('em_solver', omMachState(solver=emSolver, 
    #                                              initial_condition=Vector([0.0, 0.0, 0.0])))
    # model.connect('vol_mesh_move.vol_mesh_coords', 'em_solver.mesh-coords')

    def thermal_init(x):
        if x.normL2() < 0.06:
            return 351.25
        elif x.normL2() < 0.1:
            return 401.15
        else:
            return 283.15

    thermalSolver = MachSolver("Thermal", thermal_options, problem.comm)
    model.add_subsystem('thermal_solver', omMachState(solver=thermalSolver,
                                                      initial_condition=thermal_init))
    # model.connect('vol_mesh_move.vol_mesh_coords', 'thermal_solver.mesh-coords')
    # model.connect('em_solver.state', 'thermal_solver.mvp')

    problem.setup()
    problem.run_model()
