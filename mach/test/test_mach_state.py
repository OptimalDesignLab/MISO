import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from mach import PDESolver, MachState, MachMesh

class TestEMState(unittest.TestCase):
    em_options = {
        "mesh": {
            # "file": "data/testOMMach/parallel_wires.smb",
            # "model-file": "data/testOMMach/parallel_wires.egads",
            "file": "data/box.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "nedelec",
            "degree": 1
        },
        "time-dis": {
            "steady": True,
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 2,
            "maxiter": 5,
            "reltol": 1e-6,
            "abstol": 1e-6
        },
        "lin-solver": {
            "type": "minres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "adj-solver": {
            "type": "minres",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "lin-prec": {
            "printlevel": -1
        },
        "components": {
            # "wires": {
            #     "material": "copperwire",
            #     "attrs": [1, 2],
            #     "linear": True
            # },
            "attr1": {
                "attr": 1,
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            },
            "attr2": {
                "attr": 2,
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "wires": {
                "z": [1, 2]
            }
        }
    }
    # def test_forward(self):
    #     prob = om.Problem()

    #     emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

    #     prob.model.add_subsystem("ivc",
    #                              MachMesh(solver=emSolver),
    #                              promotes_outputs=["*"])
    #     prob.model.add_subsystem("em_solver",
    #                              MachState(solver=emSolver, 
    #                                        depends=["current_density:wires"],
    #                                        check_partials=True),
    #                              promotes_inputs=["*"],
    #                              promotes_outputs=["state"])

    #     prob.set_solver_print(level=0)
    #     prob.setup()
    #     prob["current_density:wires"] = 3e6
    #     prob.run_model()

    def test_partials(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic", solver_options=self.em_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        solver = prob.model.add_subsystem("em_solver",
                                          MachState(solver=emSolver, 
                                                    depends=["current_density:wires", "mesh_coords"],
                                                    check_partials=True),
                                          promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
                                          promotes_outputs=["state"])
        solver.set_check_partial_options(wrt="*",
                                         directional=False,
                                         form="central")

        prob.set_solver_print(level=0)
        prob.setup()
        # prob["current_density:wires"] = 1.0
        prob.run_model()

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:wires", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()
        emSolver = PDESolver(type="magnetostatic", solver_options=self.em_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("em_solver",
                                 MachState(solver=emSolver, 
                                           depends=["current_density:wires", "mesh_coords"],
                                           check_partials=True),
                                 promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")
        # prob["current_density:wires"] = 1e6
        # om.n2(problem)
        prob.run_model()

        data = prob.check_totals(of=["state"], wrt=["current_density:wires"])
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

class TestEMState2D(unittest.TestCase):
    square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "ring": {
                "attrs": [1],
                "material": {
                    "name": "box1",
                    "mu_r": 795774.7154594767
                },
            }
        },
        "bcs": {
            "essential": "all"
        },
        "current": {
            "test": {
                "z": [1]
            }
        }
    }

    def test_partials(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic", solver_options=self.square_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        solver = prob.model.add_subsystem("em_solver",
                                          MachState(solver=emSolver, 
                                                    depends=["current_density:test", "mesh_coords"],
                                                    check_partials=True),
                                          promotes_inputs=["current_density:test", ("mesh_coords", "x_em0")],
                                          promotes_outputs=["state"])
        solver.set_check_partial_options(wrt="*",
                                         directional=False,
                                         form="central",
                                         step=1e-5)

        prob.set_solver_print(level=0)
        prob.setup()

        state_size = emSolver.getFieldSize("state")
        prob["state"] = np.random.randn(state_size)

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:test", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()
        emSolver = PDESolver(type="magnetostatic", solver_options=self.square_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("em_solver",
                                 MachState(solver=emSolver, 
                                           depends=["current_density:test", "mesh_coords"],
                                           check_partials=True),
                                 promotes_inputs=["current_density:test", ("mesh_coords", "x_em0")],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")
        state_size = emSolver.getFieldSize("state")
        state = np.random.randn(state_size)

        prob["state"] = state
        prob["current_density:test"] = 1.0
        prob.run_model()

        data = prob.check_totals(of=["state"],
                                 wrt=["current_density:test", "x_em0"],
                                 form="central",
                                 step=1e-5)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

class TestThermalState(unittest.TestCase):
    square_options = {
        "mesh": {
            "file": "data/simple_square.mesh",
            "refine": 0
        },
        "space-dis": {
            "basis-type": "h1",
            "degree": 1
        },
        "lin-solver": {
            "type": "pcg",
            "printlevel": 1,
            "maxiter": 100,
            "abstol": 1e-14,
            "reltol": 1e-14
        },
        "nonlin-solver": {
            "type": "newton",
            "printlevel": 1,
            "maxiter": 5,
            "reltol": 1e-12,
            "abstol": 1e-12
        },
        "components": {
            "test": {
                "attrs": [1],
                "material": {
                    "name": "test",
                    "kappa": 1.0,
                    "conductivity": {
                        "model": "linear",
                        "sigma_T_ref": 58.14e6,
                        "T_ref": 293.15,
                        "alpha_resistivity": 3.8e-3
                    }
                },
            }
        },
        "bcs": {
            "convection": [1, 2],
            "outflux": [3, 4]
        }
    }

    def test_partials(self):
        prob = om.Problem()

        thermal_solver = PDESolver(type="thermal",
                                   solver_options=self.square_options,
                                   comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])

        solver = prob.model.add_subsystem("thermal_solver",
                                          MachState(solver=thermal_solver, 
                                                    depends=["mesh_coords", "h", "fluid_temp", "thermal_load"],
                                                    check_partials=True),
                                          promotes_inputs=[("mesh_coords", "x_conduct0"), "h", "fluid_temp", "thermal_load"],
                                          promotes_outputs=["state"])
        solver.set_check_partial_options(wrt="*",
                                         directional=False,
                                         form="central",
                                         step=1e-5)

        prob.set_solver_print(level=0)
        prob.setup()

        state_size = thermal_solver.getFieldSize("state")
        prob["state"] = np.random.randn(state_size)
        prob["h"] = 1.0
        prob["fluid_temp"] = 1.0
        prob["thermal_load"] = np.random.randn(state_size)

        data = prob.check_partials()
        # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
        # om.partial_deriv_plot("state", "current_density:test", data,jac_method="J_rev", binary = False)
        assert_check_partials(data)

    def test_totals(self):
        prob = om.Problem()
        thermal_solver = PDESolver(type="thermal", solver_options=self.square_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=thermal_solver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("thermal_solver",
                                 MachState(solver=thermal_solver, 
                                           depends=["mesh_coords", "h", "fluid_temp", "thermal_load"],
                                           check_partials=True),
                                 promotes_inputs=[("mesh_coords", "x_conduct0"), "h", "fluid_temp", "thermal_load"],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")

        state_size = thermal_solver.getFieldSize("state")
        prob["state"] = np.random.randn(state_size)
        prob["h"] = 1.0
        prob["fluid_temp"] = 1.0
        prob["thermal_load"] = np.random.randn(state_size)
        prob.run_model()

        data = prob.check_totals(of=["state"],
                                 wrt=["x_conduct0", "h", "fluid_temp", "thermal_load"],
                                 form="central",
                                 step=1e-5)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()