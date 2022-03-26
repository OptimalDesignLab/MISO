import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from mach import PDESolver, MachState, MachMesh, MachFunctional

# em_options = {
#     "mesh": {
#         # "file": "data/testOMMach/parallel_wires.smb",
#         # "model-file": "data/testOMMach/parallel_wires.egads",
#         "file": "data/box.mesh",
#         "refine": 0
#     },
#     "space-dis": {
#         "basis-type": "nedelec",
#         "degree": 1
#     },
#     "time-dis": {
#         "steady": True,
#     },
#     "nonlin-solver": {
#         "type": "newton",
#         "printlevel": 2,
#         "maxiter": 5,
#         "reltol": 1e-6,
#         "abstol": 1e-6
#     },
#     "lin-solver": {
#         "type": "minres",
#         "printlevel": 1,
#         "maxiter": 100,
#         "abstol": 1e-14,
#         "reltol": 1e-14
#     },
#     "adj-solver": {
#         "type": "minres",
#         "printlevel": 1,
#         "maxiter": 100,
#         "abstol": 1e-14,
#         "reltol": 1e-14
#     },
#     "lin-prec": {
#         "printlevel": -1
#     },
#     "components": {
#         # "wires": {
#         #     "material": "copperwire",
#         #     "attrs": [1, 2],
#         #     "linear": True
#         # },
#         "attr1": {
#             "material": "box1",
#             "attr": 1,
#             "linear": True
#         },
#         "attr2": {
#             "material": "box2",
#             "attr": 2,
#             "linear": True
#         }
#     },
#     "bcs": {
#         "essential": "all"
#     },
#     "current": {
#         "wires": {
#             "z": [1, 2]
#         }
#     }
# }

class TestEMFunctionals(unittest.TestCase):
    def test_coulomb_forward(self):
        em_options = {
            "mesh": {
                # "file": "../../test/regression/egads/data/coulomb1984.smb",
                # "model-file": "../../test/regression/egads/data/coulomb1984.egads",
                "file": "data/coulomb1984.mesh",
                "refine": 0
            },
            "space-dis": {
                "basis-type": "nedelec",
                "degree": 1
            },
            "lin-solver": {
                "type": "minres",
                "printlevel": 1,
                "maxiter": 100,
                "abstol": 1e-14,
                "reltol": 1e-14
            },
            "nonlin-solver": {
                "type": "newton",
                "printlevel": 1,
                "maxiter": 5,
                "reltol": 1e-6,
                "abstol": 1e-6
            },
            "components": {
                "ring": {
                    "material": "copperwire",
                    "attrs": [1, 2],
                    "linear": True
                }
            },
            "bcs": {
                "essential": [1, 3]
            },
            "current": {
                "test": {
                    "ring": [1, 2]
                }
            }
        }

        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.zeros(state_size)
        inputs = {
            "current_density:test": 3e6,
            "state": state
        }
        emSolver.solveForState(inputs, state)

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])


        prob.model.add_subsystem("energy",
                                MachFunctional(solver=emSolver,
                                                func="energy"),
                                promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                promotes_outputs=["energy"])

        force_options = {
            "attributes": [1],
            "axis": [0, 0, 1]
        }
        prob.model.add_subsystem("force",
                                MachFunctional(solver=emSolver,
                                            func="force",
                                            func_options=force_options),
                                promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                promotes_outputs=["force"])

        torque_options = {
            "attributes": [1],
            "axis": [0, 0, 1],
            "about": [0.0, 0.0, 0.0]
        }
        prob.model.add_subsystem("torque",
                                MachFunctional(solver=emSolver,
                                                func="torque",
                                                func_options=torque_options),
                                promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                promotes_outputs=["torque"])

        prob.setup()
        prob.run_model()

        energy = prob.get_val("energy")[0]
        self.assertAlmostEqual(energy, 0.0142746123)
        force = prob.get_val("force")[0]
        self.assertAlmostEqual(force, -0.0791988853)
        torque = prob.get_val("torque")[0]
        self.assertAlmostEqual(torque, 0.0000104977)

    def test_energy_partials_coulomb(self):
        em_options = {
            "mesh": {
                # "file": "../../test/regression/egads/data/coulomb1984.smb",
                # "model-file": "../../test/regression/egads/data/coulomb1984.egads",
                "file": "data/coulomb1984.mesh",
                "refine": 0
            },
            "space-dis": {
                "basis-type": "nedelec",
                "degree": 1
            },
            "lin-solver": {
                "type": "minres",
                "printlevel": 1,
                "maxiter": 100,
                "abstol": 1e-14,
                "reltol": 1e-14
            },
            "nonlin-solver": {
                "type": "newton",
                "printlevel": 1,
                "maxiter": 5,
                "reltol": 1e-6,
                "abstol": 1e-6
            },
            "components": {
                "ring": {
                    "material": "copperwire",
                    "attrs": [1, 2],
                    "linear": True
                }
            },
            "bcs": {
                "essential": [1, 3]
            },
            "current": {
                "test": {
                    "ring": [1, 2]
                }
            }
        }
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

        state_size = emSolver.getFieldSize("state")
        state = np.zeros(state_size)
        inputs = {
            "current_density:test": 3e6,
            "state": state
        }
        emSolver.solveForState(inputs, state)

        ivc = prob.model.add_subsystem("ivc",
                                       om.IndepVarComp(),
                                       promotes_outputs=["state"])
        ivc.add_output("state", val=state)

        prob.model.add_subsystem("mesh",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])


        energy = prob.model.add_subsystem("energy",
                                          MachFunctional(solver=emSolver,
                                                         func="energy"),
                                          promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                          promotes_outputs=["energy"])
        energy.set_check_partial_options(wrt="*", directional=True)

        force_options = {
            "attributes": [1],
            "axis": [0, 0, 1]
        }
        force = prob.model.add_subsystem("force",
                                         MachFunctional(solver=emSolver,
                                                        func="force",
                                                        func_options=force_options),
                                         promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                         promotes_outputs=["force"])
        force.set_check_partial_options(wrt="*", directional=True)

        torque_options = {
            "attributes": [1],
            "axis": [0, 0, 1],
            "about": [0.0, 0.0, 0.0]
        }
        torque = prob.model.add_subsystem("torque",
                                          MachFunctional(solver=emSolver,
                                                         func="torque",
                                                         func_options=torque_options),
                                          promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                          promotes_outputs=["torque"])
        torque.set_check_partial_options(wrt="*", directional=True)

        # prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()

        # prob.check_partials(form="central", step=1e-5)
        # data = prob.check_partials(out_stream=None, form="central", step=1e-7)
        data = prob.check_partials(form="central")
        assert_check_partials(data)

    # def test_energy_partials_par_wires(self):
        # em_options = {
        #     "mesh": {
        #         "file": "data/testOMMach/parallel_wires.smb",
        #         "model-file": "data/testOMMach/parallel_wires.egads",
        #         "refine": 0
        #     },
        #     "space-dis": {
        #         "basis-type": "nedelec",
        #         "degree": 1
        #     },
        #     "time-dis": {
        #         "steady": True,
        #         "steady-abstol": 1e-12,
        #         "steady-reltol": 1e-10,
        #         "ode-solver": "PTC",
        #         "t-final": 100,
        #         "dt": 1e12,
        #         "max-iter": 10
        #     },
        #     "lin-solver": {
        #         "type": "minres",
        #         "printlevel": 0,
        #         "maxiter": 100,
        #         "abstol": 1e-14,
        #         "reltol": 1e-14
        #     },
        #     "lin-prec": {
        #         "type": "hypreams",
        #         "printlevel": 0
        #     },
        #     "nonlin-solver": {
        #         "type": "newton",
        #         "printlevel": 3,
        #         "maxiter": 50,
        #         "reltol": 1e-10,
        #         "abstol": 1e-12
        #     },
        #     "components": {
        #         "wires": {
        #             "material": "copperwire",
        #             "attrs": [1, 2],
        #             "linear": True
        #         },
        #     },
        #     "bcs": {
        #         "essential": "all"
        #     },
        #     "problem-opts": {
        #         "current": {
        #             "z": [1, 2]
        #         }
        #     }
        # }
        # prob = om.Problem()

        # emSolver = MachSolver("Magnetostatic", em_options, prob.comm)
        # state_size = emSolver.getFieldSize("state")
        # state = np.zeros(state_size)
        # inputs = {
        #     "current_density": 10e6,
        #     "state": state
        # }
        # emSolver.solveForState(inputs, state)

        # energy = prob.model.add_subsystem("energy",
        #                                   omMachFunctional(solver=emSolver,
        #                                                   func="energy",
        #                                                   depends=["mesh_coords", "state"]),
        #                                   promotes_inputs=["mesh_coords", "state"],
        #                                   promotes_outputs=["energy"])
        # # energy.set_check_partial_options(wrt="*", directional=True)

        # force_options = {
        #     "attributes": [1],
        #     "axis": [0, 1, 0]
        # }
        # force = prob.model.add_subsystem("force",
        #                                  omMachFunctional(solver=emSolver,
        #                                                  func="force",
        #                                                  depends=["mesh_coords", "state"],
        #                                                  options=force_options),
        #                                  promotes_inputs=["mesh_coords", "state"],
        #                                  promotes_outputs=["force"])
        # # force.set_check_partial_options(wrt="*", directional=True)

        # torque_options = {
        #     "attributes": [1],
        #     "axis": [0, 0, 1],
        #     "about": [0.0, 0.0, 0.0]
        # }
        # torque = prob.model.add_subsystem("torque",
        #                                   omMachFunctional(solver=emSolver,
        #                                                   func="torque",
        #                                                   depends=["mesh_coords", "state"],
        #                                                   options=torque_options),
        #                                   promotes_inputs=["mesh_coords", "state"],
        #                                   promotes_outputs=["torque"])
        # # torque.set_check_partial_options(wrt="*", directional=True)

        # prob.set_solver_print(level=0)
        # prob.setup()
        # prob["state"] = state
        # prob.run_model()

        # data = prob.check_partials(out_stream=None, form="central", step=1e-7)
        # assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    # # def test_forward(self):
    # #     prob = om.Problem()

    # #     emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

    # #     prob.model.add_subsystem("ivc",
    # #                              MachMesh(solver=emSolver),
    # #                              promotes_outputs=["*"])
    # #     prob.model.add_subsystem("em_solver",
    # #                              MachState(solver=emSolver, 
    # #                                        depends=["current_density:wires"],
    # #                                        check_partials=True),
    # #                              promotes_inputs=["*"],
    # #                              promotes_outputs=["state"])

    # #     prob.set_solver_print(level=0)
    # #     prob.setup()
    # #     prob["current_density:wires"] = 3e6
    # #     prob.run_model()

    # def test_partials(self):
    #     prob = om.Problem()

    #     emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

    #     prob.model.add_subsystem("ivc",
    #                              MachMesh(solver=emSolver),
    #                              promotes_outputs=["*"])
    #     solver = prob.model.add_subsystem("em_solver",
    #                              MachState(solver=emSolver, 
    #                                        depends=["current_density:wires"],
    #                                        check_partials=True),
    #                              promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
    #                              promotes_outputs=["state"])
    #     solver.set_check_partial_options(wrt="*",
    #                                      directional=False,
    #                                      form="central")

    #     prob.set_solver_print(level=0)
    #     prob.setup()
    #     # prob["current_density:wires"] = 1.0
    #     prob.run_model()

    #     data = prob.check_partials()
    #     # om.partial_deriv_plot("state", "state", data, jac_method="J_rev", binary = False)
    #     om.partial_deriv_plot("state", "mesh_coords", data, jac_method="J_rev", binary = False)
    #     # om.partial_deriv_plot("state", "current_density:wires", data,jac_method="J_rev", binary = False)
    #     assert_check_partials(data)

    # def test_totals(self):
    #     prob = om.Problem()
    #     emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

    #     prob.model.add_subsystem("ivc",
    #                              MachMesh(solver=emSolver),
    #                              promotes_outputs=["*"])
    #     prob.model.add_subsystem("em_solver",
    #                              MachState(solver=emSolver, 
    #                                        depends=["current_density:wires"],
    #                                        check_partials=True),
    #                              promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
    #                              promotes_outputs=["state"])

    #     prob.setup(mode="rev")
    #     # prob["current_density:wires"] = 1e6
    #     # om.n2(problem)
    #     prob.run_model()

    #     data = prob.check_totals(of=["state"], wrt=["current_density:wires"])
    #     assert_check_totals(data, atol=1.-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()