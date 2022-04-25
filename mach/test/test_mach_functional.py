import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from mach import PDESolver, MachMesh, MachFunctional

class TestEMFunctionals(unittest.TestCase):
    coulomb_options = {
        "mesh": {
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

    box_options = {
        "mesh": {
            "file": "data/simple_box.mesh",
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
                "material": "box1",
                "attrs": [1],
                "linear": True
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

    def test_coulomb_forward(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.coulomb_options,
                             comm=prob.comm)

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
                                                func="energy",
                                                depends=["state", "mesh_coords"]),
                                 promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                 promotes_outputs=["energy"])

        force_options = {
            "attributes": [1],
            "axis": [0, 0, 1]
        }
        prob.model.add_subsystem("force",
                                 MachFunctional(solver=emSolver,
                                                func="force",
                                                func_options=force_options,
                                                depends=["state", "mesh_coords"]),
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
                                                func_options=torque_options,
                                                depends=["state", "mesh_coords"]),
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
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.coulomb_options,
                             comm=prob.comm)

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
                                                         func="energy",
                                                         depends=["state", "mesh_coords"]),
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
                                                        func_options=force_options,
                                                        depends=["state", "mesh_coords"]),
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
                                                         func_options=torque_options,
                                                         depends=["state", "mesh_coords"]),
                                          promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                          promotes_outputs=["torque"])
        torque.set_check_partial_options(wrt="*", directional=True)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)

    def test_flux_magnitude(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.box_options,
                             comm=prob.comm)

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

        flux_mag = prob.model.add_subsystem("flux_magnitude",
                                            MachFunctional(solver=emSolver,
                                                           func="flux_magnitude",
                                                           check_partials=True,
                                                           depends=["state", "mesh_coords"]),
                                            promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                            promotes_outputs=["flux_magnitude"])
        flux_mag.set_check_partial_options(wrt="*", directional=False)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)

    def test_flux_density(self):
        prob = om.Problem()

        emSolver = PDESolver(type="magnetostatic",
                             solver_options=self.box_options,
                             comm=prob.comm)

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

        flux = prob.model.add_subsystem("flux_density",
                                        MachFunctional(solver=emSolver,
                                                       func="flux_density",
                                                       check_partials=True,
                                                       depends=["state", "mesh_coords"]),
                                        promotes_inputs=[("mesh_coords", "x_em0"), "state"],
                                        promotes_outputs=["flux_density"])
        flux.set_check_partial_options(wrt="*", directional=True)

        prob.setup()
        prob.run_model()

        data = prob.check_partials(form="central")
        assert_check_partials(data)
if __name__ == "__main__":
    unittest.main()