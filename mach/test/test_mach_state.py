import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from mach import PDESolver, MachState, MachMesh

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
            "material": "box1",
            "attr": 1,
            "linear": True
        },
        "attr2": {
            "material": "box2",
            "attr": 2,
            "linear": True
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
class TestEMState(unittest.TestCase):
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

        emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        solver = prob.model.add_subsystem("em_solver",
                                 MachState(solver=emSolver, 
                                           depends=["current_density:wires"],
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
        emSolver = PDESolver(type="magnetostatic", solver_options=em_options, comm=prob.comm)

        prob.model.add_subsystem("ivc",
                                 MachMesh(solver=emSolver),
                                 promotes_outputs=["*"])
        prob.model.add_subsystem("em_solver",
                                 MachState(solver=emSolver, 
                                           depends=["current_density:wires"],
                                           check_partials=True),
                                 promotes_inputs=["current_density:wires", ("mesh_coords", "x_em0")],
                                 promotes_outputs=["state"])

        prob.setup(mode="rev")
        # prob["current_density:wires"] = 1e6
        # om.n2(problem)
        prob.run_model()

        data = prob.check_totals(of=["state"], wrt=["current_density:wires"])
        assert_check_totals(data, atol=1.-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()