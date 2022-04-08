import numpy as np
import openmdao.api as om

from .pyMach import PDESolver

def _getMeshCoordsName(solver_options):
    type = solver_options["solver-type"]["type"]

    aero_physics = set(["euler", "navierstokes"])
    if type in aero_physics:
        suffix = "aero"
    elif type == "magnetostatic":
        suffix = "em"
    elif type == "thermal":
        suffix = "conduct"
    else:
        raise RuntimeError("Bad physics given to MachSolver!")
    
    return "x_" + suffix + "0"

class MachMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates
    """

    def initialize(self):
        self.options.declare("solver", types=PDESolver, desc="the mach solver object itself", recordable=False)

    def setup(self):
        solver = self.options["solver"]

        mesh_size = solver.getFieldSize("mesh_coords")
        mesh_coords = np.zeros(mesh_size)
        solver.getMeshCoordinates(mesh_coords)

        solver_options = solver.getOptions()

        mesh_name = _getMeshCoordsName(solver_options)
        self.add_output(mesh_name, distributed=True, val=mesh_coords,
                        desc="mesh node coordinates", tags=["mphys_coordinates"])

class MachState(om.ImplicitComponent):
    """OpenMDAO component that converges the state variables"""

    def initialize(self):
        self.options.declare("solver", types=PDESolver, desc="the mach solver object itself", recordable=False)
        self.options.declare("depends", types=list)
        self.options.declare("check_partials", default=False)

    def setup(self):
        solver = self.options["solver"]

        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        for input in self.options["depends"]:
            if ext_fields and input in solver_options["external-fields"]:
                self.add_input(input,
                               shape=solver.getFieldSize(input),
                               tags=["mphys_coupling"])
            else:
                self.add_input(input,
                               tags=["mphys_input"])

        # state inputs
        # mesh_name = _getMeshCoordsName(solver_options)

        self.add_input("mesh_coords",
                       distributed=True,
                       shape_by_conn=True,
                       desc="volume mesh node coordinates",
                       tags=["mphys_coordinates"])

        # state outputs
        local_state_size = solver.getStateSize()
        state = np.zeros(local_state_size)
        self.add_output("state",
                        val=state,
                        distributed=True,
                        desc="Mach state vector",
                        tags=["mphys_coupling"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        solver = self.options["solver"]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        solver.calcResidual(input_dict, residuals["state"])

    def solve_nonlinear(self, inputs, outputs):
        solver = self.options["solver"]

        state = outputs["state"]
        # if (self.options["initial_condition"] is not None):
        #     u_init = self.options["initial_condition"]
        #     solver.setFieldValue(state, u_init)

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        solver.solveForState(input_dict, state)

    def linearize(self, inputs, outputs, residuals):
        # cache inputs
        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        self.linear_inputs = input_dict

        solver = self.options["solver"]
        solver.linearize(input_dict)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        solver = self.options["solver"]

        try:
            if mode == "fwd":
                if "state" in d_residuals: 
                    if "state" in d_outputs: 
                        solver.jacobianVectorProduct(wrt_dot=d_outputs["state"],
                                                     wrt="state",
                                                     res_dot=d_residuals["state"])

                    for input in d_inputs:
                        solver.jacobianVectorProduct(wrt_dot=d_inputs[input],
                                                     wrt=input,
                                                     res_dot=d_residuals["state"])

            elif mode == "rev":
                if "state" in d_residuals: 
                    if "state" in d_outputs: 
                        solver.vectorJacobianProduct(res_bar=d_residuals["state"],
                                                     wrt="state",
                                                     wrt_bar=d_outputs["state"])

                    for input in d_inputs:
                        solver.vectorJacobianProduct(res_bar=d_residuals["state"],
                                                     wrt=input,
                                                     wrt_bar=d_inputs[input])
        except NotImplementedError as err:
            if self.options["check_partials"]:
                pass
            else:
                raise err

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            if self.options["check_partials"]:
                pass
            else:
                raise NotImplementedError("forward mode requested but not implemented")

        if mode == "rev":
            solver = self.options["solver"]
            input_dict = self.linear_inputs
            solver.solveForAdjoint(input_dict,
                                   d_outputs["state"],
                                   d_residuals["state"])