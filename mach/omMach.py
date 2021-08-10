from types import FunctionType

import openmdao.api as om
import numpy as np

from .pyMach import MachSolver, Vector

class omMachState(om.ImplicitComponent):
    """OpenMDAO component that converges the state variables"""

    def initialize(self):
        self.options.declare("solver", types=MachSolver)
        self.options.declare("depends", types=list)
        self.options.declare("initial_condition", default=None)

    def setup(self):
        solver = self.options["solver"]

        if self.comm.rank == 0:
            print("Adding state inputs")

        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        for input in self.options["depends"]:
            print("adding input", input)
            if input == "state":
                self.add_input(input, shape=solver.getStateSize())
            elif input == "mesh_coords":
                mesh_size = solver.getFieldSize(input)
                mesh_coords = np.zeros(mesh_size)
                solver.getField(input, mesh_coords)
                self.add_input(input, mesh_coords)
            elif ext_fields:
                if input in solver_options["external-fields"]:
                    self.add_input(input, shape=solver.getFieldSize(input))
            else:
                self.add_input(input)


        if self.comm.rank == 0:
            print("Adding state outputs")

        local_state_size = solver.getStateSize()
        self.add_output("state", shape=local_state_size)

    # def setup_partials(self):
        # for input in self.options["depends"]:
            # self.declare_partials("state", input)
        # self.declare_partials("state", "state")

    def apply_nonlinear(self, inputs, outputs, residuals):
        solver = self.options["solver"]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))

        residual = residuals["state"]
        solver.calcResidual(input_dict, residual)

    def solve_nonlinear(self, inputs, outputs):
        solver = self.options["solver"]

        state = outputs["state"]
        if (self.options["initial_condition"] is not None):
            u_init = self.options["initial_condition"]
            solver.setFieldValue(state, u_init)

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        solver.solveForState(input_dict, state)

    def linearize(self, inputs, outputs, residuals):
        solver = self.options["solver"]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))

        solver.linearize(input_dict)


    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        solver = self.options["solver"]

        if mode == "rev":
            if "state" in d_residuals: 
                res_bar = d_residuals["state"]
                if "state" in d_outputs: 
                    solver.vectorJacobianProduct(res_bar,
                                                 wrt="state",
                                                 wrt_bar=d_outputs["state"])

                for input in d_inputs:
                    solver.vectorJacobianProduct(res_bar,
                                                 wrt=input,
                                                 wrt_bar=d_inputs[input])

        elif mode == "fwd":
            pass
            # raise NotImplementedError

    def solve_linear(self, d_outputs, d_residuals, mode):
        solver = self.options["solver"]

        if mode == "rev":
            if "state" in d_residuals:
                if "state" in d_outputs:
                    d_residuals["state"] = solver.invertStateJacTranspose(d_outputs["state"])

        elif mode == "fwd":
            raise NotImplementedError
        

class omMachFunctional(om.ExplicitComponent):
    """OpenMDAO component that computes functionals given the state variables"""
    def initialize(self):
        self.options.declare("solver", types=MachSolver)
        self.options.declare("func", types=str)
        self.options.declare("depends", types=list)
        self.options.declare("options", default=None, types=dict)

    def setup(self):
        solver = self.options["solver"]

        if self.comm.rank == 0:
            print("Adding functional inputs")

        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        for input in self.options["depends"]:
            print("adding input", input)
            if input == "state":
                self.add_input(input, shape=solver.getStateSize())
            elif input == "mesh_coords":
                mesh_size = solver.getFieldSize(input)
                mesh_coords = np.zeros(mesh_size)
                solver.getField(input, mesh_coords)
                self.add_input(input, mesh_coords)
            elif ext_fields:
                if input in solver_options["external-fields"]:
                    self.add_input(input, shape=solver.getFieldSize(input))
            else:
                self.add_input(input)

        if self.comm.rank == 0:
            print("Adding functional outputs")

        func = self.options["func"]
        if self.options["options"]:
            solver.createOutput(func, self.options["options"])
        else:
            solver.createOutput(func)
        print("adding output", func)
        self.add_output(func)

    def setup_partials(self):
        func = self.options["func"]
        for input in self.options["depends"]:
            self.declare_partials(func, input)

    def compute(self, inputs, outputs):
        solver = self.options["solver"]
        func = self.options["func"]
        input_dict = dict(zip(inputs.keys(), inputs.values()))
        outputs[func] = solver.calcOutput(func, input_dict)

    def compute_partials(self, inputs, partials):
        solver = self.options["solver"]
        func = self.options["func"]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        for input in inputs:
            solver.calcOutputPartial(of=func, wrt=input,
                                     inputs=input_dict,
                                     partial=partials[func, input][0])