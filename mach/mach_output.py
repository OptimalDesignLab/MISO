import numpy as np
import openmdao.api as om

from .pyMach import PDESolver

class MachFunctional(om.ExplicitComponent):
    """OpenMDAO component that computes functionals given the state variables"""
    def initialize(self):
        self.options.declare("solver", types=PDESolver, desc="the mach solver object itself", recordable=False)
        self.options.declare("func", types=str)
        self.options.declare("func_options", default=None, types=dict)
        self.options.declare("depends", default=None, types=list)
        self.options.declare("check_partials", default=False)

    def setup(self):
        solver = self.options["solver"]

        self.add_input("state",
                       distributed=True,
                       shape_by_conn=True,
                       desc="Mach state vector",
                       tags=["mphys_coupling"])
    
        self.add_input("mesh_coords",
                       distributed=True,
                       shape_by_conn=True,
                       desc="volume mesh node coordinates",
                       tags=["mphys_coordinates"])

        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        if self.options["depends"] is not None:
            for input in self.options["depends"]:
                if ext_fields and input in solver_options["external-fields"]:
                    self.add_input(input,
                                shape=solver.getFieldSize(input),
                                tags=["mphys_coupling"])
                else:
                    self.add_input(input,
                                tags=["mphys_input"])

        func = self.options["func"]
        if self.options["func_options"]:
            solver.createOutput(func, self.options["func_options"])
        else:
            solver.createOutput(func)

        self.add_output(func,
                        distributed=False,
                        shape=1,
                        tags=["mphys_result"])

        # self.declare_partials(func, "state")
        # self.declare_partials(func, "mesh_coords")

    def setup_partials(self):
        if self.options["depends"] is not None:
            func = self.options["func"]
            for input in self.options["depends"]:
                self.declare_partials(func, input)

    def compute(self, inputs, outputs):
        solver = self.options["solver"]
        func = self.options["func"]
        input_dict = dict(zip(inputs.keys(), inputs.values()))
        outputs[func] = solver.calcOutput(func, input_dict)

    # def compute_partials(self, inputs, partials):
    #     solver = self.options["solver"]
    #     func = self.options["func"]

    #     input_dict = dict(zip(inputs.keys(), inputs.values()))
    #     for input in inputs:
    #         solver.calcOutputPartial(of=func, wrt=input,
    #                                  inputs=input_dict,
    #                                  partial=partials[func, input][0])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        solver = self.options["solver"]
        func = self.options["func"]
        if mode == 'fwd':
            if func in d_outputs:
                input_dict = dict(zip(inputs.keys(), inputs.values()))
                for input in inputs:
                    if input in d_inputs:
                        partial = np.zeros(d_inputs[input].size)                        
                        solver.calcOutputPartial(of=func, wrt=input,
                                                inputs=input_dict,
                                                partial=partial)

                        d_outputs[func] += np.dot(partial, d_inputs[input])

        elif mode == 'rev':
            if func in d_outputs:
                input_dict = dict(zip(inputs.keys(), inputs.values()))
                for input in inputs:
                    if input in d_inputs:
                        partial = np.zeros(d_inputs[input].size)
                        solver.calcOutputPartial(of=func, wrt=input,
                                                inputs=input_dict,
                                                partial=partial)

                        d_inputs[input] += d_outputs[func] * partial
