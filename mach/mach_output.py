import numpy as np
import openmdao.api as om
from openmdao.utils.mpi import MPI

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

        output_size = solver.getOutputSize(func)
        distributed = True if output_size != 1 else False
        self.add_output(func,
                        distributed=distributed,
                        shape=output_size,
                        tags=["mphys_result"])

    #     # self.declare_partials(func, "state")
    #     # self.declare_partials(func, "mesh_coords")

    # def setup_partials(self):
    #     if self.options["depends"] is not None:
    #         func = self.options["func"]
    #         for input in self.options["depends"]:
    #             self.declare_partials(func, input)

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
                        func_dot = np.zeros_like(d_outputs[func])
                        solver.outputJacobianVectorProduct(of=func,
                                                           inputs=input_dict,
                                                           wrt_dot=d_inputs[input],
                                                           wrt=input,
                                                           out_dot=func_dot)

                        if MPI and self.comm.size > 1:
                            # In Fwd, allreduce the result of the dot product with the subjac.
                            # Allocate buffer of same size and dtype for storing the result.
                            func_dot_global = np.zeros_like(func_dot)
                            self.comm.Allreduce(func_dot, func_dot_global, op=MPI.SUM)
                            d_outputs[func] += func_dot_global
                        else:
                            # Recommended to make sure your code can run without MPI too, for testing.
                            d_outputs[func] += func_dot

        elif mode == 'rev':
            if func in d_outputs:
                input_dict = dict(zip(inputs.keys(), inputs.values()))
                for input in inputs:
                    if input in d_inputs:
                        if MPI and self.comm.size > 1:
                            # In Rev, allreduce the serial derivative vector before the dot product.
                            # Allocate buffer of same size and dtype for storing the result.
                            func_bar = np.zeros_like(d_outputs[func])
                            self.comm.Allreduce(d_outputs[func], func_bar, op=MPI.SUM)
                        else:
                            # Recommended to make sure your code can run without MPI too, for testing.
                            func_bar = d_outputs[func]

                        solver.outputVectorJacobianProduct(of=func,
                                                           inputs=input_dict,
                                                           out_bar=func_bar,
                                                           wrt=input,
                                                           wrt_bar=d_inputs[input])

