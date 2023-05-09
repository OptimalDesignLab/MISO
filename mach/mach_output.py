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

        # hold map of vector-valued I/O names -> contiguous vectors to pass to Mach
        self.vectors = dict()

        func = self.options["func"]
        if self.options["func_options"]:
            solver.createOutput(func, self.options["func_options"])
        else:
            solver.createOutput(func)    

        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        if self.options["depends"] is not None:
            for input in self.options["depends"]:
                if isinstance(input, tuple):
                    input = input[0]

                if not isinstance(input, str):
                    raise ValueError("Unsupported input type!")

                print(f"setting input: {input}")

                if input == "state":
                    self.add_input("state",
                                   distributed=True,
                                   shape_by_conn=True,
                                   desc="Mach state vector",
                                   tags=["mphys_coupling"])
                    self.vectors["state"] = np.empty(0)

                elif input == "mesh_coords":
                    mesh_size = solver.getFieldSize(input)
                    self.add_input("mesh_coords",
                                #    distributed=True,
                                   shape=mesh_size,
                                #    shape_by_conn=True,
                                   desc="volume mesh node coordinates",
                                   tags=["mphys_coordinates"])
                    self.vectors["mesh_coords"] = np.empty(0)
                else:
                    input_size = solver.getFieldSize(input)
                    if input_size == 0:
                        input_size = 1

                    # distributed = True if input_size != 1 else False
                    if ext_fields and input in solver_options["external-fields"]:
                        tag = "mphys_coupling"
                        distributed = True
                    else:
                        tag = "mphys_input"
                        distributed = False

                    distributed = False
                    self.add_input(input,
                                   distributed=distributed,
                                   shape=input_size,
                                   tags=tag)
                    if input_size > 1:
                        self.vectors[input] = np.empty(input_size)


        output_size = solver.getOutputSize(func)
        # distributed = True if output_size != 1 else False
        if output_size != 1:
            tag = "mphys_coupling"
            distributed = True
        else:
            tag = "mphys_result"
            distributed = False

        # tag = "mphys_result"
        # distributed = False
        self.add_output(func,
                        distributed=distributed,
                        shape=output_size,
                        tags=tag)
        self.vectors[func] = np.empty(output_size)

    def compute(self, inputs, outputs):
        solver = self.options["solver"]
        func = self.options["func"]

        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                if self.vectors[input].shape != inputs[input].shape:
                    self.vectors[input].resize(inputs[input].shape)
                self.vectors[input][:] = inputs[input][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(self.vectors)  

        output = self.vectors[func]
        solver.calcOutput(func, input_dict, output)
        outputs[func][:] = self.vectors[func][:]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        solver = self.options["solver"]
        func = self.options["func"]
        # print(f"Calling compute_jacvec_product for func {func} with inputs: {inputs}, d_inputs: {d_inputs}, d_outputs: {d_outputs}, and mode: {mode}")

        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                if self.vectors[input].shape != inputs[input].shape:
                    self.vectors[input].resize(inputs[input].shape)
                self.vectors[input][:] = inputs[input][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(self.vectors)  

        # for input in inputs:
            # print(f"inputs[{input}] stride: {inputs[input].strides}")

        # for input in inputs:
        #     if input in d_inputs:
                # print(f"d_inputs[{input}]: {d_inputs[input]}")
                # print(f"d_inputs[{input}] stride: {d_inputs[input].strides}")

        # print(f"d_outputs[{func}] stride: {d_outputs[func].strides}")

        try:
            if mode == 'fwd':
                if func in d_outputs:
                    # print(f"func {func} is in d_outputs")
                    for input in inputs:
                        if input in d_inputs:
                            # print(f"input {input} is in d_inputs")
                            # print(f"")
                            func_dot = np.zeros_like(d_outputs[func])
                            func_dot = np.zeros_like(d_outputs[func])

                            # print(f"output {func} jacobianVectorProduct wrt {input}")
                            # print(f"wrt_dot for input {input}: {d_inputs[input]} for fun {func}")
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

                            # print(f"out_dot for func {func}: {d_outputs[func]}")
            elif mode == 'rev':
                if func in d_outputs:
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

                            # print(f"output {func} vectorJacobianProduct wrt {input}")
                            # if np.linalg.norm(func_bar, 2) != 0.0:
                            #     print(f"func_bar norm: {np.linalg.norm(func_bar, 2)}, func_bar shape: {func_bar.shape}")
                            # else:
                            #     print("zero func_bar!")

                            solver.outputVectorJacobianProduct(of=func,
                                                               inputs=input_dict,
                                                               out_bar=func_bar,
                                                               wrt=input,
                                                               wrt_bar=d_inputs[input])
                            # print(f"wrt_bar norm: {np.linalg.norm(d_inputs[input], 2)}")
                            
        except Exception as err:
            if isinstance(err, NotImplementedError):
                if self.options["check_partials"]:
                    print(f"\n\nNot implemented error passed!!!\n\n")
                    pass
                else:
                    print(f"\n\nNot implemented error raised!!!\n\n")
                    raise err
            else:
                print("\n\ngeneric exception!!!\n\n")
                raise err

        # except NotImplementedError as err:
        #     if self.options["check_partials"]:
        #         pass
        #     else:
        #         raise err


