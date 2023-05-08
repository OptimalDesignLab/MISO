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
        self.options.declare(
            "solver",
            types=PDESolver,
            desc="the mach solver object itself",
            recordable=False,
        )

    def setup(self):
        solver = self.options["solver"]

        mesh_size = solver.getFieldSize("mesh_coords")
        mesh_coords = np.zeros(mesh_size)
        solver.getMeshCoordinates(mesh_coords)

        solver_options = solver.getOptions()

        mesh_name = _getMeshCoordsName(solver_options)
        self.add_output(
            mesh_name,
            distributed=True,
            val=mesh_coords,
            desc="mesh node coordinates",
            tags=["mphys_coordinates"],
        )


class MachState(om.ImplicitComponent):
    """OpenMDAO component that converges the state variables"""

    def initialize(self):
        self.options.declare(
            "solver",
            types=PDESolver,
            desc="the mach solver object itself",
            recordable=False,
        )
        self.options.declare("depends", types=list)
        self.options.declare("check_partials", default=False)

    def setup(self):
        solver = self.options["solver"]

        # hold map of vector-valued I/O names -> contiguous vectors to pass to Mach
        self.vectors = dict()

        # state inputs
        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        for input in self.options["depends"]:
            if input == "mesh_coords":
                mesh_size = solver.getFieldSize(input)
                self.add_input(
                    "mesh_coords",
                    #    distributed=True,
                    shape=mesh_size,
                    #    shape_by_conn=True,
                    desc="volume mesh node coordinates",
                    tags=["mphys_coordinates"],
                )
                self.vectors["mesh_coords"] = np.empty(0)
            else:
                input_size = solver.getFieldSize(input)
                if input_size == 0:
                    input_size = 1
                if ext_fields and input in solver_options["external-fields"]:
                    self.add_input(input, shape=input_size, tags=["mphys_coupling"])
                else:
                    self.add_input(input, shape=input_size, tags=["mphys_input"])
                if input_size > 1:
                    self.vectors[input] = np.empty(input_size)

        # mesh_coords_size = solver.getFieldSize("mesh_coords")
        # self.add_input("mesh_coords",
        #                shape=mesh_coords_size,
        #             #    distributed=True,
        #             #    shape_by_conn=True,
        #                desc="volume mesh node coordinates",
        #                tags=["mphys_coordinates"])

        # state outputs
        local_state_size = solver.getStateSize()
        state = np.zeros(local_state_size)

        mesh_name = _getMeshCoordsName(solver_options)
        if mesh_name == "x_conduct0":
            ref = 1.0
        else:
            ref = 1e6
            # ref = 1.0

        self.add_output(
            "state",
            val=state,
            distributed=True,
            desc="Mach state vector",
            tags=["mphys_coupling"],
            ref=ref,
        )
        self.vectors["state"] = state
        self.vectors["state_res"] = np.empty_like(state)

    def apply_nonlinear(self, inputs, outputs, residuals):
        solver = self.options["solver"]

        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                if self.vectors[input].shape != inputs[input].shape:
                    self.vectors[input].resize(inputs[input].shape)
                self.vectors[input][:] = inputs[input][:]

        # Copy vector outputs into internal contiguous data buffers
        for output in outputs:
            if output in self.vectors:
                self.vectors[output][:] = outputs[output][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        input_dict.update(self.vectors)

        residual = self.vectors["state_res"]
        solver.calcResidual(input_dict, residual)
        residuals["state"][:] = residual[:]

    def solve_nonlinear(self, inputs, outputs):
        solver = self.options["solver"]

        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                if self.vectors[input].shape != inputs[input].shape:
                    self.vectors[input].resize(inputs[input].shape)
                self.vectors[input][:] = inputs[input][:]

        # Copy vector outputs into internal contiguous data buffers
        for output in outputs:
            if output in self.vectors:
                self.vectors[output][:] = outputs[output][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        input_dict.update(self.vectors)

        state = self.vectors["state"]
        solver.solveForState(input_dict, state)
        outputs["state"][:] = state[:]

    def linearize(self, inputs, outputs, residuals):
        # cache inputs
        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                if self.vectors[input].shape != inputs[input].shape:
                    self.vectors[input].resize(inputs[input].shape)
                self.vectors[input][:] = inputs[input][:]

        # Copy vector outputs into internal contiguous data buffers
        for output in outputs:
            if output in self.vectors:
                self.vectors[output][:] = outputs[output][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))
        input_dict.update(self.vectors)
        self.linear_inputs = input_dict

        solver = self.options["solver"]
        solver.linearize(input_dict)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        solver = self.options["solver"]

        solver_options = solver.getOptions()
        mesh_name = _getMeshCoordsName(solver_options)
        if mesh_name == "x_conduct0":
            solver_type = "thermal"
        else:
            solver_type = "EM"
        try:
            if mode == "fwd":
                if "state" in d_residuals:
                    if "state" in d_outputs:
                        if np.linalg.norm(d_outputs["state"], 2) != 0.0:
                            print(f"{solver_type} solver jacobianVectorProduct wrt state")
                            solver.jacobianVectorProduct(
                                wrt_dot=d_outputs["state"],
                                wrt="state",
                                res_dot=d_residuals["state"],
                            )
                        else:
                            print(f"{solver_type} solver jacobianVectorProduct wrt state zero res_dot")
                            print("zero wrt_dot!")

                    for input in d_inputs:
                        if np.linalg.norm(d_inputs[input], 2) != 0.0:
                            print(f"{solver_type} solver jacobianVectorProduct wrt {input}")
                            solver.jacobianVectorProduct(
                                wrt_dot=d_inputs[input],
                                wrt=input,
                                res_dot=d_residuals["state"],
                            )
                        else:
                            print(f"{solver_type} solver jacobianVectorProduct wrt {input} zero res_dot")
                            print("zero wrt_dot!")

            elif mode == "rev":
                if "state" in d_residuals:
                    if np.linalg.norm(d_residuals["state"], 2) != 0.0:
                        print(f"{solver_type} solver adjoint norm: {np.linalg.norm(d_residuals['state'], 2)}")


                        if "state" in d_outputs:
                            print(f"{solver_type} solver vectorJacobianProduct wrt state")
                            solver.vectorJacobianProduct(
                                res_bar=d_residuals["state"],
                                wrt="state",
                                wrt_bar=d_outputs["state"],
                            )

                        for input in d_inputs:
                            print(f"{solver_type} solver vectorJacobianProduct wrt {input}")
                            solver.vectorJacobianProduct(
                                res_bar=d_residuals["state"],
                                wrt=input,
                                wrt_bar=d_inputs[input],
                            )
                    else:
                        print(f"{solver_type} solver zero adjoint!")
                        # print("zero res_bar!")

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

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            if self.options["check_partials"]:
                pass
            else:
                raise NotImplementedError("forward mode requested but not implemented")

        if mode == "rev":

            solver = self.options["solver"]
            solver_options = solver.getOptions()
            mesh_name = _getMeshCoordsName(solver_options)
            if mesh_name == "x_conduct0":
                solver_type = "thermal"
            else:
                solver_type = "EM"

            # print("!!!!!!! Solving for adjoint !!!!!!!")
            print(f"{solver_type} solver solving for adjoint!")
            if np.linalg.norm(d_outputs["state"], 2) != 0.0:
                input_dict = self.linear_inputs
                solver.solveForAdjoint(
                    input_dict, d_outputs["state"], d_residuals["state"]
                )
                print(f"adjoint norm: {np.linalg.norm(d_residuals['state'])}")
                # solver.solveForAdjoint(input_dict,
                #                        state_bar,
                #                        d_residuals["state"])
            else:
                print("zero fun_bar!")
