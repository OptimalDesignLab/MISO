import copy
from mach.omMeshMovement import omMeshMove

from mphys import Builder
import numpy as np
import openmdao.api as om

from .pyMach import MachSolver

class omMachState(om.ImplicitComponent):
    """OpenMDAO component that converges the state variables"""

    def initialize(self):
        self.options.declare("solver", types=MachSolver)
        self.options.declare("depends", types=list)
        self.options.declare("initial_condition", default=None)

    def setup(self):
        solver = self.options["solver"]

        # solver_options = solver.getOptions()
        # ext_fields = "external-fields" in solver_options
        # for input in self.options["depends"]:
        #     if ext_fields and input in solver_options["external-fields"]:
        #         self.add_input(input, shape=solver.getFieldSize(input), tags=["mphys_coupling"])
        #     else:
        #         self.add_input(input, tags=["mphys_input"])

        # state inputs
        mesh_size = solver.getFieldSize(input)
        mesh_coords = np.empty(mesh_size)
        solver.getField("mesh_coords", mesh_coords)
        self.add_input("mesh_coords", val=mesh_coords, distributed=True, tags=["mphys_coupling"])

        # state outputs
        local_state_size = solver.getStateSize()
        self.add_output("state", distributed=True, shape=local_state_size, tags=["mphys_coupling"])

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

class MachCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare("solver", recordable=False)
        self.options.declare("check_partials")
        self.options.declare("scenario_name", default=None)
        self.options.declare("problem_setup", default=None)

    def setup(self):
        self.solver = self.options["solver"]
        self.check_partials = self.options["check_partials"]
        self.coupled = self.options["coupled"]
        self.conduction = self.options["conduction"]

        # Promote state variables/rhs with physics-specific tag that MPhys expects
        solver_options = self.solver.getOptions()
        physics = solver_options["physics"]
        aero_physics = set("euler", "navierstokes")
        if physics in aero_physics:
            suffix = "aero"
        elif physics == "magnetostatic":
            suffix = "em"
        elif physics == "thermal":
            suffix = "conduct"
        else:
            raise RuntimeError("Bad physics given to MachSolver!")
        mesh_input = "x_" + suffix + "0"

        self.add_subsystem("solver",
                           MachState(solver=self.solver, check_partials=self.check_partials),
                           promotes_inputs=[("mesh_coords", "mach_vol_coords")],
                           promotes_outputs=[("state", "mach_states")])

class MachMesh(om.ExplicitComponent):
    """
    Component to read the initial mesh coordinates
    """

    def initialize(self):
        self.options.declare('solver', default=None, desc='the mach solver object itself', recordable=False)

    def setup(self):
        solver = self.options['solver']

        mesh_size = solver.getFieldSize("mesh_coords")
        mesh_coords = np.zeros(mesh_size)
        solver.getField("mesh_coords", mesh_coords)

        solver_options = solver.getOptions()
        physics = solver_options["physics"]

        aero_physics = set("euler", "navierstokes")
        if physics in aero_physics:
            suffix = "aero"
        elif physics == "magnetostatic":
            suffix = "em"
        elif physics == "thermal":
            suffix = "conduct"
        else:
            raise RuntimeError("Bad physics given to MachSolver!")
        
        mesh_output = "x_" + suffix + "0"

        self.add_output(mesh_output, distributed=True, val=mesh_coords, shape=mesh_coords.size,
                        desc='mesh node coordinates', tags=['mphys_coordinates'])

class MachMeshGroup(om.Group):
    def initialize(self):
        self.options.declare('solver', default=None, desc='the mach solver object itself', recordable=False)

    def setup(self):
        solver = self.options['solver']
        self.add_subsystem('mesh', MachMesh(solver=solver))

class MachBuilder(Builder):
    def __init__(self, solver_options, check_partials=False):
        self.solver_options = copy.deepcopy(solver_options)
        self.check_partials = check_partials
    
    def initialize(self, comm):
        solver_options = copy.deepcopy(self.solver_options)

        # Create mach solver instance
        self.comm = comm
        self.solver = MachSolver(options=solver_options, comm=comm)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return MachCouplingGroup(solver=self.solver,
                                 check_partials=self.check_partials,
                                 scenario_name=scenario_name,
                                 problem_setup=self.problem_setup)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return MachMeshGroup(solver=self.solver)

    def get_pre_coupling_subsystem(self, scenario_name=None):
        initial_dvs = self.get_initial_dvs()
        return MachPrecouplingGroup(solver=self.solver, initial_dv_vals=initial_dvs)

    def get_post_coupling_subsystem(self, scenario_name=None):
        return MachOutputsGroup(
            solver=self.solver,
            check_partials=self.check_partials,
            scenario_name=scenario_name,
            problem_setup=self.problem_setup
        )

    def get_number_of_nodes(self):
        """
        Get the number of nodes on this processor
        """
        num_states = self.get_ndof()
        state_size = self.solver.getStateSize()
        return state_size // num_states

    def get_ndof(self):
        """
        Get the number of states per node
        """
        return self.solver.getNumStates()


        