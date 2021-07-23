from types import FunctionType

import openmdao.api as om
import numpy as np

from .pyMach import MachSolver, Vector

class omMachState(om.ImplicitComponent):
    """OpenMDAO component that converges the state variables"""

    def initialize(self):
        # self.options.declare('options_file', types=str)
        # self.options.declare('options_dict', types=dict)
        self.options.declare('solver', types=MachSolver)
        self.options.declare('initial_condition')
        self.options.declare('depends', types=list)
        # self.options['distributed'] = True

    def setup(self):
        solver = self.options['solver']

        if self.comm.rank == 0:
            print('Adding state inputs')

        # self.add_input('current_density')
        # self.add_input('fill_factor')

        solver_options = solver.getOptions()
        # if "external-fields" in solver_options:
        #     for ext_field in solver_options["external-fields"]:
        #         self.add_input(ext_field, shape=solver.getFieldSize(ext_field))

        for input in self.options['depends']:
            if "external-fields" in solver_options:
                if input in solver_options["external-fields"]:
                    self.add_input(input, shape=solver.getFieldSize(input))
                elif input == "state":
                    self.add_input(input, shape=solver.getStateSize())
                else:
                    self.add_input(input)
            elif input == "state":
                self.add_input(input, shape=solver.getStateSize())
            else:
                self.add_input(input)


        if self.comm.rank == 0:
            print('Adding state outputs')

        local_state_size = solver.getStateSize()
        self.add_output('state', shape=local_state_size)

        #self.declare_partials(of='state', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute the residual
        """
        solver = self.options['solver']

        mesh_coords = inputs['mesh-coords']
        state = solver.getNewField(outputs['state'])
        residual = solver.getNewField(residuals['state'])

        # TODO: change these methods in machSolver to support numpy array 
        # as argument and do the conversion internally
        solver.setResidualInput("mesh-coords", mesh_coords)
        solver.calcResidual(state, residual)


    def solve_nonlinear(self, inputs, outputs):
        """
        Converge the state
        """
        solver = self.options['solver']

        state = solver.getNewField(outputs['state'])

        u_init = self.options['initial_condition']
        solver.setFieldValue(state, u_init)

        input_dict = { k:v for (k,v) in zip(inputs.keys(), inputs.values())}
        print(input_dict)
        solver.solveForState(input_dict, state)
        # solver.printField("state", state, "state")

        B = solver.getField("B")
        solver.printField("B", B, "B", 0)


    def linearize(self, inputs, outputs, residuals):
        """
        Perform assembly of Jacobians/linear forms?
            Only makes sense if this is always called before apply_linear
        """

        solver = self.options['solver']
        state = solver.getNewField(outputs['state'])

        solver.setState(state)


    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):

        solver = self.options['solver']

        if mode == 'rev':
            if 'state' in d_residuals: 
                if 'state' in d_outputs: 
                    d_outputs['state'] = solver.multStateJacTranspose(d_residuals['state'])
        
                if 'mesh-coords' in d_inputs: 
                    d_inputs['mesh-coords'] = solver.multMeshJacTranspose(d_residuals['state'])

                if 'current_density' in d_inputs: 
                    raise NotImplementedError 

                if 'fill_factor' in d_inputs: 
                    raise NotImplementedError 

        elif mode == 'fwd':
            raise NotImplementedError

    def solve_linear(self, d_outputs, d_residuals, mode):

        solver = self.options['solver']

        if mode == 'rev':
            if 'state' in d_residuals: 
                if 'state' in d_outputs: 
                    d_residuals['state'] = solver.invertStateJacTranspose(d_outputs['state'])

        elif mode == 'fwd':
            raise NotImplementedError
        

class omMachFunctional(om.ExplicitComponent):
    """OpenMDAO component that computes functionals given the state variables"""
    def initialize(self):
        self.options.declare('solver', types=MachSolver)
        self.options.declare('func', types=str)
        self.options.declare('depends', types=list)
        self.options.declare('options', default=None, types=dict)

    def setup(self):
        solver = self.options['solver']

        if self.comm.rank == 0:
            print('Adding functional inputs')

        solver_options = solver.getOptions()
        ext_fields = "external-fields" in solver_options
        for input in self.options['depends']:
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
            print('Adding functional outputs')

        func = self.options['func']
        if self.options['options']:
            solver.createOutput(func, self.options['options'])
        else:
            solver.createOutput(func)
        print("adding output", func)
        self.add_output(func)

    def setup_partials(self):
        func = self.options['func']
        for input in self.options['depends']:
            self.declare_partials(func, input)

    def compute(self, inputs, outputs):
        solver = self.options['solver']
        func = self.options['func']
        outputs[func] = solver.calcOutput(func, dict(zip(inputs.keys(), inputs.values())))

    def compute_partials(self, inputs, partials):
        solver = self.options['solver']
        func = self.options['func']

        inputDict = dict(zip(inputs.keys(), inputs.values()))
        for input in inputs:
            solver.calcOutputPartial(of=func, wrt=input,
                                     inputs=inputDict,
                                     partial=partials[func, input][0])