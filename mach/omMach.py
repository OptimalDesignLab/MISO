from types import FunctionType

import openmdao.api as om
import numpy as np

from .pyMach import MachSolver, Vector

class omMach(om.Group):
    """
    Group that combines the components for the state variables and functionals
    """
    def initialize(self):
        self.options.declare('options_file', types=str, default=None, allow_none=True)
        self.options.declare('options_dict', types=dict, default=None, allow_none=True)
        self.options.declare('solver_type', types=str)
        self.options.declare('initial_condition', types=FunctionType)

    def setup(self):
        # indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        # indeps.add_output('current_density', 10000)
        # indeps.add_output('fill_factor', 0.5)

        if self.options['options_file']:
            solver_opts = self.options['options_file']
        elif self.options['options_dict']:
            solver_opts = self.options['options_dict']
        
        solver_type = self.options['solver_type']
        self.solver = MachSolver(solver_type, solver_opts, self.comm)

        initial_condition = self.options['initial_condition']

        self.add_subsystem('state',
                           omMachState(solver=self.solver,
                                       initial_condition=initial_condition),
                           promotes_inputs=['mesh-coords'])

        self.add_subsystem('functionals',
                           omMachFunctional(solver=self.solver),
                           promotes_inputs=['mesh-coords'],
                           promotes_outputs=['func'])

        # self.connect('current_density', ['state.current_density', 'functionals.current_density'])
        # self.connect('fill_factor', ['state.fill_factor', 'functionals.fill_factor'])
        self.connect('state.state', 'functionals.state')


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
        self.options.declare('options', types=dict, allow_none=True)
        # self.options['distributed'] = True

    def setup(self):
        solver = self.options['solver']

        if self.comm.rank == 0:
            print('Adding functional inputs')

        solver_options = solver.getOptions()
        for input in self.options['depends']:
            if "external-fields" in solver_options:
                if input in solver_options["external-fields"]:
                    print("adding input ", input)
                    if input == "mesh_coords":
                        mesh_size = solver.getFieldSize(input)
                        print(mesh_size)
                        mesh_coords = np.zeros(mesh_size)
                        print(mesh_coords.shape)
                        solver.getField(input, mesh_coords)
                        self.add_input(input, mesh_coords)
                    else:
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
            print('Adding functional outputs')

        func = self.options['func']
        if self.options['options']:
            solver.createOutput(func, self.options['options'])
        else:
            solver.createOutput(func)
        self.add_output(func)

    def setup_partials(self):
        func = self.options['func']
        for input in self.options['depends']:
            self.declare_partials(func, input)

    def compute(self, inputs, outputs):
        solver = self.options['solver']
        func = self.options['func']
        print("mesh_coords: ", inputs["mesh_coords"])
        outputs[func] = solver.calcOutput(func, dict(zip(inputs.keys(), inputs.values())))
        print("mesh_coords: ", inputs["mesh_coords"])
        print("torque: ", outputs[func])

    def compute_partials(self, inputs, partials):
        solver = self.options['solver']
        func = self.options['func']

        inputDict = dict(zip(inputs.keys(), inputs.values()))
        for input in inputs:
            print("calcOutputPartial: ")
            print("\t of: ", func)
            print("\t wrt: ", input)
            print("\t inputs: ", inputDict)
            print("\t partial: ", partials[func, input])
            print("\t partial shape: ", partials[func, input].shape)
            print("\t partial [0] shape: ", partials[func, input][0].shape)
            solver.calcOutputPartial(of=func, wrt=input,
                                     inputs=inputDict,
                                     partial=partials[func, input][0])