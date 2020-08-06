from .pyMach import MachSolver

import openmdao.api as om

class omMach(om.Group):
    """
    Group that combines the components for the state variables and functionals
    """
    def initialize(self):
        # self.options.declare('options_file', types=str)
        pass

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('current_density', 10000)
        indeps.add_output('fill_factor', 0.5)

        # self.solver = MachSolver(options_file=self.options['options_file'])
        self.solver = MachSolver(self.comm)

        self.add_subsystem('state',
                           omMachState(solver=self.solver),
                           promotes_inputs=['vol_mesh_coords'])
                           # promotes_outputs=['state'])
        self.add_subsystem('functionals',
                           omMachFunctionals(solver=self.solver),
                           promotes_outputs=['func'])

        self.connect('current_density', ['state.current_density', 'functionals.current_density'])
        self.connect('fill_factor', ['state.fill_factor', 'functionals.fill_factor'])
        self.connect('state.state', 'functionals.statee')


class omMachState(om.ImplicitComponent):
    """OpenMDAO component that converges the state variables"""

    def initialize(self):
        self.options.declare('optionsFile', types=str)
        self.options.declare('solver', types=MachSolver)
        # self.options['distributed'] = True

    def setup(self):
        solver = self.options['solver']

        if self.comm.rank == 0:
            print('Adding state inputs')

        local_mesh_size = solver.getMeshSize()
        self.add_input('vol_mesh_coords', shape=local_mesh_size)
        self.add_input('current_density')
        self.add_input('fill_factor')

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

        mesh_coords = inputs['vol_mesh_coords']
        state = outputs['state']
        residual = residuals['state']

        solver.setMeshCoordinates(mesh_coords);
        solver.calcResidual(state, residual);


    def solve_nonlinear(self, inputs, outputs):
        """
        Converge the state
        """
        solver = self.options['solver']

        mesh_coords = inputs['vol_mesh_coords']
        state = outputs['state']

        solver.setMeshCoordinates(mesh_coords);
        solver.calcState(state);

    def linearize(self, inputs, outputs, residuals):
        """
        Perform assembly of Jacobians/linear forms?
            Only makes sense if this is always called before apply_linear
        """
        solver = self.options['solver']

        solver.setState(outputs['state'])


    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):

        solver = self.options['solver']

        if mode == 'rev':
            if 'state' in d_residuals: 
                if 'state' in d_outputs: 
                    d_outputs['state'] = solver.multStateJacTranspose(d_residuals['state'])
        
                if 'vol_mesh_coords' in d_inputs: 
                    d_inputs['vol_mesh_coords'] = solver.multMeshJacTranspose(d_residuals['state'])

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
        

class omMachFunctionals(om.ExplicitComponent):
    """OpenMDAO component that computes functionals given the state variables"""
    def initialize(self):
        self.options.declare('solver', types=MachSolver)
        # self.options['distributed'] = True

    def setup(self):
        solver = self.options['solver']

        if self.comm.rank == 0:
            print('Adding functional inputs')

        local_mesh_size = solver.getMeshSize()
        self.add_input('vol_mesh_coords', shape=local_mesh_size)

        local_state_size = solver.getStateSize()
        self.add_input('state', shape=local_state_size)
        self.add_input('current_density')
        self.add_input('fill_factor')

        if self.comm.rank == 0:
            print('Adding functional outputs')

        self.add_output('func')

        #self.declare_partials(of='func', wrt='*')

    def compute(self, inputs, outputs):
        solver = self.options['solver']


        mesh_coords = inputs['vol_mesh_coords']
        state = inputs['state']

        solver.setMeshCoordinates(mesh_coords)
        outputs['func'] = solver.calcFunctional(state, "drag")
