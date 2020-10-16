import numpy as np
import openmdao.api as om

from .pyMach import Mesh, Vector, MachSolver
from .pyMach import MeshMovement

class omMeshMove(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('solver', types=MachSolver)
        # self.options['distributed'] = True

    def setup(self):
        solver = self.options['solver']

        local_mesh_size = solver.getMeshSize()

        self.add_input('surf_mesh_disp', shape=local_mesh_size)
        self.add_output('vol_mesh_disp', shape=local_mesh_size)

        #self.declare_partials(of='state', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute the residual
        """
        solver = self.options['solver']

        surf_mesh_disp = inputs['surf_mesh_disp']
        vol_mesh_disp = outputs['vol_mesh_disp']
        
        state = solver.getNewField(vol_mesh_disp)
        residual = solver.getNewField(residuals['vol_mesh_disp'])

        # TODO: change these methods in machSolver to support numpy array 
        # as argument and do the conversion internally
        solver.calcResidual(state, residual)


    def solve_nonlinear(self, inputs, outputs):
        """
        Converge the state
        """
        solver = self.options['solver']

        surf_mesh_disp = inputs['surf_mesh_disp']
        vol_mesh_disp = outputs['vol_mesh_disp']

        # Get fields for the surface displacement and volume displacement
        initial_condition = solver.getNewField(surf_mesh_disp)
        state = solver.getNewField(vol_mesh_disp)

        # solver.printField("state", state, "state")
        # TODO: change these methods in machSolver to support numpy array 
        # as argument and do the conversion internally
        solver.setInitialField(state, initial_condition)
        solver.solveForState(state)
        solver.printField("state2", state, "state")

        mesh = Mesh(model_file='data/testOMMeshMovement/cyl.egads', mesh_file='data/testOMMeshMovement/cyl.smb')
        print(type(state))
        mesh.addDisplacement(state)
        mesh.PrintVTU("testmeshmove")


        # solver.printFields("state_post_solve", [state, uex], ["state", "uex"])

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
        
                if 'vol_mesh_disp' in d_inputs: 
                    d_inputs['vol_mesh_disp'] = solver.multMeshJacTranspose(d_residuals['state'])

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