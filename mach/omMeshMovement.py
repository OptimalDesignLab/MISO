import numpy as np
import openmdao.api as om

from .pyMach import MeshWarper

class omMeshMove(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("warper", types=MeshWarper, recordable=False)

    def setup(self):
        warper = self.options['warper']

        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surf_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surf_coords)

        local_vol_mesh_size = warper.getVolumeCoordsSize()
        vol_coords = np.empty(local_vol_mesh_size)
        warper.getInitialVolumeCoords(vol_coords)

        self.add_input('surf_mesh_coords', val=surf_coords, distributed=True, tags=["mphys_coordinates"])
        self.add_output('vol_mesh_coords', val=vol_coords, distributed=True, tags=["mphys_coupling"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        warper = self.options['warper']

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(dict(zip(outputs.keys(), outputs.values())))

        residual = residuals["vol_mesh_coords"]
        warper.calcResidual(input_dict, residual)


    def solve_nonlinear(self, inputs, outputs):
        solver = self.options["solver"]
        mesh_size = solver.getFieldSize("mesh_coords")
        mesh_coords = np.zeros(mesh_size)
        solver.getField("mesh_coords", mesh_coords)
        outputs["vol_mesh_coords"] = mesh_coords

    def linearize(self, inputs, outputs, residuals):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        pass

    def solve_linear(self, d_outputs, d_residuals, mode):
        pass