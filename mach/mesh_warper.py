import numpy as np
import openmdao.api as om

from .pyMach import MeshWarper

class MachMeshWarper(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("warper", types=MeshWarper, recordable=False)

    def setup(self):
        warper = self.options['warper']

        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surf_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surf_coords)

        local_vol_mesh_size = warper.getVolumeCoordsSize()
        self.init_vol_coords = np.empty(local_vol_mesh_size)
        warper.getInitialVolumeCoords(self.init_vol_coords)

        self.add_input('surf_mesh_coords', val=surf_coords, distributed=True, tags=["mphys_coordinates"])
        self.add_output('vol_mesh_coords', val=self.init_vol_coords, distributed=True, tags=["mphys_coupling"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        warper = self.options['warper']

        surface_coords = inputs["surf_mesh_coords"]

        indices = np.empty(surface_coords.size, dtype=np.int32)
        warper.getSurfCoordIndices(indices)

        volume_coords = outputs["vol_mesh_coords"]
        volume_coords[indices] = surface_coords[:]

        warper.calcResidual(volume_coords, residuals["vol_mesh_coords"])


    def solve_nonlinear(self, inputs, outputs):
        warper = self.options['warper']

        surface_coords = inputs["surf_mesh_coords"]

        indices = np.empty(surface_coords.size, dtype=np.int32)
        warper.getSurfCoordIndices(indices)

        volume_coords = outputs["vol_mesh_coords"]
        volume_coords[indices] = surface_coords[:]

        warper.solveForState(volume_coords)

    def linearize(self, inputs, outputs, residuals):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        pass

    def solve_linear(self, d_outputs, d_residuals, mode):
        pass
