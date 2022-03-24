import numpy as np
import openmdao.api as om

from .pyMach import MeshWarper

class MachMeshWarper(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("warper", types=MeshWarper, recordable=False)

    def setup(self):
        warper = self.options["warper"]

        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surf_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surf_coords)

        local_vol_mesh_size = warper.getVolumeCoordsSize()
        self.init_vol_coords = np.empty(local_vol_mesh_size)
        warper.getInitialVolumeCoords(self.init_vol_coords)

        self.surf_indices = np.empty(surf_coords.size, dtype=np.int32)
        warper.getSurfCoordIndices(self.surf_indices)

        self.add_input("surf_mesh_coords", val=surf_coords, distributed=True, tags=["mphys_coordinates"])
        self.add_output("vol_mesh_coords", val=self.init_vol_coords, distributed=True, tags=["mphys_coupling"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        warper = self.options["warper"]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict["state"] = outputs["vol_mesh_coords"]
        warper.calcResidual(input_dict, residuals["vol_mesh_coords"])


    def solve_nonlinear(self, inputs, outputs):
        warper = self.options["warper"]

        surface_coords = inputs["surf_mesh_coords"]

        volume_coords = outputs["vol_mesh_coords"]
        volume_coords[self.surf_indices] = surface_coords[:]

        warper.solveForState(volume_coords)

    def linearize(self, inputs, outputs, residuals):
        # cache inputs
        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict["state"] = outputs["vol_mesh_coords"]
        self.linear_inputs = input_dict

        warper = self.options["warper"]
        warper.linearize(input_dict)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        warper = self.options["warper"]
        if mode == "fwd":
            if "vol_mesh_coords" in d_residuals:
                if "vol_mesh_coords" in d_outputs:
                    warper.jacobianVectorProduct(d_outputs["vol_mesh_coords"],
                                                 "state",
                                                 d_residuals["vol_mesh_coords"])
                if "surf_mesh_coords" in d_inputs:
                    warper.jacobianVectorProduct(d_inputs["surf_mesh_coords"],
                                                 "surf_mesh_coords",
                                                 d_residuals["vol_mesh_coords"])

        elif mode == "rev":
            if "vol_mesh_coords" in d_residuals:
                if "vol_mesh_coords" in d_outputs:
                    warper.vectorJacobianProduct(d_residuals["vol_mesh_coords"],
                                                 "state",
                                                 d_outputs["vol_mesh_coords"])
                if "surf_mesh_coords" in d_inputs:
                    warper.jacobianVectorProduct(d_residuals["vol_mesh_coords"],
                                                 "surf_mesh_coords",
                                                 d_inputs["surf_mesh_coords"])

    def solve_linear(self, d_outputs, d_residuals, mode):
        # print("adj before:", d_residuals["vol_mesh_coords"])
        if mode == "fwd":
            # pass
            # print("solver fwd")
            raise NotImplementedError("forward mode requested but not implemented")

        if mode == "rev":
            # print("rev mode!")
            warper = self.options["warper"]
            input_dict = self.linear_inputs
            warper.solveForAdjoint(input_dict,
                                   d_outputs["vol_mesh_coords"],
                                   d_residuals["vol_mesh_coords"])

        # print("adj after:", d_residuals["vol_mesh_coords"])

