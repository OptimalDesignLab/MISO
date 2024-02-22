import numpy as np
import openmdao.api as om

from .pyMach import MeshWarper

class MachMeshWarper(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("warper", types=MeshWarper, recordable=False)

    def setup(self):
        warper = self.options["warper"]

        # hold map of vector-valued I/O names -> contiguous vectors to pass to Mach
        self.vectors = dict()

        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surface_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surface_coords)

        local_vol_mesh_size = warper.getVolumeCoordsSize()
        volume_coords = np.empty(local_vol_mesh_size)
        warper.getInitialVolumeCoords(volume_coords)

        self.surf_indices = np.empty(surface_coords.size, dtype=np.int32)
        warper.getSurfCoordIndices(self.surf_indices)

        self.add_input("surf_mesh_coords",
                       val=surface_coords,
                    #    distributed=True,
                       tags=["mphys_coordinates"])
        self.vectors["surf_mesh_coords"] = surface_coords

        self.add_output("vol_mesh_coords",
                        val=volume_coords,
                        distributed=True,
                        tags=["mphys_coupling"])
        self.vectors["vol_mesh_coords"] = volume_coords
        self.vectors["vol_mesh_coords_res"] = np.empty_like(volume_coords)

    def apply_nonlinear(self, inputs, outputs, residuals):
        warper = self.options["warper"]

        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                self.vectors[input][:] = inputs[input][:]

        # Copy vector outputs into internal contiguous data buffers
        for output in outputs:
            if output in self.vectors:
                self.vectors[output][:] = outputs[output][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(self.vectors)        
        input_dict["state"] = self.vectors["vol_mesh_coords"]

        residual = self.vectors["vol_mesh_coords_res"]
        warper.calcResidual(input_dict, residual)
        residuals["vol_mesh_coords"][:] = residual[:]

    def solve_nonlinear(self, inputs, outputs):
        warper = self.options["warper"]

        surface_coords = inputs["surf_mesh_coords"]

        volume_coords = self.vectors["vol_mesh_coords"]
        volume_coords[self.surf_indices] = surface_coords[:]
        warper.solveForState(volume_coords)
        outputs["vol_mesh_coords"][:] = volume_coords[:]

    def linearize(self, inputs, outputs, residuals):
        # cache inputs
        # Copy vector inputs into internal contiguous data buffers
        for input in inputs:
            if input in self.vectors:
                self.vectors[input][:] = inputs[input][:]

        # Copy vector outputs into internal contiguous data buffers
        for output in outputs:
            if output in self.vectors:
                self.vectors[output][:] = outputs[output][:]

        input_dict = dict(zip(inputs.keys(), inputs.values()))
        input_dict.update(self.vectors)        
        input_dict["state"] = self.vectors["vol_mesh_coords"]
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
        if mode == "fwd":
            raise NotImplementedError("forward mode requested but not implemented")

        if mode == "rev":
            warper = self.options["warper"]
            input_dict = self.linear_inputs
            warper.solveForAdjoint(input_dict,
                                   d_outputs["vol_mesh_coords"],
                                   d_residuals["vol_mesh_coords"])
