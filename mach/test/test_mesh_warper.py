import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from mach import MachMeshWarper, MeshWarper


warper_options = {
    "mesh": {
        "file": "data/cyl.mesh"
    },
    "space-dis": {
        "degree": 1,
        "basis-type": "H1"
    },
    "nonlin-solver": {
        "type": "newton",
        "printlevel": 3,
        "maxiter": 50,
        "reltol": 1e-10,
        "abstol": 1e-12
    },
    "lin-solver": {
        "type": "pcg",
        "printlevel": 1,
        "maxiter": 100,
        "abstol": 1e-14,
        "reltol": 1e-14
    },
    "adj-solver": {
        "type": "gmres",
        "printlevel": 1,
        "maxiter": 100,
        "abstol": 1e-14,
        "reltol": 1e-14
    },
    "lin-prec": {
        "printlevel": -1
    },
    "bcs": {
      "essential": "all"
    }
}

class TestMachMeshWarper(unittest.TestCase):
    def test_forward(self):
        problem = om.Problem()
        model = problem.model
        ivc = model.add_subsystem("ivc", om.IndepVarComp())

        warper = MeshWarper(warper_options, problem.comm)
        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surf_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surf_coords)

        for i in range(0, surf_coords.size, 3):
            surf_coords[i + 0] += 1.0
            surf_coords[i + 1] += 1.0
            surf_coords[i + 2] += 0.0

        ivc.add_output("surf_mesh_coords", val=surf_coords)
        model.add_subsystem("vol_mesh_move", MachMeshWarper(warper=warper))

        model.connect("ivc.surf_mesh_coords", "vol_mesh_move.surf_mesh_coords")

        problem.setup()
        problem.run_model()

        local_vol_mesh_size = warper.getVolumeCoordsSize()
        init_vol_coords = np.empty(local_vol_mesh_size)
        warper.getInitialVolumeCoords(init_vol_coords)

        volume_coords = problem.get_val("vol_mesh_move.vol_mesh_coords")

        for i in range(0, volume_coords.size, 3):
            self.assertAlmostEqual(init_vol_coords[i + 0] + 1.0, volume_coords[i + 0])
            self.assertAlmostEqual(init_vol_coords[i + 1] + 1.0, volume_coords[i + 1])
            self.assertAlmostEqual(init_vol_coords[i + 2] + 0.0, volume_coords[i + 2])

    def test_partials(self):
        problem = om.Problem()
        model = problem.model
        ivc = model.add_subsystem("ivc", om.IndepVarComp())

        warper = MeshWarper(warper_options, problem.comm)
        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surf_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surf_coords)

        for i in range(0, surf_coords.size, 3):
            surf_coords[i + 0] += 1.0
            surf_coords[i + 1] += 1.0
            surf_coords[i + 2] += 0.0

        ivc.add_output("surf_mesh_coords", val=surf_coords)
        model.add_subsystem("vol_mesh_move", MachMeshWarper(warper=warper))

        model.connect("ivc.surf_mesh_coords", "vol_mesh_move.surf_mesh_coords")

        problem.setup()
        problem.run_model()

        data = problem.check_partials()
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    def test_totals(self):
        problem = om.Problem()
        model = problem.model
        problem.model.nonlinear_solver = om.NonlinearBlockGS()
        
        ivc = model.add_subsystem("ivc", om.IndepVarComp())

        warper = MeshWarper(warper_options, problem.comm)
        local_surf_mesh_size = warper.getSurfaceCoordsSize()
        surf_coords = np.empty(local_surf_mesh_size)
        warper.getInitialSurfaceCoords(surf_coords)

        for i in range(0, surf_coords.size, 3):
            surf_coords[i + 0] += 1.0
            surf_coords[i + 1] += 1.0
            surf_coords[i + 2] += 0.0

        ivc.add_output("surf_mesh_coords", val=surf_coords)
        model.add_subsystem("vol_mesh_move", MachMeshWarper(warper=warper))

        model.connect("ivc.surf_mesh_coords", "vol_mesh_move.surf_mesh_coords")

        problem.setup(mode="rev")
        # om.n2(problem)
        problem.run_model()

        data = problem.check_totals(of=['vol_mesh_move.vol_mesh_coords'], wrt=['vol_mesh_move.surf_mesh_coords'])
        assert_check_totals(data, atol=1.e-6, rtol=1.e-6)

if __name__ == "__main__":
    unittest.main()