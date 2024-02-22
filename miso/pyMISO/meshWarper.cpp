#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>

#include "mfem.hpp"
#include "mpi.h"
#include "nlohmann/json.hpp"

#include "abstract_solver.hpp"
#include "mesh_warper.hpp"
#include "mpi_comm.hpp"
#include "py_miso_utils.hpp"

namespace py = pybind11;
using namespace miso;

void initMeshWarper(py::module &m)
{
   /// imports mpi4py's C interface
   if (import_mpi4py() < 0)
   {
      return;
   }

   py::class_<MeshWarper, AbstractSolver2>(m, "MeshWarper")
       .def(py::init(
                [](const nlohmann::json &warper_options, mpi_comm comm)
                { return std::make_unique<MeshWarper>(comm, warper_options); }),
            py::arg("warper_options"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))

       .def("getSurfaceCoordsSize", &MeshWarper::getSurfaceCoordsSize)
       .def("getInitialSurfaceCoords",
            [](MeshWarper &self, const py::array_t<double> &surface_coords)
            {
               auto coord_vec = npBufferToMFEMVector(surface_coords);
               self.getInitialSurfaceCoords(coord_vec);
            })
       .def("getVolumeCoordsSize", &MeshWarper::getVolumeCoordsSize)
       .def("getInitialVolumeCoords",
            [](MeshWarper &self, const py::array_t<double> &volume_coords)
            {
               auto coord_vec = npBufferToMFEMVector(volume_coords);
               self.getInitialVolumeCoords(coord_vec);
            })
       .def("getSurfCoordIndices",
            [](MeshWarper &self, const py::array_t<int> &indices)
            {
               auto indices_array = npBufferToMFEMArray(indices);
               self.getSurfCoordIndices(indices_array);
            });
}
