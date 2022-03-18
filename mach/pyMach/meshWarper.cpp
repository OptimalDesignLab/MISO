#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <mpi4py/mpi4py.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mfem.hpp"
#include "mpi.h"
#include "nlohmann/json.hpp"

#include "abstract_solver.hpp"
#include "mesh_warper.hpp"
#include "mpi_comm.hpp"

namespace py = pybind11;
using namespace mach;

namespace
{
mfem::Vector npBufferToMFEMVector(const py::array_t<double> &buffer)
{
   auto info = buffer.request();
   /* Some sanity checks ... */
   if (info.format != py::format_descriptor<double>::format())
   {
      throw std::runtime_error(
          "Incompatible format:\n"
          "\texpected a double array!");
   }
   if (info.ndim != 1)
   {
      throw std::runtime_error(
          "Incompatible dimensions:\n"
          "\texpected a 1D array!");
   }
   return {static_cast<double *>(info.ptr), static_cast<int>(info.shape[0])};
}

template <typename T>
mfem::Array<T> npBufferToMFEMArray(const py::array_t<T> &buffer)
{
   auto info = buffer.request();
   /* Some sanity checks ... */
   if (info.format != py::format_descriptor<T>::format())
   {
      throw std::runtime_error("Incompatible format!");
   }
   if (info.ndim != 1)
   {
      throw std::runtime_error(
          "Incompatible dimensions:\n"
          "\texpected a 1D array!");
   }
   return {static_cast<T *>(info.ptr), static_cast<int>(info.shape[0])};
}

}  // anonymous namespace

void initMeshWarper(py::module &m)
{
   /// imports mpi4py's C interface
   if (import_mpi4py() < 0)
   {
      return;
   }

   py::class_<MeshWarper, AbstractSolver2>(m, "MeshWarper")
       .def(py::init(
                [](const std::string &opt_file_name, mpi_comm comm)
                {
                   nlohmann::json json_options;
                   std::ifstream options_file(opt_file_name);
                   options_file >> json_options;
                   return std::make_unique<MeshWarper>(comm, json_options);
                }),
            py::arg("opt_file_name"),
            py::arg("comm") = mpi_comm(MPI_COMM_WORLD))
       .def(py::init(
                [](const nlohmann::json &json_options, mpi_comm comm)
                { return std::make_unique<MeshWarper>(comm, json_options); }),
            py::arg("json_options"),
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
