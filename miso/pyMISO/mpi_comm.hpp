#ifndef MPI4PY_COMM
#define MPI4PY_COMM

#include <pybind11/pybind11.h>
#include <mpi.h>
#include <mpi4py/mpi4py.h>

/**
 * Code to cast between MPI communicators and mpi4py communicators
 * inspired by:
 * https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
 */

namespace py = pybind11;

struct mpi_comm
{
public:
   mpi_comm() = default;
   mpi_comm(MPI_Comm comm) : comm(comm) { }
   inline operator MPI_Comm() { return comm; }

   MPI_Comm comm = MPI_COMM_WORLD;
};

namespace pybind11
{
namespace detail
{
template <>
struct type_caster<mpi_comm>
{
public:
   PYBIND11_TYPE_CASTER(mpi_comm, _("mpi_comm"));

   // Python -> C++
   bool load(handle src, bool)
   {
      auto *py_src = src.ptr();

      // Check that we have been passed an mpi4py communicator
      if (PyObject_TypeCheck(py_src, &PyMPIComm_Type))
      {
         // Convert to regular MPI communicator
         value.comm = *PyMPIComm_Get(py_src);
      }
      else
      {
         return false;
      }

      return !PyErr_Occurred();
   }

   // C++ -> Python
   static handle cast(mpi_comm src,
                      return_value_policy /* policy */,
                      handle /* parent */)
   {
      // Create an mpi4py handle
      return PyMPIComm_New(src.comm);
   }
};

}  // namespace detail

}  // namespace pybind11

#endif
