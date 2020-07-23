#ifndef MPI4PY_COMM
#define MPI4PY_COMM

#include <pybind11/pybind11.h>
#include <mpi.h>
#include <mpi4py/mpi4py.h>

/**
 * Code to cast between MPI communicators and mpi4py communicators
 * taken from:
 * https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
 */

namespace py = pybind11;

struct mpi4py_comm
{
public:
   mpi4py_comm() = default;
   mpi4py_comm(MPI_Comm value) : value(value) {}
   inline operator MPI_Comm () { return value; }

   MPI_Comm value;
};


namespace pybind11
{

namespace detail
{

template <>
struct type_caster<mpi4py_comm>
{
public:
   PYBIND11_TYPE_CASTER(mpi4py_comm, _("mpi4py_comm"));

   // Python -> C++
   bool load(handle src, bool)
   {
      PyObject *py_src = src.ptr();

      // Check that we have been passed an mpi4py communicator
      if (PyObject_TypeCheck(py_src, &PyMPIComm_Type))
      {
         // Convert to regular MPI communicator
         value.value = *PyMPIComm_Get(py_src);
      } else
      {
         return false;
      }

      return !PyErr_Occurred();
   }

   // C++ -> Python
   static handle cast(mpi4py_comm src,
                      return_value_policy /* policy */,
                      handle /* parent */)
   {
      // Create an mpi4py handle
      return PyMPIComm_New(src.value);
   }
};

} // namespace pybind11::detail

} // namespace pybind11

#endif
