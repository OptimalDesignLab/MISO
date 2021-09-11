#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include "mfem.hpp"

namespace py = pybind11;

using namespace mfem;

void initVector(py::module &m)
{
   py::class_<Vector>(m, "Vector", py::buffer_protocol())
      /// method to allow Vector object to be constructable from a numpy array
      .def(py::init([](py::array_t<double, py::array::f_style> b)
      {

         /* Request a buffer descriptor from Python */
         py::buffer_info info = b.request();

         /* Some sanity checks ... */
         if (info.format != py::format_descriptor<double>::format())
            throw std::runtime_error("Incompatible format:\n"
                                       "\texpected a double array!");

         if (info.ndim != 1)
            throw std::runtime_error("Incompatible dimensions:\n"
                                       "\texpected a 1D array!");

         return new Vector((double*)info.ptr, info.shape[0]);
      }))

      /// method that allows a Vector to be converted to a numpy array
      .def_buffer([](Vector &self) -> py::buffer_info
      {
         return py::buffer_info
         (
            self.HostReadWrite(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            1,
            {self.Size()},
            {sizeof(double)}
         );
      })
      /// construct Vector of given size initialized to zero
      .def(py::init([](int size)
      {
         auto self = new Vector(size);
         *self = 0.0;
         return self;
      }))

      .def("normL2", &Vector::Norml2)

      /// construct unintialized Vector
      .def(py::init<>())

      /// return the size
      .def("size", &Vector::Size)

      /// set the size of the vector
      .def("setSize", (void (Vector::*)(int))&Vector::SetSize)

      /// methods to allow indexing vector
      .def("__getitem__", [](Vector& self, size_t index)
      {
         return &self(index);
      }, py::return_value_policy::reference)

      .def("__setitem__", [](Vector& self, size_t index, double val)
      {
         self(index) = val;
      }, py::return_value_policy::reference)

      /// method allowing printing a vector
      .def("__repr__", [](Vector& self)
      {
         std::ostringstream out;
         out << '(';
         self.Print(out, self.Size());
         out.seekp(-1, std::ios_base::end);
         out << ')';
         return out.str();
      })
      ;
}