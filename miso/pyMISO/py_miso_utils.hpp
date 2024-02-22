#ifndef MISO_PYMISO_UTILS
#define MISO_PYMISO_UTILS

#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pybind11_json.hpp"

#include "mfem.hpp"

#include "miso_input.hpp"

namespace miso
{
inline double *npBufferToDoubleArray(const py::array_t<double> &buffer,
                                     std::vector<pybind11::ssize_t> &shape,
                                     int expected_dim = 1)
{
   auto info = buffer.request();

   /* Some sanity checks ... */
   if (info.format != py::format_descriptor<double>::format())
   {
      throw std::runtime_error(
          "Incompatible format:\n"
          "\texpected a double array!");
   }
   if (info.ndim != expected_dim)
   {
      throw std::runtime_error(
          "Incompatible dimensions:\n"
          "\texpected a 1D array!");
   }

   shape = std::move(info.shape);
   return static_cast<double *>(info.ptr);
}

inline double *npBufferToDoubleArray(const py::array_t<double> &buffer,
                                     int expected_dim = 1)
{
   std::vector<pybind11::ssize_t> shape;
   return npBufferToDoubleArray(buffer, shape, expected_dim);
}

inline mfem::Vector npBufferToMFEMVector(const py::array_t<double> &buffer)
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
   if (info.strides[0] != sizeof(double))
   {
      throw std::runtime_error(
          "Incompatible stride:\n"
          "\texpected a contiguous array!");
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
   if (info.strides[0] != sizeof(T))
   {
      throw std::runtime_error(
          "Incompatible stride:\n"
          "\texpected a contiguous array!");
   }
   return {static_cast<T *>(info.ptr), static_cast<int>(info.shape[0])};
}

inline MISOInputs pyDictToMISOInputs(const py::dict &py_inputs)
{
   MISOInputs inputs(py_inputs.size());

   for (const auto &input : py_inputs)
   {
      const auto &key = input.first.cast<std::string>();

      const char *val_name = input.second.ptr()->ob_type->tp_name;
      bool is_number = strncmp("float", val_name, 5) == 0 ||
                       strncmp("int", val_name, 3) == 0;
      if (is_number)
      {
         const auto &value = input.second.cast<double>();
         inputs.emplace(key, value);
      }
      else
      {
         const auto &value_buffer = input.second.cast<py::array_t<double>>();
         auto vector = npBufferToMFEMVector(value_buffer);

         if (vector.Size() == 1)
         {
            inputs.emplace(key, vector(0));
         }
         else
         {
            inputs.emplace(key, vector);
         }
      }
   }
   return inputs;
}

}  // namespace miso

#endif
