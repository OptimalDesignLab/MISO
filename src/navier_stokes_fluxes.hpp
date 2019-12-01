/// Functions related to Euler equations

#ifndef MACH_EULER_FLUXES
#define MACH_EULER_FLUXES

#include <algorithm> // std::max
#include "utils.hpp"

namespace mach
{
/// For constants related to the Navier-Stokes equations
namespace navierstokes
{
/// heat capacity ratio for air
const double gamma = 1.4;
/// ratio minus one
const double gami = gamma - 1.0;
/// gas constant
const double R = 287;
/// specfic heat for constant volume
const double cv = R / gami;
/// use constant `kappa` and `mu` for time being
const double mu = 1.81e-05;
const double kappa = 0.026;
} // namespace navierstokes

/// Applies the matrix `Cij` to `dW/dX`
/// \param[in] i - index `i` in `Cij` matrix
/// \param[in] j - index `j` in `Cij` matrix is calculated
/// \param[in] vec - the vector being multiplied
/// \param[out] mat_vec - the result of the operation
template <typename xdouble, int dim>
void applyViscousScaling(int i, int j, const xdouble *q, const xdouble *vec,
                         xdouble *mat_vec)
{
   using namespace navierstokes;
   xdouble E = q[dim + 2] / q[0];
   for (int p = 0; p < dim; ++p)
   {
      e -= 0.5 * (q[p + 1] / q[0]) * (q[p + 1] / q[0]);
   }
   e += E;
   T = e / cv;
   // apply diagonal block matrices (Cij; i=j) on `vec`
  if (i == j)
   {
      // get all entries of `mat_vec` except last one
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[k + 1] += T * mu * (vec[k + 1] +
                                    ((q[k + 1] / q[0]) * vec[dim + 1]));
      }
      mat_vec[i + 1] += (1 / 3) * T * mu * (vec[i + 1] + 
                                           ((q[i + 1] / q[0]) * vec[dim + 1]));
      // get last entry of `mat_vec`
      for (int p = 0; p < dim; ++p)
      {
         mat_vec[dim + 1] += T * mu * (q[p + 1] / q[0]) * (vec[p + 1] +
                                             ((q[p + 1] / q[0]) * vec[dim + 1]));
      }
      mat_vec[dim + 1] += (1 / 3) * T * mu * (q[i + 1] / q[0]) * (vec[i + 1] +
                                             ((q[i + 1] / q[0]) * vec[dim + 1]));
      mat_vec[dim + 1] += T * T * kappa * vec[dim + 1];
   }
   // apply non-diagonal block matrices (Cij; i!=j) on `vec`
   else
   {
      // get all non-zero entries of `mat_vec` except last 
      mat_vec[i + 1] -= (2 / 3) * T * mu * (vec[j + 1] +
                                     ((q[j + 1] / q[0]) * vec[dim + 1]));
      mat_vec[j + 1] += T * mu * (vec[i + 1] + ((q[i + 1] / q[0]) * vec[dim + 1]));
      // get last entry of `mat_vec`
      mat_vec[dim + 1]  =(q[j + 1] / q[0]) * vec[i + 1];
      mat_vec[dim + 1] += (q[i + 1] / q[0]) * ((-(2 / 3) * vec[j + 1]) +
                                               ((1 / 3) * (q[j + 1] / q[0]) * vec[dim + 1]));
      mat_vec[dim + 1] *= T * mu;
   }
}

} // namespace mach

#endif