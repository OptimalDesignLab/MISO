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

// To do: we may not need this, as it is same as for inviscid case
/// Convert conservative variables `q` to entropy variables `w`
/// \param[in] q - conservative variables that we want to convert from
/// \param[out] w - entropy variables we want to convert to
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcEntropyVars(const xdouble *q, xdouble *w)
{
   xdouble u[dim];
   for (int i = 0; i < dim; ++i)
   {
      u[i] = q[i+1]/q[0];
   }
   xdouble p = pressure<xdouble,dim>(q);
   xdouble s = log(p/pow(q[0],euler::gamma));
   xdouble fac = 1.0/p;
   w[0] = (euler::gamma-s)/euler::gami - 0.5*dot<xdouble,dim>(u,u)*fac*q[0];
   for (int i = 0; i < dim; ++i)
   {
      w[i+1] = q[i+1]*fac;
   }
   w[dim+1] = -q[0]*fac;
}

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