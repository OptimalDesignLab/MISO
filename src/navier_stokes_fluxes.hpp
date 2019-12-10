/// Functions related to Navier-Stokes equations
#ifndef MACH_NAVIER_STOKES_FLUXES
#define MACH_NAVIER_STOKES_FLUXES

#include <algorithm> // std::max
#include "utils.hpp"
#include "euler_fluxes.hpp"

namespace mach
{

/// For constants related to the Navier-Stokes equations
/// \todo Some of these constants really belong in the euler namespace (e.g. R and cv)
namespace navierstokes
{
/// Ratio of Sutherland's constant and free-stream temperature
const double ST = 198.6d0/460.d0;
} // namespace navierstokes

/// Returns the dynamic viscosity based on Sutherland's law
/// \param[in] q - state used to define the viscosity
/// \returns mu - nondimensionalized viscosity
/// \note This assumes the free-stream temperature given by navierstokes::ST
xdouble calcSutherlandViscosity(const xdouble *q)
{
   xdouble a = sqrt(euler::gamma*pressure<xdouble,dim>(q)/q[0]);
   xdouble a2 = a*a;
   xdouble a3 = a*a*a;
   return a3*(1.0 + navierstokes::ST)/(a2 + navierstokes::ST);
}

/// Compute the product \f$ \sum_{d'=0}^{dim} C_{d,d'} D_{d'} w \f$
/// \param[in] d - desired space component of the flux
/// \param[in] Re - Reynolds number
/// \param[in] Pr - Prandtl number
/// \param[in] q - state used to evaluate the \f$ C_d,d'} \f$ matrices
/// \param[in] Dw - derivatives of entropy varaibles, stored column-major
/// \param[out] mat_vec - stores the resulting matrix vector product
template <typename xdouble, int dim>
void applyViscousScaling(int d, double Re, double Pr, const xdouble *q,
                         const xdouble *Dw, xdouble *mat_vec)
{
   for (int k = 0; k < dim+2; ++k) {
      mat_vec[k] = 0.0;
   }
   xdouble mu = calcSutherlandViscosity<xdouble, dim>(q)/Re;
   for (int d2 = 0; d2 < dim; ++d2) {
      applyCijMatrix(d, d2, mu, Pr, q, Dq+(d2*(dim+2)), mat_vec);
   }
}

/// Applies the matrix `Cij` to `dW/dX`
/// \param[in] i - index `i` in `Cij` matrix
/// \param[in] j - index `j` in `Cij` matrix is calculated
/// \param[in] mu - nondimensionalized dynamic viscosity 
/// \param[in] Pr - Prandtl number
/// \param[in] q - state used to evaluate `Cij` matrix
/// \param[in] vec - the vector being multiplied
/// \param[in,out] mat_vec - the result of the operation
/// \warning `mat_vec` is added to and *not* initialized to zero
/// \note this function has been nondimensionalized
template <typename xdouble, int dim>
void applyCijMatrix(int i, int j, const xdouble mu, const xdouble Pr,
                    const xdouble *q, const xdouble *vec, xdouble *mat_vec)
{
   // define the velocity components
   xdouble u[dim];
   for (int i = 0; i < dim; ++i)
   {
      u[i] = q[i+1]/q[0];
   }
   xdouble RTmu = pressure<xdouble,dim>(q)*mu/q[0];
   xdouble RT2k = (RTmu*RTmu)*euler::gamma/(mu*euler::gami*Pr);
   // apply diagonal block matrices (Cij; i=j) on `vec`
   if (i == j)
   {
      // get all entries of `mat_vec` except last one
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[k + 1] += RTmu * (vec[k + 1] + u[k]*vec[dim + 1]));
      }
      mat_vec[i + 1] += (1 / 3) * RTmu * (vec[i + 1] + u[i]*vec[dim+1]);
      // get last entry of `mat_vec`
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[dim + 1] += RTmu * u[k] * (vec[k + 1] + u[k] * vec[dim + 1]);
      }
      mat_vec[dim + 1] += (1 / 3) * RTmu * u[i] * (vec[i + 1] + 
                                                   u[i] * vec[dim + 1]);
      mat_vec[dim + 1] += RT2k * vec[dim + 1];
   }
   else // apply off-diagonal block matrices (Cij; i!=j) on `vec`
   {
      // get all non-zero entries of `mat_vec` except last
      mat_vec[i + 1] -= (2 / 3) * RTmu * (vec[j + 1] + u[j] * vec[dim + 1]);
      mat_vec[j + 1] += RTmu * (vec[i + 1] + u[i] * vec[dim + 1]);
      // get last entry of `mat_vec`
      mat_vec[dim + 1] += RTmu * (u[j] * vec[i + 1] 
                                  - (2 / 3) * u[i] * vec[j + 1]
                                  + (1 / 3) * u[i] * u[j] * vec[dim + 1]));
   }
}

} // namespace mach

#endif