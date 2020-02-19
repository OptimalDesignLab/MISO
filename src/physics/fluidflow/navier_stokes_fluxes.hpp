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
const double ST = 198.60/460.0;
} // namespace navierstokes

/// Returns the dynamic viscosity based on Sutherland's law
/// \param[in] q - state used to define the viscosity
/// \returns mu - **nondimensionalized** viscosity
/// \note This assumes the free-stream temperature given by navierstokes::ST
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSutherlandViscosity(const xdouble *q)
{
   xdouble a = sqrt(euler::gamma*pressure<xdouble,dim>(q)/q[0]);
   xdouble a2 = a*a;
   xdouble a3 = a*a*a;
   return a3*(1.0 + navierstokes::ST)/(a2 + navierstokes::ST);
}

template <typename xdouble, int dim>
void setZeroNormalDeriv(const xdouble *dir, const xdouble *Dw, xdouble *Dwt)
{
   xdouble nrm[dim];
   xdouble fac = 1.0 / sqrt(dot<xdouble, dim>(dir, dir));
   int num_state = dim+2;
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir[i] * fac;
   }
   for (int s = 0; s < dim+2; ++s)
   {
      xdouble Dw_nrm = 0.0;
      for (int i = 0; i < dim; ++i)
      {
         Dw_nrm += Dw[s + i*num_state]*nrm[i];
      }
      for (int i = 0; i < dim; ++i)
      {
         Dwt[s + i*num_state] = Dw[s + i*num_state] - nrm[i]*Dw_nrm;
      }
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
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \warning `mat_vec` is added to and *not* initialized to zero
/// \note this function has been nondimensionalized
template <typename xdouble, int dim>
void applyCijMatrix(int i, int j, const xdouble mu, const xdouble Pr,
                    const xdouble *q, const xdouble *vec, xdouble *mat_vec)
{
   using namespace std;
   // define the velocity components
   xdouble u[dim];
   for (int k = 0; k < dim; ++k)
   {
      u[k] = q[k+1]/q[0];
   }
   xdouble RTmu = pressure<xdouble,dim>(q)/q[0];
   xdouble RT2k = RTmu*RTmu*euler::gamma*mu/(euler::gami*Pr);
   RTmu *= mu;
   // apply diagonal block matrices (Cij; i=j) on `vec`
   if (i == j)
   {
      // get all entries of `mat_vec` except last one
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[k + 1] += RTmu * (vec[k + 1] + u[k]*vec[dim + 1]);
      }
      mat_vec[i + 1] +=  RTmu * (vec[i + 1] + u[i]*vec[dim+1])/3.0;
      // get last entry of `mat_vec`
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[dim + 1] += RTmu * u[k] * (vec[k + 1] + u[k] * vec[dim + 1]);
      }
     mat_vec[dim + 1] +=  RTmu * u[i] * (vec[i + 1] + 
                                               u[i] * vec[dim + 1])/3.0;
     mat_vec[dim + 1] += RT2k * vec[dim + 1];
   }
   else // apply off-diagonal block matrices (Cij; i!=j) on `vec`
   {
      // get all non-zero entries of `mat_vec` except last
      mat_vec[i + 1] -= 2 * RTmu * (vec[j + 1] + u[j] * vec[dim + 1])/3.0;
      mat_vec[j + 1] += RTmu * (vec[i + 1] + u[i] * vec[dim + 1]);
      // get last entry of `mat_vec`
      mat_vec[dim + 1] += RTmu * (u[j] * vec[i + 1] 
                                  - 2 * u[i] * vec[j + 1]/3.0
                                  +  u[i] * u[j] * vec[dim + 1]/3.0);
   }
}

/// Compute the product \f$ \sum_{d'=0}^{dim} C_{d,d'} D_{d'} w \f$
/// \param[in] d - desired space component of the flux
/// \param[in] mu - nondimensionalized dynamic viscosity 
/// \param[in] Pr - Prandtl number
/// \param[in] q - state used to evaluate the \f$ C_d,d'} \f$ matrices
/// \param[in] Dw - derivatives of entropy varaibles, stored column-major
/// \param[out] mat_vec - stores the resulting matrix vector product
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void applyViscousScaling(int d, xdouble mu, double Pr, const xdouble *q,
                         const xdouble *Dw, xdouble *mat_vec)
{
   for (int k = 0; k < dim+2; ++k)
   {
      mat_vec[k] = 0.0;
   }
   for (int d2 = 0; d2 < dim; ++d2) {
      applyCijMatrix<xdouble, dim>(d, d2, mu, Pr, q, Dw+(d2*(dim+2)), mat_vec);
   }
}

/// Computes an entropy conservative adiabatic-wall flux for the derivatives
/// \param[in] dir - desired (scaled) normal vector to the wall
/// \param[in] mu - nondimensionalized dynamic viscosity 
/// \param[in] Pr - Prandtl number
/// \param[in] q - state at the wall location
/// \param[in] Dw - space derivatives of the entropy variables (column major)
/// \param[out] flux - wall flux
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This **does not** account for the no-slip condition.
template <typename xdouble, int dim>
void calcAdiabaticWallFlux(const xdouble *dir, xdouble mu, double Pr,
                           const xdouble *q, const xdouble *Dw, 
                           xdouble *flux)
{
   for (int k = 0; k < dim+2; ++k)
   {
      flux[k] = 0.0;
   }
   xdouble Dw_bnd[dim+2];
   for (int d = 0; d < dim; ++d)
   {
      for (int d2 = 0; d2 < dim; ++d2)
      {
         for (int k = 0; k < dim+2; ++k)
         {
            Dw_bnd[k] = Dw[d2*(dim+2) + k];
         }
         if (d2 == d) {
            Dw_bnd[dim+1] = 0.0; // set flux to zero
         }
         // we "sneak" dir[d] into the computation via mu
         applyCijMatrix<xdouble, dim>(d, d2, mu*dir[d], Pr, q, Dw_bnd, flux);
      }
   }
}

/// Computes an entropy stable no-slip penalty
/// \param[in] dir - desired (scaled) normal vector to the wall
/// \param[in] Jac - mapping Jacobian at the wall
/// \param[in] mu - nondimensionalized dynamic viscosity 
/// \param[in] Pr - Prandtl number
/// \param[in] qfs - a fixed state (e.g. free-stream value)
/// \param[in] q - state at the wall location
/// \param[out] flux - wall flux
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcNoSlipPenaltyFlux(const xdouble *dir, const xdouble Jac, 
                           xdouble mu, double Pr, const xdouble *qfs,
                           const xdouble *q, xdouble *flux)
{
   // evaluate the difference w - w_bc, where w_bc = [w[0], 0, 0, ...,w[dim+1]]
   xdouble dw[dim+2];
   dw[0] = 0.0;
   dw[dim+1] = 0.0;
   xdouble p = pressure<xdouble,dim>(q);
   for (int d = 0; d < dim; ++d)
   {
      dw[d+1] = q[d+1]/p;
   }
   // initialize flux; recall that applyCijMatrix adds to its output
   for (int k = 0; k < dim+2; ++k)
   {
      flux[k] = 0.0;
   }
   for (int d = 0; d < dim; ++d)
   {
      applyCijMatrix<xdouble, dim>(d, d, mu*dir[d], Pr, qfs, dw, flux);
   }
   // scale the penalty
   xdouble fac = sqrt(dot<xdouble,dim>(dir, dir))/Jac;
   for (int k = 0; k < dim+2; ++k)
   {
      flux[k] *= fac;
   }
}

} // namespace mach

#endif
