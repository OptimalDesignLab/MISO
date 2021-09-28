/// Functions related to Navier-Stokes equations
#ifndef MACH_NAVIER_STOKES_FLUXES
#define MACH_NAVIER_STOKES_FLUXES

#include <algorithm>  // std::max

#include "utils.hpp"
#include "euler_fluxes.hpp"

namespace mach
{
/// For constants related to the Navier-Stokes equations
namespace navierstokes
{
/// Ratio of Sutherland's constant and free-stream temperature
const double ST = 198.60 / 460.0;
}  // namespace navierstokes

/// Returns the dynamic viscosity based on Sutherland's law
/// \param[in] q - state used to define the viscosity
/// \returns mu - **nondimensionalized** viscosity
/// \note This assumes the free-stream temperature given by navierstokes::ST
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSutherlandViscosity(const xdouble *q)
{
   xdouble a = sqrt(euler::gamma * pressure<xdouble, dim>(q) / q[0]);
   xdouble a2 = a * a;
   xdouble a3 = a * a * a;
   return a3 * (1.0 + navierstokes::ST) / (a2 + navierstokes::ST);
}

template <typename xdouble, int dim>
void setZeroNormalDeriv(const xdouble *dir, const xdouble *Dw, xdouble *Dwt)
{
   xdouble nrm[dim];
   xdouble fac = 1.0 / sqrt(dot<xdouble, dim>(dir, dir));
   int num_state = dim + 2;
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir[i] * fac;
   }
   for (int s = 0; s < dim + 2; ++s)
   {
      xdouble Dw_nrm = 0.0;
      for (int i = 0; i < dim; ++i)
      {
         Dw_nrm += Dw[s + i * num_state] * nrm[i];
      }
      for (int i = 0; i < dim; ++i)
      {
         Dwt[s + i * num_state] = Dw[s + i * num_state] - nrm[i] * Dw_nrm;
      }
   }
}

/// Applies the matrix `Cij` to `dW/dX`
/// \param[in] i - index `i` in `Cij` matrix
/// \param[in] j - index `j` in `Cij` matrix
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
void applyCijMatrix(int i,
                    int j,
                    const xdouble mu,
                    const xdouble Pr,
                    const xdouble *q,
                    const xdouble *vec,
                    xdouble *mat_vec)
{
   using namespace std;
   // define the velocity components
   xdouble u[dim];
   for (int k = 0; k < dim; ++k)
   {
      u[k] = q[k + 1] / q[0];
   }
   xdouble RTmu = pressure<xdouble, dim>(q) / q[0];
   xdouble RT2k = RTmu * RTmu * euler::gamma * mu / (euler::gami * Pr);
   RTmu *= mu;
   // apply diagonal block matrices (Cij; i=j) on `vec`
   if (i == j)
   {
      // get all entries of `mat_vec` except last one
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[k + 1] += RTmu * (vec[k + 1] + u[k] * vec[dim + 1]);
      }
      mat_vec[i + 1] += RTmu * (vec[i + 1] + u[i] * vec[dim + 1]) / 3.0;
      // get last entry of `mat_vec`
      for (int k = 0; k < dim; ++k)
      {
         mat_vec[dim + 1] += RTmu * u[k] * (vec[k + 1] + u[k] * vec[dim + 1]);
      }
      mat_vec[dim + 1] +=
          RTmu * u[i] * (vec[i + 1] + u[i] * vec[dim + 1]) / 3.0;
      mat_vec[dim + 1] += RT2k * vec[dim + 1];
   }
   else  // apply off-diagonal block matrices (Cij; i!=j) on `vec`
   {
      // get all non-zero entries of `mat_vec` except last
      mat_vec[i + 1] -= 2 * RTmu * (vec[j + 1] + u[j] * vec[dim + 1]) / 3.0;
      mat_vec[j + 1] += RTmu * (vec[i + 1] + u[i] * vec[dim + 1]);
      // get last entry of `mat_vec`
      mat_vec[dim + 1] +=
          RTmu * (u[j] * vec[i + 1] - 2 * u[i] * vec[j + 1] / 3.0 +
                  u[i] * u[j] * vec[dim + 1] / 3.0);
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
void applyViscousScaling(int d,
                         xdouble mu,
                         double Pr,
                         const xdouble *q,
                         const xdouble *Dw,
                         xdouble *mat_vec)
{
   for (int k = 0; k < dim + 2; ++k)
   {
      mat_vec[k] = 0.0;
   }
   for (int d2 = 0; d2 < dim; ++d2)
   {
      applyCijMatrix<xdouble, dim>(
          d, d2, mu, Pr, q, Dw + (d2 * (dim + 2)), mat_vec);
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
void calcAdiabaticWallFlux(const xdouble *dir,
                           xdouble mu,
                           double Pr,
                           const xdouble *q,
                           const xdouble *Dw,
                           xdouble *flux)
{
   for (int k = 0; k < dim + 2; ++k)
   {
      flux[k] = 0.0;
   }
   // Determine the unit normal and find the component of the gradient of w[end]
   // in the direction nrm; this is used to project out the normal component
   xdouble nrm[dim];
   xdouble fac = 1.0 / sqrt(dot<xdouble, dim>(dir, dir));
   xdouble Dw_nrm = 0.0;
   int num_state = dim + 2;
   for (int d = 0; d < dim; ++d)
   {
      nrm[d] = dir[d] * fac;
      Dw_nrm += Dw[d * num_state + dim + 1] * nrm[d];
   }
   // Apply C_ij matrices to the boundary values of Dw
   xdouble Dw_bnd[dim + 2];
   for (int d = 0; d < dim; ++d)
   {
      for (int d2 = 0; d2 < dim; ++d2)
      {
         for (int k = 0; k < dim + 2; ++k)
         {
            Dw_bnd[k] = Dw[d2 * (dim + 2) + k];
         }
         // find the corrected component for Dw[end]
         Dw_bnd[dim + 1] -= Dw_nrm * nrm[d2];
         // we "sneak" dir[d] into the computation via mu
         applyCijMatrix<xdouble, dim>(d, d2, mu * dir[d], Pr, q, Dw_bnd, flux);
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
void calcNoSlipPenaltyFlux(const xdouble *dir,
                           const xdouble Jac,
                           xdouble mu,
                           double Pr,
                           const xdouble *qfs,
                           const xdouble *q,
                           xdouble *flux)
{
   // evaluate the difference w - w_bc, where w_bc = [w[0], 0, 0, ...,w[dim+1]]
   xdouble dw[dim + 2];
   dw[0] = 0.0;
   dw[dim + 1] = 0.0;
   auto p = pressure<xdouble, dim>(q);
   for (int d = 0; d < dim; ++d)
   {
      dw[d + 1] = q[d + 1] / p;
   }
   // initialize flux; recall that applyCijMatrix adds to its output
   for (int k = 0; k < dim + 2; ++k)
   {
      flux[k] = 0.0;
   }
   for (int d = 0; d < dim; ++d)
   {
      applyCijMatrix<xdouble, dim>(d, d, mu, Pr, qfs, dw, flux);
   }
   // scale the penalty
   xdouble fac = sqrt(dot<xdouble, dim>(dir, dir)) / Jac;
   for (int k = 1; k < dim + 1; ++k)
   {
      flux[k] *= fac;
   }
   // zero out the last entry; first entry should be zeroed already
   flux[dim + 1] = 0.0;
}

/// Computes the dual-consistent term for the no-slip wall penalty
/// \param[in] dir - desired (scaled) normal vector to the wall
/// \param[in] mu - nondimensionalized dynamic viscosity
/// \param[in] Pr - Prandtl number
/// \param[in] q - state at the wall location
/// \param[out] fluxes - fluxes to be scaled by Dw (column major)
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcNoSlipDualFlux(const xdouble *dir,
                        xdouble mu,
                        double Pr,
                        const xdouble *q,
                        xdouble *fluxes)
{
   int num_state = dim + 2;
   // zero out the fluxes, since applyCijMatrix accummulates
   for (int i = 0; i < num_state * dim; ++i)
   {
      fluxes[i] = 0.0;
   }
   // evaluate the difference w - w_bc, where w_bc = [w[0], 0, 0, ...,w[dim+1]]
   xdouble dw[num_state];
   dw[0] = 0.0;
   dw[dim + 1] = 0.0;
   auto p = pressure<xdouble, dim>(q);
   for (int d = 0; d < dim; ++d)
   {
      dw[d + 1] = q[d + 1] / p;
   }
   // loop over the normal components
   for (int d = 0; d < dim; ++d)
   {
      // loop over the derivative directions
      for (int d2 = 0; d2 < dim; ++d2)
      {
         // we "sneak" dir[d] into the computation via mu
         // !!!! we also sneak the sign into the computation here
         applyCijMatrix<xdouble, dim>(
             d2, d, -mu * dir[d], Pr, q, dw, fluxes + (d2 * num_state));
      }
   }
#if 1
   // The following appears to be redundant (as far as the error goes)
   // Project out the normal component from the last variable
   xdouble nrm[dim];
   xdouble fac = 1.0 / sqrt(dot<xdouble, dim>(dir, dir));
   xdouble flux_nrm = 0.0;
   for (int d = 0; d < dim; ++d)
   {
      nrm[d] = dir[d] * fac;
      flux_nrm += fluxes[d * num_state + dim + 1] * nrm[d];
   }
   for (int d = 0; d < dim; ++d)
   {
      fluxes[d * num_state + dim + 1] -= flux_nrm * nrm[d];
   }
#endif
}

/// MMS source term for a particular Navier-Stokes verification
/// \param[in] mu - nondimensionalized dynamic viscosity
/// \param[in] Pr - Prandtl number
/// \param[in] x - location at which to evaluate the source
/// \param[out] src - the source value
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
void calcViscousMMS(double mu, double Pr, const xdouble *x, xdouble *src)
{
   double gamma = euler::gamma;
   const double rho0 = 1.0;
   const double rhop = 0.05;
   const double U0 = 0.5;
   const double Up = 0.05;
   const double T0 = 1.0;
   const double Tp = 0.05;
   double kappa = mu * gamma / (Pr * euler::gami);

   src[0] =
       M_PI *
       (-Up * rhop * pow(sin(M_PI * x[0]), 2) * pow(sin(2 * M_PI * x[0]), 2) *
            sin(M_PI * x[1]) * cos(M_PI * x[1]) +
        2 * Up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(M_PI * x[0]) -
        Up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            pow(sin(2 * M_PI * x[0]), 2) * cos(M_PI * x[1]) -
        2 * rhop *
            (4 * U0 * x[1] * (x[1] - 1) -
             Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
            sin(M_PI * x[0]) * sin(M_PI * x[1]) * cos(M_PI * x[0]));

   src[1] =
       2 * Tp * x[0] *
           (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (2 * pow(x[0], 2) - 3 * x[0] + 1) +
       0.33333333333333337 * pow(M_PI, 2) * Up * mu *
           (sin(M_PI * (4 * x[0] - x[1])) + sin(M_PI * (4 * x[0] + x[1]))) +
       2.6666666666666665 * pow(M_PI, 2) * Up * mu * pow(sin(M_PI * x[0]), 2) *
           sin(2 * M_PI * x[1]) -
       2.6666666666666665 * pow(M_PI, 2) * Up * mu * sin(2 * M_PI * x[1]) *
           pow(cos(M_PI * x[0]), 2) +
       M_PI * Up * rhop *
           (4 * U0 * x[1] * (x[1] - 1) -
            Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
           pow(sin(M_PI * x[0]), 2) * pow(sin(2 * M_PI * x[0]), 2) *
           sin(M_PI * x[1]) * cos(M_PI * x[1]) -
       4 * M_PI * Up *
           (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (4 * U0 * x[1] * (x[1] - 1) -
            Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
           sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(M_PI * x[0]) +
       M_PI * Up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (4 * U0 * x[1] * (x[1] - 1) -
            Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
           pow(sin(2 * M_PI * x[0]), 2) * cos(M_PI * x[1]) +
       2 * Up * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
           (2 * U0 * x[1] + 2 * U0 * (x[1] - 1) -
            M_PI * Up * pow(sin(M_PI * x[0]), 2) * cos(2 * M_PI * x[1])) *
           pow(sin(2 * M_PI * x[0]), 2) * sin(M_PI * x[1]) +
       4 * mu *
           (2 * U0 + pow(M_PI, 2) * Up * pow(sin(M_PI * x[0]), 2) *
                         sin(2 * M_PI * x[1])) +
       (1.0 / 2.0) * M_PI * rhop *
           (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                       pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
           (cos(M_PI * (2 * x[0] - x[1])) - cos(M_PI * (2 * x[0] + x[1]))) +
       2 * M_PI * rhop *
           pow(4 * U0 * x[1] * (x[1] - 1) -
                   Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]),
               2) *
           sin(M_PI * x[0]) * sin(M_PI * x[1]) * cos(M_PI * x[0]);

   src[2] = 2 * Tp * x[1] *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                (2 * pow(x[1], 2) - 3 * x[1] + 1) +
            M_PI * pow(Up, 2) * rhop * pow(sin(M_PI * x[0]), 2) *
                pow(sin(2 * M_PI * x[0]), 4) * pow(sin(M_PI * x[1]), 2) *
                cos(M_PI * x[1]) -
            16 * M_PI * pow(Up, 2) *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                pow(sin(M_PI * x[0]), 3) * pow(sin(M_PI * x[1]), 2) *
                pow(cos(M_PI * x[0]), 3) * cos(M_PI * x[1]) +
            2 * M_PI * pow(Up, 2) *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                pow(sin(2 * M_PI * x[0]), 4) * sin(M_PI * x[1]) *
                cos(M_PI * x[1]) -
            0.33333333333333337 * pow(M_PI, 2) * Up * mu *
                (sin(M_PI * (2 * x[0] - 2 * x[1])) +
                 sin(M_PI * (2 * x[0] + 2 * x[1]))) -
            9.3333333333333339 * pow(M_PI, 2) * Up * mu *
                pow(sin(2 * M_PI * x[0]), 2) * sin(M_PI * x[1]) +
            8 * pow(M_PI, 2) * Up * mu * sin(M_PI * x[1]) *
                pow(cos(2 * M_PI * x[0]), 2) +
            2 * M_PI * Up * rhop *
                (4 * U0 * x[1] * (x[1] - 1) -
                 Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                sin(M_PI * x[0]) * pow(sin(2 * M_PI * x[0]), 2) *
                pow(sin(M_PI * x[1]), 2) * cos(M_PI * x[0]) +
            4 * M_PI * Up *
                (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                (4 * U0 * x[1] * (x[1] - 1) -
                 Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                sin(2 * M_PI * x[0]) * sin(M_PI * x[1]) * cos(2 * M_PI * x[0]) +
            M_PI * rhop *
                (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                            pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]);

   src[3] =
       (1.0 / 2.0) *
       (2 * M_PI * Up *
            (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            (2 * T0 +
             2 * Tp *
                 (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                  pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
             (gamma - 1) *
                 (2 * T0 +
                  2 * Tp *
                      (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                       pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
                  pow(Up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                      pow(sin(M_PI * x[1]), 2) +
                  pow(4 * U0 * x[1] * (x[1] - 1) -
                          Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]),
                      2))) *
            sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) * cos(M_PI * x[0]) -
        M_PI * Up *
            (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
            (2 * T0 +
             2 * Tp *
                 (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                  pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
             (gamma - 1) *
                 (2 * T0 +
                  2 * Tp *
                      (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                       pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2)) +
                  pow(Up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                      pow(sin(M_PI * x[1]), 2) +
                  pow(4 * U0 * x[1] * (x[1] - 1) -
                          Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]),
                      2))) *
            pow(sin(2 * M_PI * x[0]), 2) * cos(M_PI * x[1]) -
        Up *
            (4 * Tp * x[1] *
                 (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                 (2 * pow(x[1], 2) - 3 * x[1] + 1) +
             2 * M_PI * rhop *
                 (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                 pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]) +
             (gamma - 1) *
                 (4 * Tp * x[1] *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (2 * pow(x[1], 2) - 3 * x[1] + 1) +
                  2 * M_PI * rhop *
                      (T0 +
                       Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                      pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]) +
                  M_PI * rhop *
                      (pow(Up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                           pow(sin(M_PI * x[1]), 2) +
                       pow(4 * U0 * x[1] * (x[1] - 1) -
                               Up * pow(sin(M_PI * x[0]), 2) *
                                   sin(2 * M_PI * x[1]),
                           2)) *
                      pow(sin(M_PI * x[0]), 2) * cos(M_PI * x[1]) +
                  2 * (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (M_PI * pow(Up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                           sin(M_PI * x[1]) * cos(M_PI * x[1]) +
                       2 * (4 * U0 * x[1] * (x[1] - 1) - Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                           (2 * U0 * x[1] + 2 * U0 * (x[1] - 1) -
                            M_PI * Up * pow(sin(M_PI * x[0]), 2) *
                                cos(2 * M_PI * x[1]))))) *
            pow(sin(2 * M_PI * x[0]), 2) * sin(M_PI * x[1]) +
        2 * (gamma - 1) *
            (-2 * Tp * kappa * (6 * pow(x[0], 2) - 6 * x[0] + 1) -
             2 * Tp * kappa * (6 * pow(x[1], 2) - 6 * x[1] + 1) +
             5.3333333333333339 * pow(M_PI, 2) * pow(Up, 2) * mu *
                 (1 - 2 * pow(sin(M_PI * x[1]), 2)) * pow(sin(M_PI * x[0]), 3) *
                 sin(M_PI * x[1]) * pow(cos(M_PI * x[0]), 3) -
             21.333333333333332 * pow(M_PI, 2) * pow(Up, 2) * mu *
                 pow(sin(M_PI * x[0]), 3) * sin(M_PI * x[1]) *
                 pow(cos(M_PI * x[0]), 3) * pow(cos(M_PI * x[1]), 2) -
             5.333333333333333 * pow(M_PI, 2) * pow(Up, 2) * mu *
                 pow(sin(M_PI * x[0]), 2) * pow(sin(2 * M_PI * x[1]), 2) *
                 pow(cos(M_PI * x[0]), 2) +
             9.3333333333333339 * pow(M_PI, 2) * pow(Up, 2) * mu *
                 pow(sin(2 * M_PI * x[0]), 4) * pow(sin(M_PI * x[1]), 2) -
             1.3333333333333333 * pow(M_PI, 2) * pow(Up, 2) * mu *
                 pow(sin(2 * M_PI * x[0]), 4) * pow(cos(M_PI * x[1]), 2) -
             24 * pow(M_PI, 2) * pow(Up, 2) * mu *
                 pow(sin(2 * M_PI * x[0]), 2) * pow(sin(M_PI * x[1]), 2) *
                 pow(cos(2 * M_PI * x[0]), 2) -
             2.6666666666666665 * pow(M_PI, 2) * Up * mu *
                 (4 * U0 * x[1] * (x[1] - 1) -
                  Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                 pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1]) -
             1.3333333333333335 * pow(M_PI, 2) * Up * mu *
                 (4 * U0 * x[1] * (x[1] - 1) -
                  Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                 sin(2 * M_PI * x[0]) * cos(2 * M_PI * x[0]) *
                 cos(M_PI * x[1]) +
             2.6666666666666665 * pow(M_PI, 2) * Up * mu *
                 (4 * U0 * x[1] * (x[1] - 1) -
                  Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                 sin(2 * M_PI * x[1]) * pow(cos(M_PI * x[0]), 2) -
             16 * M_PI * Up * mu *
                 (2 * U0 * x[1] + 2 * U0 * (x[1] - 1) -
                  M_PI * Up * pow(sin(M_PI * x[0]), 2) * cos(2 * M_PI * x[1])) *
                 sin(2 * M_PI * x[0]) * sin(M_PI * x[1]) *
                 cos(2 * M_PI * x[0]) -
             4 * mu *
                 (2 * U0 + pow(M_PI, 2) * Up * pow(sin(M_PI * x[0]), 2) *
                               sin(2 * M_PI * x[1])) *
                 (4 * U0 * x[1] * (x[1] - 1) -
                  Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) -
             4 * mu *
                 pow(2 * U0 * x[1] + 2 * U0 * (x[1] - 1) -
                         M_PI * Up * pow(sin(M_PI * x[0]), 2) *
                             cos(2 * M_PI * x[1]),
                     2)) -
        (4 * U0 * x[1] * (x[1] - 1) -
         Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
            (4 * Tp * x[0] *
                 (rho0 + rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                 (2 * pow(x[0], 2) - 3 * x[0] + 1) +
             M_PI * rhop *
                 (T0 + Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                 (cos(M_PI * (2 * x[0] - x[1])) -
                  cos(M_PI * (2 * x[0] + x[1]))) +
             (gamma - 1) *
                 (4 * Tp * x[0] *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (2 * pow(x[0], 2) - 3 * x[0] + 1) +
                  4 * M_PI * Up *
                      (rho0 +
                       rhop * pow(sin(M_PI * x[0]), 2) * sin(M_PI * x[1])) *
                      (2 * Up * pow(sin(2 * M_PI * x[0]), 3) *
                           pow(sin(M_PI * x[1]), 2) * cos(2 * M_PI * x[0]) -
                       (4 * U0 * x[1] * (x[1] - 1) -
                        Up * pow(sin(M_PI * x[0]), 2) * sin(2 * M_PI * x[1])) *
                           sin(M_PI * x[0]) * sin(2 * M_PI * x[1]) *
                           cos(M_PI * x[0])) +
                  M_PI * rhop *
                      (T0 +
                       Tp * (pow(x[0], 4) - 2 * pow(x[0], 3) + pow(x[0], 2) +
                             pow(x[1], 4) - 2 * pow(x[1], 3) + pow(x[1], 2))) *
                      (cos(M_PI * (2 * x[0] - x[1])) -
                       cos(M_PI * (2 * x[0] + x[1]))) +
                  2 * M_PI * rhop *
                      (pow(Up, 2) * pow(sin(2 * M_PI * x[0]), 4) *
                           pow(sin(M_PI * x[1]), 2) +
                       pow(4 * U0 * x[1] * (x[1] - 1) -
                               Up * pow(sin(M_PI * x[0]), 2) *
                                   sin(2 * M_PI * x[1]),
                           2)) *
                      sin(M_PI * x[0]) * sin(M_PI * x[1]) *
                      cos(M_PI * x[0])))) /
       (gamma - 1);
}

}  // namespace mach

#endif
