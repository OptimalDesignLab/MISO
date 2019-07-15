/// Functions related to Euler equations

#ifndef MACH_EULER_FLUXES
#define MACH_EULER_FLUXES

#include "utils.hpp"

namespace mach
{

/// For constants related to the Euler equations
namespace euler
{
   /// heat capcity ratio for air
   const double gamma = 1.4;
   /// ratio minus one
   const double gami = gamma - 1.0;
}

/// Pressure based on the ideal gas law equation of state
/// \param[in] q - the conservative variables
/// \tparam xdouble - either double or adouble
/// \tparam dim - number of physical dimensions
template <typename xdouble, int dim>
inline xdouble pressure(const xdouble *q)
{
   return euler::gami*(q[dim+1] - 0.5*dot<xdouble,dim>(q+1,q+1)/q[0]);
}

/// Log-average function used in the Ismail-Roe flux function
/// \param[in] aL - nominal left state variable to average
/// \param[in] aR - nominal right state variable to average
/// \returns the logarithmic average of `aL` and `aR`.
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
xdouble logavg(const xdouble &aL, const xdouble &aR)
{
  xdouble xi = aL/aR;
  xdouble f = (xi - 1)/(xi + 1);
  xdouble u = f*f;
  double eps = 1.0e-3;
  xdouble F;
  if (u < eps)
  {
     F = 1.0 + u*(1./3. + u*(1./5. + u*(1./7. + u/9.)));
  }
  else
  {
     F = (log(xi)/2.0)/f;
  }
  return (aL + aR)/(2.0*F);
}

/// Ismail-Roe two-point (dyadic) entropy conservative flux function
/// \param[in] di - physical coordinate direction in which flux is wanted
/// \param[in] qL - conservative variables at "left" state
/// \param[in] qR - conservative variables at "right" state
/// \param[out] flux - fluxes in the direction `di`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcIsmailRoeFlux(int di, const xdouble *qL, const xdouble *qR,
                       xdouble *flux)
{
   xdouble pL = pressure<xdouble,dim>(qL);
   xdouble pR = pressure<xdouble,dim>(qR);
   xdouble zL[dim + 2];
   xdouble zR[dim + 2];
   zL[0] = sqrt(qL[0] / pL);
   zR[0] = sqrt(qR[0] / pR);
   for (int i = 0; i < dim; ++i)
   {
      zL[i + 1] = zL[0] * qL[i + 1] / qL[0];
      zR[i + 1] = zR[0] * qR[i + 1] / qR[0];
   }
   zL[dim + 1] = sqrt(qL[0] * pL);
   zR[dim + 1] = sqrt(qR[0] * pR);

   xdouble rho_hat = 0.5 * (zL[0] + zR[0]) * logavg(zL[dim + 1], zR[dim + 1]);
   xdouble U = (zL[0] * qL[di + 1] / qL[0] + zR[0] * qR[di + 1] / qR[0]) / (zL[0] + zR[0]);
   xdouble p1_hat = (zL[dim + 1] + zR[dim + 1]) / (zL[0] + zR[0]);
   xdouble p2_hat = ((euler::gamma + 1.0) * logavg(zL[dim + 1], zR[dim + 1]) /
                         logavg(zL[0], zR[0]) +
                     (euler::gami) * (zL[dim + 1] + zR[dim + 1]) / (zL[0] + zR[0])) /
                    (2.0 * euler::gamma);
   xdouble h_hat = euler::gamma * p2_hat / (rho_hat * euler::gami);

   flux[0] = rho_hat * U;
   for (int i = 0; i < dim; ++i)
   {
      flux[i + 1] = (zL[i + 1] + zR[i + 1]) / (zL[0] + zR[0]); // u_hat
      h_hat += 0.5 * flux[i + 1] * flux[i + 1];
      flux[i + 1] *= rho_hat * U;
   }
   flux[di + 1] += p1_hat;
   flux[dim + 1] = rho_hat * h_hat * U;
}

/// The spectral radius of the flux Jacobian in the direction `dir`
/// \param[in] q - conservative variables used to evaluate Jacobian
/// \param[in] dir - desired direction of flux Jacobian
/// \returns absolute value of the largest eigenvalue of the Jacobian
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSpectralRadius(const xdouble *q, const xdouble *dir)
{
   xdouble press = pressure<xdouble,dim>(q);
   xdouble sndsp = sqrt(euler::gamma*press/q[0]);
   // U = u*dir[0] + v*dir[1] + ...
   xdouble U = dot<xdouble,dim>(q+1,dir)/q[0];
   xdouble dir_norm = sqrt(dot<xdouble,dim>(dir,dir));
   return fabs(U) + sndsp*dir_norm;
}

// TODO: How should we return matrices, particularly when they will be differentiated?
template <typename xdouble, int dim>
void calcdQdWProduct(const xdouble *q, const xdouble *vec, xdouble *dqdw_vec)
{
   xdouble p = pressure<xdouble,dim>(q);
   xdouble rho_inv = 1.0/q[0];
   xdouble h = (q[dim+1] + p)*rho_inv; // scaled version of h
   xdouble a2 = euler::gamma*p*rho_inv; // square of speed of sound

   // first row of dq/dw times vec
   dqdw_vec[0] = 0.0;
   for (int i = 0; i < dim+2; ++i)
   {
      dqdw_vec[0] += q[i]*vec[i];
   }

   // second through dim-1 rows of dq/dw times vec
   for (int j = 0; j < dim; ++j)
   {
      dqdw_vec[j+1] = 0.0;
      xdouble u = q[j+1]*rho_inv;
      for (int i = 0; i < dim+2; ++i)
      {
         dqdw_vec[j+1] += u*q[i]*vec[i];
      }
      dqdw_vec[j+1] += p*vec[j+1];
      dqdw_vec[j+1] += p*u*vec[dim+1];
   }

   // dim-th row of dq/dw times vec
   dqdw_vec[dim+1] = q[dim+1]*vec[0];
   for (int i = 0; i < dim; ++i)
   {
      dqdw_vec[dim+1] += q[i+1]*h*vec[i+1];
   }
   dqdw_vec[dim+1] += (q[0]*h*h - a2*p/euler::gami)*vec[dim+1];

//   dqdw[1,1] = rho
//   dqdw[2,1] = rhou
//   dqdw[3,1] = rhov
//   dqdw[4,1] = rhoe

//   dqdw[1,2] = rhou
//   dqdw[2,2] = rhou*rhou*rhoinv + p
//   dqdw[3,2] = rhou*rhov*rhoinv
//   dqdw[4,2] = rhou*h

//   dqdw[1,3] = rhov
//   dqdw[2,3] = rhou*rhov/rho
//   dqdw[3,3] = rhov*rhov*rhoinv + p
//   dqdw[4,3] = rhov*h

//   dqdw[1,4] = rhoe
//   dqdw[2,4] = h*rhou
//   dqdw[3,4] = h*rhov
//   dqdw[4,4] = rho*h*h - a2*p/gami

}

} // namespace mach

#endif