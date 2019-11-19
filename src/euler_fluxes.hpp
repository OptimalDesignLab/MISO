/// Functions related to Euler equations

#ifndef MACH_EULER_FLUXES
#define MACH_EULER_FLUXES

#include <algorithm> // std::max
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

/// Euler flux function in a given (scaled) direction
/// \param[in] dir - direction in which the flux is desired
/// \param[in] q - conservative variables
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcEulerFlux(const xdouble *dir, const xdouble *q, xdouble *flux)
{
   xdouble press = pressure<xdouble,dim>(q);
   xdouble U = dot<xdouble,dim>(q+1,dir);
   flux[0] = U;
   U /= q[0];
   for (int i = 0; i < dim; ++i)
   {
      flux[i+1] = q[i+1]*U + dir[i]*press;
   }
   flux[dim+1] = (q[dim+1] + press)*U;
}

/// Log-average function used in the Ismail-Roe flux function
/// \param[in] aL - nominal left state variable to average
/// \param[in] aR - nominal right state variable to average
/// \returns the logarithmic average of `aL` and `aR`.
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
xdouble logavg(const xdouble &aL, const xdouble &aR)
{
   xdouble xi = aL / aR;
   xdouble f = (xi - 1) / (xi + 1);
   xdouble u = f * f;
   double eps = 1.0e-3;
   xdouble F;
   if (u < eps)
   {
      F = 1.0 + u * (1. / 3. + u * (1. / 5. + u * (1. / 7. + u / 9.)));
   }
   else
   {
      F = (log(xi) / 2.0) / f;
   }
   return (aL + aR) / (2.0 * F);
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

/// Ismail-Roe entropy conservative flux function in direction `dir`
/// \param[in] dir - vector direction in which flux is wanted
/// \param[in] qL - conservative variables at "left" state
/// \param[in] qR - conservative variables at "right" state
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcIsmailRoeFaceFlux(const xdouble *dir, const xdouble *qL,
                           const xdouble *qR, xdouble *flux)
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

   xdouble U = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      U += (zL[0] * qL[i + 1] / qL[0] + zR[0] * qR[i + 1] / qR[0]) *
           dir[i] / (zL[0] + zR[0]);
   }
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
      flux[i + 1] += p1_hat * dir[i];
   }
   flux[dim + 1] = rho_hat * h_hat * U;
}

/// The spectral radius of the flux Jacobian in the direction `dir`
/// \param[in] dir - desired direction of flux Jacobian
/// \param[in] q - conservative variables used to evaluate Jacobian
/// \returns absolute value of the largest eigenvalue of the Jacobian
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
xdouble calcSpectralRadius(const xdouble *dir, const xdouble *q)
{
   xdouble press = pressure<xdouble,dim>(q);
   xdouble sndsp = sqrt(euler::gamma*press/q[0]);
   // U = u*dir[0] + v*dir[1] + ...
   xdouble U = dot<xdouble,dim>(q+1,dir)/q[0];
   xdouble dir_norm = sqrt(dot<xdouble,dim>(dir,dir));
   return fabs(U) + sndsp*dir_norm;
}

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

/// Applies the matrix `dQ/dW` to `vec`, and scales by the avg. spectral radius
/// \param[in] adjJ - the adjugate of the mapping Jacobian
/// \param[in] q - the state at which `dQ/dW` and radius are to be evaluated
/// \param[in] vec - the vector being multiplied
/// \param[out] mat_vec - the result of the operation
/// \warning adjJ must be supplied transposed from its `mfem` storage format,
/// so we can use pointer arithmetic to access its rows.
template <typename xdouble, int dim>
void applyLPSScaling(const xdouble *adjJ, const xdouble *q, const xdouble *vec,
                     xdouble *mat_vec)
{
   // first, get the average spectral radii
   xdouble spect = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      spect += calcSpectralRadius<xdouble,dim>(adjJ + i*dim, q);
   }
   spect /= static_cast<xdouble>(dim);
   calcdQdWProduct<xdouble,dim>(q, vec, mat_vec);
   for (int i = 0; i < dim+2; ++i)
   {
      mat_vec[i] *= spect;
   }
}

/// Boundary flux that uses characteristics to determine which state to use
/// \param[in] dir - direction in which the flux is desired
/// \param[in] qbnd - boundary values of the conservative variables
/// \param[in] q - interior domain values of the conservative variables
/// \param[in] work - a work vector of size `dim+2`
/// \param[out] flux - fluxes in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note the "flux Jacobian" is computed using `qbnd`, so the boundary values
/// define what is inflow and what is outflow.
template <typename xdouble, int dim>
void calcBoundaryFlux(const xdouble *dir, const xdouble *qbnd, const xdouble *q,
                      xdouble *work, xdouble *flux)
{
   // Define some constants
   const xdouble sat_Vn = 0.0; // 0.025
   const xdouble sat_Vl = 0.0; // 0.025

   // Define some constants used to construct the "Jacobian"
   const double dA = sqrt(dot<xdouble,dim>(dir,dir));
   const double fac = 1.0/qbnd[0];
   const double phi = 0.5*dot<xdouble,dim>(qbnd+1,qbnd+1)*fac*fac;
   const double H = euler::gamma*qbnd[dim+1]*fac - euler::gami*phi;
   const double a = sqrt(euler::gami*(H - phi));
   const double Un = dot<xdouble,dim>(qbnd+1,dir)*fac;
   double lambda1 = Un + dA*a;
   double lambda2 = Un - dA*a;
   double lambda3 = Un;
   const double rhoA = fabs(Un) + dA*a;
   lambda1 = 0.5*(std::max(fabs(lambda1), sat_Vn*rhoA) - lambda1);
   lambda2 = 0.5*(std::max(fabs(lambda2), sat_Vn*rhoA) - lambda2);
   lambda3 = 0.5*(std::max(fabs(lambda3), sat_Vl*rhoA) - lambda3);

   xdouble *dq = work;
   for (int i = 0; i < dim+2; ++i)
   {
      dq[i] = q[i] - qbnd[i];
   }
   calcEulerFlux<xdouble,dim>(dir, q, flux);

   // diagonal matrix multiply; note that flux was initialized by calcEulerFlux
   for (int i = 0; i < dim+2; ++i)
   {
      flux[i] += lambda3*dq[i];
   }

   // some scalars needed for E1*dq, E2*dq, E3*dq, and E4*dq
   xdouble tmp1 = 0.5*(lambda1 + lambda2) - lambda3;
   xdouble E1dq_fac = tmp1*euler::gami/(a*a);
   xdouble E2dq_fac = tmp1/(dA*dA);
   xdouble E34dq_fac = 0.5*(lambda1 - lambda2)/(dA*a);

   // get E1*dq + E4*dq and add to flux
   xdouble Edq = phi*dq[0] + dq[dim+1] - dot<xdouble,dim>(qbnd+1,dq+1)*fac;
   flux[0] += E1dq_fac*Edq;
   for (int i = 0; i < dim; ++i)
   {
      flux[i+1] += Edq*(E1dq_fac*qbnd[i+1]*fac + euler::gami*E34dq_fac*dir[i]);
   }
   flux[dim+1] += Edq*(E1dq_fac*H + euler::gami*E34dq_fac*Un);

   // get E2*dq + E3*dq and add to flux
   Edq = -Un*dq[0] + dot<xdouble,dim>(dir, dq+1);
   flux[0] += E34dq_fac*Edq;
   for (int i = 0; i < dim; ++i)
   {
      flux[i+1] += Edq*(E2dq_fac*dir[i] + E34dq_fac*qbnd[i+1]*fac);
   }
   flux[dim+1] += Edq*(E2dq_fac*Un + E34dq_fac*H);
}

/// Isentropic vortex exact state as a function of position
/// \param[in] x - location at which the exact state is desired
/// \param[out] qbnd - vortex conservative variable at `x`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \note  I reversed the flow direction to be clockwise, so the problem and
/// mesh are consistent with the LPS paper (that is, because the triangles are
/// subdivided from the quads using the opposite diagonal)
template <typename xdouble>
void calcIsentropicVortexState(const xdouble *x, xdouble *qbnd)
{
   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   xdouble rinv = ri/sqrt(x[0]*x[0] + x[1]*x[1]);
   xdouble rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                          1.0/euler::gami);
   xdouble Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                     (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   xdouble theta;
   if (x[0] > 1e-15)
   {
      theta = atan(x[1]/x[0]);
   }
   else
   {
      theta = M_PI/2.0;
   }
   xdouble press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   xdouble a = sqrt(euler::gamma*press/rho);

   qbnd[0] = rho;
   qbnd[1] = rho*a*Ma*sin(theta);
   qbnd[2] = -rho*a*Ma*cos(theta);
   qbnd[3] = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

/// A wrapper for `calcBoundaryFlux` in the case of the isentropic vortex
/// \param[in] x - location at which the boundary flux is desired
/// \param[in] dir - desired (scaled) direction of the flux
/// \param[in] q - conservative state variable on the interior of the boundary
/// \param[out] flux - the boundary flux in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
void calcIsentropicVortexFlux(const xdouble *x, const xdouble *dir,
                              const xdouble *q, xdouble *flux)
{
   xdouble qbnd[4];
   xdouble work[4];
   calcIsentropicVortexState<xdouble>(x, qbnd);
   calcBoundaryFlux<xdouble,2>(dir, qbnd, q, work, flux);
}

/// Wedge shock exact state as a function of position
/// \param[in] x - location at which the exact state is desired
/// \param[out] qbnd - vortex conservative variable at `x`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \note  Taken from Fundamentals of Aerodynamics (Anderson)
template <typename xdouble>
void calcWedgeShockState(const xdouble *x, xdouble *qbnd)
{
   double Mai = 2.4; //Ma1
   double rhoi = 1.0; //rho1
   double prsi = 1.0/euler::gamma;
   //assuming theta = 25 degrees, Ma1 = 2.4
   xdouble theta = (25/360)*2*M_PI;
   double beta = (52/360)*2*M_PI; //taken from Figure 9.9, Anderson for theta = 25 degrees, Ma1 = 2.4
   
   xdouble a = sqrt(euler::gamma*press/rho);
   
   xdouble Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                     (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   
   xdouble rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                          1.0/euler::gami);
   xdouble press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);

   xdouble thresh = .5/tan(beta); //assuming wedge tip is origin

   // if behind shock, set back to upstream state
   if(x(0) <= thresh)
   {
      theta = 0;
      Ma = Mai;
      rho = rhoi;
      press = prsi;
   }

   qbnd[0] = rho;
   qbnd[1] = rho*a*Ma*cos(theta);
   qbnd[2] = rho*a*Ma*sin(theta);
   qbnd[3] = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

/// A wrapper for `calcBoundaryFlux` in the case of the wedge shock
/// \param[in] x - location at which the boundary flux is desired
/// \param[in] dir - desired (scaled) direction of the flux
/// \param[in] q - conservative state variable on the interior of the boundary
/// \param[out] flux - the boundary flux in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
void calcWedgeShockFlux(const xdouble *x, const xdouble *dir,
                              const xdouble *q, xdouble *flux)
{
   xdouble qbnd[4];
   xdouble work[4];
   calcWedgeShockState<xdouble>(x, qbnd);
   calcBoundaryFlux<xdouble,2>(dir, qbnd, q, work, flux);
}

/// removes the component of momentum normal to the wall from `q`
/// \param[in] dir - vector perpendicular to the wall (does not need to be unit)
/// \param[in] q - the state whose momentum is being projected
/// \param[in] qbnd - the state with the normal component removed
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void projectStateOntoWall(const xdouble *dir, const xdouble *q, xdouble *qbnd)
{
   xdouble nrm[dim];
   xdouble fac = 1.0/sqrt(dot<xdouble,dim>(dir,dir));
   xdouble Unrm = 0.0;
   for (int i = 0; i < dim; ++i)
   {
      nrm[i] = dir[i]*fac;
      Unrm += nrm[i]*q[i+1];
   }
   qbnd[0] = q[0];
   qbnd[dim+1] = q[dim+1];
   for (int i = 0; i < dim; ++i)
   {
      qbnd[i+1] = q[i+1] - nrm[i]*Unrm;
   }
}

/// computes an adjoint consistent slip wall boundary condition 
/// \param[in] x - not used
/// \param[in] dir - desired (scaled) normal vector to the wall
/// \param[in] q - conservative state variable on the boundary
/// \param[out] flux - the boundary flux in the direction `dir`
/// \tparam xdouble - typically `double` or `adept::adouble`
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <typename xdouble, int dim>
void calcSlipWallFlux(const xdouble *x, const xdouble *dir, const xdouble *q,
                      xdouble *flux)
{
   xdouble qbnd[dim+2];
   projectStateOntoWall<xdouble,dim>(dir, q, qbnd);
   calcEulerFlux<xdouble,dim>(dir, qbnd, flux);
   //calcIsentropicVortexFlux<xdouble>(x, dir, q, flux);
}

} // namespace mach

#endif