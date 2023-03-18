/// Functions related to Euler equations

#ifndef MACH_POTENTIAL_FLUXES
#define MACH_POTENTIAL_FLUXES

#include <algorithm>  // std::max

#include "adept.h"

#include "utils.hpp"

using adept::adouble;

namespace mach
{
/// For constants related to the Euler equations
namespace euler
{
/// gas constant
const double R = 287;
/// heat capcity ratio for air
const double gamma = 1.4;
/// ratio minus one
const double gami = gamma - 1.0;
}  // namespace euler
/// use this for circle
#if 1
/// MMS source term for a `potential` solution verification
/// \param[in] x - location at which to evaluate the source
/// \param[out] src - the source value
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
void calcPotentialMMS(const xdouble *x, xdouble *src)
{
   using euler::gami;
   using euler::gamma;
   const double xc = 5.0;
   const double yc = 5.0;
   const double Ma = 0.2; //0.2
   const double circ = 0.0;
   const double rad = 0.5;
   src[0] = 0.0;
   src[1] = 0.0;
   src[2] = 0.0;
   src[3] =
      Ma*(x[0] - xc)*(pow(Ma, 2)*gamma*(2*pow(rad, 2)*pow(x[1] - yc, 2)*(4.0*pow(rad, 4)*pow(x[0] - xc, 2)*(pow(x[0] - xc, 2) 
      - 3*pow(x[1] - yc, 2)) - 2.0*(pow(x[0] - xc, 2)*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2)) 
      - pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)))*(pow(rad, 2)*pow(x[0] - xc, 2) 
      - pow(rad, 2)*pow(x[1] - yc, 2) + pow(x[0] - xc, 2)*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2)) 
      - pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)) + (pow(x[0] - xc, 2) + pow(x[1] - yc, 2))*(pow(rad, 2) 
      + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)))) + (pow(x[0] - xc, 2)*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) 
      - 1.0*pow(x[1] - yc, 2)) - pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) 
      + 1.0*pow(x[1] - yc, 2)))*(4.0*pow(rad, 4)*pow(x[1] - yc, 2)*(-3*pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) 
      + 2.0*(pow(x[0] - xc, 2)*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2)) 
      - pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)))*(-pow(rad, 2)*pow(x[0] - xc, 2) 
      + pow(rad, 2)*pow(x[1] - yc, 2) - pow(x[0] - xc, 2)*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2)) 
      + pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)) + (pow(x[0] - xc, 2) 
      + pow(x[1] - yc, 2))*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2))))) + 2*(0.5*pow(Ma, 2)*gami*gamma*pow(pow(x[0] - xc, 2)
       + pow(x[1] - yc, 2), 4) - pow(Ma, 2)*gamma*(2.0*pow(rad, 4)*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 0.5*pow(pow(x[0] - xc, 2)*(pow(rad, 2) 
       - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2)) - pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)), 2)) 
       + 0.5*pow(Ma, 2)*gamma*pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 4) + 1.0*gami*pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 4)
        + 1.0*pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 4))*(pow(rad, 2)*pow(x[0] - xc, 2) - pow(rad, 2)*pow(x[1] - yc, 2) 
        + pow(rad, 2)*(-pow(x[0] - xc, 2) + 3*pow(x[1] - yc, 2)) + pow(x[0] - xc, 2)*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) 
        - 1.0*pow(x[1] - yc, 2)) - pow(x[1] - yc, 2)*(pow(rad, 2) + 1.0*pow(x[0] - xc, 2) + 1.0*pow(x[1] - yc, 2)) 
        - (pow(x[0] - xc, 2) + pow(x[1] - yc, 2))*(pow(rad, 2) - 1.0*pow(x[0] - xc, 2) - 1.0*pow(x[1] - yc, 2))))/(gami*gamma*pow(pow(x[0] - xc, 2)
         + pow(x[1] - yc, 2), 7)) ;
}
#endif
#if 0
/// MMS source term for a `potential` solution verification
/// \param[in] x - location at which to evaluate the source
/// \param[out] src - the source value
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
void calcPotentialMMS(const xdouble *x, xdouble *src)
{
   using euler::gami;
   using euler::gamma;
   const double xc = 5.0;
   const double yc = 5.0;
   const double Ma = 0.2;
   const double circ = 0.0;
   const double rad = 0.5;
   src[0] = 0.0;

   src[1] =
       pow(Ma, 2) * (x[0] - xc) *
       (-4.0 * pow(rad, 2) * pow(x[1] - yc, 2) *
            (pow(rad, 2) * pow(x[0] - xc, 2) - pow(rad, 2) * pow(x[1] - yc, 2) +
             pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                  1.0 * pow(x[1] - yc, 2)) -
             pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                  1.0 * pow(x[1] - yc, 2)) +
             (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                 (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                  1.0 * pow(x[1] - yc, 2))) -
        2.0 * pow(rad, 2) *
            (pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                  1.0 * pow(x[1] - yc, 2)) -
             pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                  1.0 * pow(x[1] - yc, 2))) *
            (-pow(x[0] - xc, 2) + 3 * pow(x[1] - yc, 2)) +
        4.0 *
            (pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                  1.0 * pow(x[1] - yc, 2)) -
             pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                  1.0 * pow(x[1] - yc, 2))) *
            (-pow(rad, 2) * pow(x[0] - xc, 2) +
             pow(rad, 2) * pow(x[1] - yc, 2) -
             pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                  1.0 * pow(x[1] - yc, 2)) +
             pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                  1.0 * pow(x[1] - yc, 2)) +
             (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                 (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                  1.0 * pow(x[1] - yc, 2)))) /
       pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 5);

   src[2] = pow(Ma, 2) * pow(rad, 2) * (x[1] - yc) *
            (pow(rad, 2) * pow(x[0] - xc, 2) *
                 (8.0 * pow(x[0] - xc, 2) - 24.0 * pow(x[1] - yc, 2)) +
             4.0 * pow(x[0] - xc, 2) *
                 (-pow(rad, 2) * pow(x[0] - xc, 2) +
                  pow(rad, 2) * pow(x[1] - yc, 2) +
                  pow(x[0] - xc, 2) *
                      (-pow(rad, 2) + pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) +
                  pow(x[1] - yc, 2) *
                      (pow(rad, 2) + pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) +
                  (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                      (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                       1.0 * pow(x[1] - yc, 2))) -
             2.0 *
                 (pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                       1.0 * pow(x[1] - yc, 2)) -
                  pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                       1.0 * pow(x[1] - yc, 2))) *
                 (3 * pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) /
            pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 5);

   src[3] =
       (x[0] - xc) *
       (pow(Ma, 4) * gamma *
            (2 * pow(rad, 2) * pow(x[1] - yc, 2) *
                 (gami * (4.0 * pow(rad, 4) * pow(x[0] - xc, 2) *
                              (-pow(x[0] - xc, 2) + 3 * pow(x[1] - yc, 2)) +
                          2.0 *
                              (pow(x[0] - xc, 2) *
                                   (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                    1.0 * pow(x[1] - yc, 2)) -
                               pow(x[1] - yc, 2) *
                                   (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                    1.0 * pow(x[1] - yc, 2))) *
                              (pow(rad, 2) * pow(x[0] - xc, 2) -
                               pow(rad, 2) * pow(x[1] - yc, 2) +
                               pow(x[0] - xc, 2) *
                                   (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                    1.0 * pow(x[1] - yc, 2)) -
                               pow(x[1] - yc, 2) *
                                   (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                    1.0 * pow(x[1] - yc, 2)) +
                               (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                                   (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                    1.0 * pow(x[1] - yc, 2)))) +
                  4.0 * pow(rad, 4) * pow(x[0] - xc, 2) *
                      (pow(x[0] - xc, 2) - 3 * pow(x[1] - yc, 2)) -
                  2.0 *
                      (pow(x[0] - xc, 2) *
                           (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                            1.0 * pow(x[1] - yc, 2)) -
                       pow(x[1] - yc, 2) *
                           (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                            1.0 * pow(x[1] - yc, 2))) *
                      (pow(rad, 2) * pow(x[0] - xc, 2) -
                       pow(rad, 2) * pow(x[1] - yc, 2) +
                       pow(x[0] - xc, 2) *
                           (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                            1.0 * pow(x[1] - yc, 2)) -
                       pow(x[1] - yc, 2) *
                           (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                            1.0 * pow(x[1] - yc, 2)) +
                       (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                           (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                            1.0 * pow(x[1] - yc, 2)))) -
             (pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                   1.0 * pow(x[1] - yc, 2)) -
              pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                   1.0 * pow(x[1] - yc, 2))) *
                 (gami * (4.0 * pow(rad, 4) * pow(x[1] - yc, 2) *
                              (-3 * pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) +
                          2.0 *
                              (pow(x[0] - xc, 2) *
                                   (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                    1.0 * pow(x[1] - yc, 2)) -
                               pow(x[1] - yc, 2) *
                                   (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                    1.0 * pow(x[1] - yc, 2))) *
                              (-pow(rad, 2) * pow(x[0] - xc, 2) +
                               pow(rad, 2) * pow(x[1] - yc, 2) -
                               pow(x[0] - xc, 2) *
                                   (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                    1.0 * pow(x[1] - yc, 2)) +
                               pow(x[1] - yc, 2) *
                                   (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                    1.0 * pow(x[1] - yc, 2)) +
                               (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                                   (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                    1.0 * pow(x[1] - yc, 2)))) -
                  4.0 * pow(rad, 4) * pow(x[1] - yc, 2) *
                      (-4 * pow(-x[0] + xc, 2) + pow(x[0] - xc, 2) +
                       pow(x[1] - yc, 2)) -
                  2.0 *
                      (-pow(x[0] - xc, 2) *
                           (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                            1.0 * pow(x[1] - yc, 2)) +
                       pow(x[1] - yc, 2) *
                           (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                            1.0 * pow(x[1] - yc, 2))) *
                      (pow(rad, 2) * pow(x[0] - xc, 2) -
                       pow(rad, 2) * pow(x[1] - yc, 2) +
                       pow(x[0] - xc, 2) *
                           (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                            1.0 * pow(x[1] - yc, 2)) -
                       pow(x[1] - yc, 2) *
                           (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                            1.0 * pow(x[1] - yc, 2)) -
                       (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                           (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                            1.0 * pow(x[1] - yc, 2))))) +
        2 * (pow(Ma, 4) * gami * gamma * (2.0 * pow(rad, 4) * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) + 0.5 * pow(pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) - 1.0 * pow(x[1] - yc, 2)) - pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) + 1.0 * pow(x[1] - yc, 2)), 2)) + pow(Ma, 2) * (-pow(Ma, 2) * gamma * (2.0 * pow(rad, 4) * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) + 0.5 * pow(pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) - 1.0 * pow(x[1] - yc, 2)) - pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) + 1.0 * pow(x[1] - yc, 2)), 2)) + 0.5 * pow(Ma, 2) * gamma * pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 4) + 1.0 * pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 4)) + 1.0 * gami * pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 4)) *
            (pow(rad, 2) * pow(x[0] - xc, 2) - pow(rad, 2) * pow(x[1] - yc, 2) +
             pow(rad, 2) * (-pow(x[0] - xc, 2) + 3 * pow(x[1] - yc, 2)) +
             pow(x[0] - xc, 2) * (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                                  1.0 * pow(x[1] - yc, 2)) -
             pow(x[1] - yc, 2) * (pow(rad, 2) + 1.0 * pow(x[0] - xc, 2) +
                                  1.0 * pow(x[1] - yc, 2)) -
             (pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) *
                 (pow(rad, 2) - 1.0 * pow(x[0] - xc, 2) -
                  1.0 * pow(x[1] - yc, 2)))) /
       (Ma * gami * gamma * pow(pow(x[0] - xc, 2) + pow(x[1] - yc, 2), 7));
}
#endif
#if 0
/// use this for ellipse
/// MMS source term for a `potential` solution verification
/// \param[in] x - location at which to evaluate the source
/// \param[out] src - the source value
/// \tparam xdouble - typically `double` or `adept::adouble`
template <typename xdouble>
void calcPotentialMMS(const xdouble *x, xdouble *src)
{
   using euler::gami;
   using euler::gamma;
   const double xc = 10.0;
   const double yc = 10.0;
   const double Ma = 0.2;
   const double a = 2.5;
   const double b = sqrt(a * (a - 1.0));
   xdouble signx = 1.0;
   if (x[0] - xc < 0)
   {
      signx = -1.0;
   }
   src[0] =
           -2.0 * Ma * (pow(a, 2) - pow(b, 2)) *
               (-1.0 * signx *
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        1.0 / 4.0) *
                    (-0.5 * (2.0 * x[0] - 2.0 * xc) * (-2 * x[1] + 2 * yc) *
                         (x[1] - yc) /
                         (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                          16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2)) +
                     0.5 * (2.0 * x[0] - 2.0 * xc) *
                         (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                          pow(x[1] - yc, 2)) /
                         (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                          16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2))) *
                    sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                    -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                        pow(x[1] - yc, 2))) +
                1.0 * signx *
                    (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                     (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                         (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                          0.25 * pow(x[1] - yc, 2))) *
                    cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                    -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                        pow(x[1] - yc, 2))) /
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        3.0 / 4.0)) *
               (1.0 * signx *
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        1.0 / 4.0) *
                    sin(0.5 * atan2(
                                  (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                  -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                      pow(x[1] - yc, 2))) +
                0.5 * x[1] - 0.5 * yc) /
               (pow(-pow(b, 2) -
                        pow(1.0 * signx *
                                    pow(0.25 * pow(x[0] - xc, 2) *
                                                pow(x[1] - yc, 2) +
                                            pow(-pow(b, 2) +
                                                    0.25 * pow(x[0] - xc, 2) -
                                                    0.25 * pow(x[1] - yc, 2),
                                                2),
                                        1.0 / 4.0) *
                                    sin(0.5 * atan2(
                                                  (2.0 * x[0] - 2.0 * xc) *
                                                      (x[1] - yc),
                                                  -4.0 * pow(b, 2) +
                                                      pow(x[0] - xc, 2) -
                                                      pow(x[1] - yc, 2))) +
                                0.5 * x[1] - 0.5 * yc,
                            2) +
                        pow(1.0 * signx *
                                    pow(0.25 * pow(x[0] - xc, 2) *
                                                pow(x[1] - yc, 2) +
                                            pow(-pow(b, 2) +
                                                    0.25 * pow(x[0] - xc, 2) -
                                                    0.25 * pow(x[1] - yc, 2),
                                                2),
                                        1.0 / 4.0) *
                                    cos(0.5 * atan2(
                                                  (2.0 * x[0] - 2.0 * xc) *
                                                      (x[1] - yc),
                                                  -4.0 * pow(b, 2) +
                                                      pow(x[0] - xc, 2) -
                                                      pow(x[1] - yc, 2))) +
                                0.5 * x[0] - 0.5 * xc,
                            2),
                    2) +
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 * atan2(
                                          (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2(
                                       (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2(
                                       (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc)) -
           2.0 * Ma * (pow(a, 2) - pow(b, 2)) *
               (1.0 * signx *
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        1.0 / 4.0) *
                    cos(0.5 * atan2(
                                  (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                  -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                      pow(x[1] - yc, 2))) +
                0.5 * x[0] - 0.5 * xc) *
               (1.0 * signx *
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        1.0 / 4.0) *
                    (-0.5 * (2.0 * x[0] - 2.0 * xc) * (-2 * x[1] + 2 * yc) *
                         (x[1] - yc) /
                         (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                          16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2)) +
                     0.5 * (2.0 * x[0] - 2.0 * xc) *
                         (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                          pow(x[1] - yc, 2)) /
                         (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                          16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2))) *
                    cos(0.5 * atan2(
                                  (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                  -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                      pow(x[1] - yc, 2))) +
                1.0 * signx *
                    (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                     (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                         (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                          0.25 * pow(x[1] - yc, 2))) *
                    sin(0.5 * atan2(
                                  (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                  -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                      pow(x[1] - yc, 2))) /
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        3.0 / 4.0) +
                0.5) /
               (pow(-pow(b, 2) -
                        pow(1.0 * signx *
                                    pow(0.25 * pow(x[0] - xc, 2) *
                                                pow(x[1] - yc, 2) +
                                            pow(-pow(b, 2) +
                                                    0.25 * pow(x[0] - xc, 2) -
                                                    0.25 * pow(x[1] - yc, 2),
                                                2),
                                        1.0 / 4.0) *
                                    sin(0.5 * atan2(
                                                  (2.0 * x[0] - 2.0 * xc) *
                                                      (x[1] - yc),
                                                  -4.0 * pow(b, 2) +
                                                      pow(x[0] - xc, 2) -
                                                      pow(x[1] - yc, 2))) +
                                0.5 * x[1] - 0.5 * yc,
                            2) +
                        pow(1.0 * signx *
                                    pow(0.25 * pow(x[0] - xc, 2) *
                                                pow(x[1] - yc, 2) +
                                            pow(-pow(b, 2) +
                                                    0.25 * pow(x[0] - xc, 2) -
                                                    0.25 * pow(x[1] - yc, 2),
                                                2),
                                        1.0 / 4.0) *
                                    cos(0.5 * atan2(
                                                  (2.0 * x[0] - 2.0 * xc) *
                                                      (x[1] - yc),
                                                  -4.0 * pow(b, 2) +
                                                      pow(x[0] - xc, 2) -
                                                      pow(x[1] - yc, 2))) +
                                0.5 * x[0] - 0.5 * xc,
                            2),
                    2) +
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 * atan2(
                                          (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2(
                                       (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2(
                                       (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc)) -
           2.0 * Ma * (pow(a, 2) - pow(b, 2)) *
               (1.0 * signx *
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        1.0 / 4.0) *
                    sin(0.5 * atan2(
                                  (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                  -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                      pow(x[1] - yc, 2))) +
                0.5 * x[1] - 0.5 * yc) *
               (1.0 * signx *
                    pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                            pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                    0.25 * pow(x[1] - yc, 2),
                                2),
                        1.0 / 4.0) *
                    cos(0.5 * atan2(
                                  (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                  -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                      pow(x[1] - yc, 2))) +
                0.5 * x[0] - 0.5 * xc) *
               (-(2 *
                      (-2.0 * signx *
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               1.0 / 4.0) *
                           (-0.5 * (2.0 * x[0] - 2.0 * xc) *
                                (-2 * x[1] + 2 * yc) * (x[1] - yc) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2)) +
                            0.5 * (2.0 * x[0] - 2.0 * xc) *
                                (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                 pow(x[1] - yc, 2)) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2))) *
                           sin(0.5 * atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                       2.0 * signx *
                           (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                            (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                                (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                 0.25 * pow(x[1] - yc, 2))) *
                           cos(0.5 * atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) /
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               3.0 / 4.0)) *
                      (1.0 * signx *
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               1.0 / 4.0) *
                           cos(0.5 * atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                       0.5 * x[0] - 0.5 * xc) -
                  2 * (1.0 * signx * pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) - 0.25 * pow(x[1] - yc, 2), 2), 1.0 / 4.0) * sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc), -4.0 * pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5 * x[1] - 0.5 * yc) *
                      (2.0 * signx *
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               1.0 / 4.0) *
                           (-0.5 * (2.0 * x[0] - 2.0 * xc) *
                                (-2 * x[1] + 2 * yc) * (x[1] - yc) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2)) +
                            0.5 * (2.0 * x[0] - 2.0 * xc) *
                                (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                 pow(x[1] - yc, 2)) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2))) *
                           cos(0.5 *
                               atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) +
                       2.0 * signx *
                           (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                            (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                                (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                 0.25 * pow(x[1] - yc, 2))) *
                           sin(0.5 *
                               atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) /
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               3.0 / 4.0) +
                       1.0)) *
                    (-pow(b, 2) -
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 sin(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[1] - 0.5 * yc,
                         2) +
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 cos(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[0] - 0.5 * xc,
                         2)) -
                (-4.0 * signx *
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         1.0 / 4.0) *
                     (-0.5 * (2.0 * x[0] - 2.0 * xc) * (-2 * x[1] + 2 * yc) *
                          (x[1] - yc) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2)) +
                      0.5 * (2.0 * x[0] - 2.0 * xc) *
                          (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                           pow(x[1] - yc, 2)) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2))) *
                     sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) +
                 4.0 * signx *
                     (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                      (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                          (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                           0.25 * pow(x[1] - yc, 2))) *
                     cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) /
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         3.0 / 4.0)) *
                    pow(1.0 * signx *
                                pow(0.25 * pow(x[0] - xc, 2) *
                                            pow(x[1] - yc, 2) +
                                        pow(-pow(b, 2) +
                                                0.25 * pow(x[0] - xc, 2) -
                                                0.25 * pow(x[1] - yc, 2),
                                            2),
                                    1.0 / 4.0) *
                                sin(0.5 *
                                    atan2(
                                        (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                        -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                            pow(x[1] - yc, 2))) +
                            0.5 * x[1] - 0.5 * yc,
                        2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) -
                (-1.0 * signx *
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         1.0 / 4.0) *
                     (-0.5 * (2.0 * x[0] - 2.0 * xc) * (-2 * x[1] + 2 * yc) *
                          (x[1] - yc) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2)) +
                      0.5 * (2.0 * x[0] - 2.0 * xc) *
                          (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                           pow(x[1] - yc, 2)) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2))) *
                     sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) +
                 1.0 * signx *
                     (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                      (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                          (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                           0.25 * pow(x[1] - yc, 2))) *
                     cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) /
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         3.0 / 4.0)) *
                    pow(1.0 * signx *
                                pow(0.25 * pow(x[0] - xc, 2) *
                                            pow(x[1] - yc, 2) +
                                        pow(-pow(b, 2) +
                                                0.25 * pow(x[0] - xc, 2) -
                                                0.25 * pow(x[1] - yc, 2),
                                            2),
                                    1.0 / 4.0) *
                                sin(0.5 *
                                    atan2(
                                        (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                        -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                            pow(x[1] - yc, 2))) +
                            0.5 * x[1] - 0.5 * yc,
                        2) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc) -
                (1.0 * signx *
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         1.0 / 4.0) *
                     sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) +
                 0.5 * x[1] - 0.5 * yc) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc) *
                    (2.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         (-0.5 * (2.0 * x[0] - 2.0 * xc) *
                              (-2 * x[1] + 2 * yc) * (x[1] - yc) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2)) +
                          0.5 * (2.0 * x[0] - 2.0 * xc) *
                              (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                               pow(x[1] - yc, 2)) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2))) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * signx *
                         (0.0625 * pow(x[0] - xc, 2) * (2 * x[1] - 2 * yc) +
                          (1.0 / 4.0) * (-1.0 * x[1] + yc) *
                              (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                               0.25 * pow(x[1] - yc, 2))) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) /
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             3.0 / 4.0) +
                     1.0)) /
               pow(pow(-pow(b, 2) -
                           pow(1.0 * signx *
                                       pow(0.25 * pow(x[0] - xc, 2) *
                                                   pow(x[1] - yc, 2) +
                                               pow(-pow(b, 2) +
                                                       0.25 *
                                                           pow(x[0] - xc, 2) -
                                                       0.25 * pow(x[1] - yc, 2),
                                                   2),
                                           1.0 / 4.0) *
                                       sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                           (x[1] - yc),
                                                       -4.0 * pow(b, 2) +
                                                           pow(x[0] - xc, 2) -
                                                           pow(x[1] - yc, 2))) +
                                   0.5 * x[1] - 0.5 * yc,
                               2) +
                           pow(1.0 * signx *
                                       pow(0.25 * pow(x[0] - xc, 2) *
                                                   pow(x[1] - yc, 2) +
                                               pow(-pow(b, 2) +
                                                       0.25 *
                                                           pow(x[0] - xc, 2) -
                                                       0.25 * pow(x[1] - yc, 2),
                                                   2),
                                           1.0 / 4.0) *
                                       cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                           (x[1] - yc),
                                                       -4.0 * pow(b, 2) +
                                                           pow(x[0] - xc, 2) -
                                                           pow(x[1] - yc, 2))) +
                                   0.5 * x[0] - 0.5 * xc,
                               2),
                       2) +
                       pow(1.0 * signx *
                                   pow(0.25 * pow(x[0] - xc, 2) *
                                               pow(x[1] - yc, 2) +
                                           pow(-pow(b, 2) +
                                                   0.25 * pow(x[0] - xc, 2) -
                                                   0.25 * pow(x[1] - yc, 2),
                                               2),
                                       1.0 / 4.0) *
                                   sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                       (x[1] - yc),
                                                   -4.0 * pow(b, 2) +
                                                       pow(x[0] - xc, 2) -
                                                       pow(x[1] - yc, 2))) +
                               0.5 * x[1] - 0.5 * yc,
                           2) *
                           (1.0 * signx *
                                pow(0.25 * pow(x[0] - xc, 2) *
                                            pow(x[1] - yc, 2) +
                                        pow(-pow(b, 2) +
                                                0.25 * pow(x[0] - xc, 2) -
                                                0.25 * pow(x[1] - yc, 2),
                                            2),
                                    1.0 / 4.0) *
                                cos(0.5 *
                                    atan2(
                                        (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                        -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                            pow(x[1] - yc, 2))) +
                            0.5 * x[0] - 0.5 * xc) *
                           (4.0 * signx *
                                pow(0.25 * pow(x[0] - xc, 2) *
                                            pow(x[1] - yc, 2) +
                                        pow(-pow(b, 2) +
                                                0.25 * pow(x[0] - xc, 2) -
                                                0.25 * pow(x[1] - yc, 2),
                                            2),
                                    1.0 / 4.0) *
                                cos(0.5 *
                                    atan2(
                                        (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                        -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                            pow(x[1] - yc, 2))) +
                            2.0 * x[0] - 2.0 * xc),
                   2) +
           1.0 * Ma *
               ((-pow(a, 2) -
                 pow(1.0 * signx *
                             pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2),
                                 1.0 / 4.0) *
                             sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                 (x[1] - yc),
                                             -4.0 *
                                                     pow(b, 2) +
                                                 pow(x[0] - xc, 2) -
                                                 pow(x[1] - yc, 2))) +
                         0.5 *
                             x[1] -
                         0.5 * yc,
                     2) +
                 pow(1.0 * signx *
                             pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2),
                                 1.0 / 4.0) *
                             cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                 (x[1] - yc),
                                             -4.0 *
                                                     pow(b, 2) +
                                                 pow(x[0] - xc, 2) -
                                                 pow(x[1] - yc, 2))) +
                         0.5 *
                             x[0] -
                         0.5 * xc,
                     2)) *
                    (-pow(b, 2) -
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 sin(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[1] - 0.5 * yc,
                         2) +
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 cos(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[0] - 0.5 * xc,
                         2)) +
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 * atan2(
                                          (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc)) *
               (-(-2 *
                      (2.0 * signx *
                           ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                                pow(x[1] - yc, 2) +
                            (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                                (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                 0.25 * pow(x[1] - yc, 2))) *
                           sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                               (x[1] - yc),
                                           -4.0 * pow(b, 2) +
                                               pow(x[0] - xc, 2) -
                                               pow(x[1] - yc, 2))) /
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               3.0 / 4.0) +
                       2.0 * signx *
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               1.0 / 4.0) *
                           (-0.5 * (2 * x[0] - 2 * xc) *
                                (2.0 * x[0] - 2.0 * xc) * (x[1] - yc) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2)) +
                            0.5 * (2.0 * x[1] - 2.0 * yc) *
                                (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                 pow(x[1] - yc, 2)) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2))) *
                           cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                               (x[1] - yc),
                                           -4.0 * pow(b, 2) +
                                               pow(x[0] - xc, 2) -
                                               pow(x[1] - yc, 2)))) *
                      (1.0 * signx *
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               1.0 / 4.0) *
                           sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                               (x[1] - yc),
                                           -4.0 * pow(b, 2) +
                                               pow(x[0] - xc, 2) -
                                               pow(x[1] - yc, 2))) +
                       0.5 * x[1] - 0.5 * yc) +
                  2 * (1.0 * signx * pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) - 0.25 * pow(x[1] - yc, 2), 2), 1.0 / 4.0) * cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc), -4.0 * pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5 * x[0] - 0.5 * xc) *
                      (2.0 * signx *
                           ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                                pow(x[1] - yc, 2) +
                            (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                                (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                 0.25 * pow(x[1] - yc, 2))) *
                           cos(0.5 *
                               atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) /
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               3.0 / 4.0) -
                       2.0 * signx *
                           pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2),
                               1.0 / 4.0) *
                           (-0.5 * (2 * x[0] - 2 * xc) *
                                (2.0 * x[0] - 2.0 * xc) * (x[1] - yc) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2)) +
                            0.5 * (2.0 * x[1] - 2.0 * yc) *
                                (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                 pow(x[1] - yc, 2)) /
                                (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 16.0 *
                                     pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                             0.25 * pow(x[1] - yc, 2),
                                         2))) *
                           sin(0.5 *
                               atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) +
                       1.0)) *
                    (-pow(b, 2) -
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 sin(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[1] - 0.5 * yc,
                         2) +
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 cos(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[0] - 0.5 * xc,
                         2)) -
                (2.0 * signx *
                     ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                          pow(x[1] - yc, 2) +
                      (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                          (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                           0.25 * pow(x[1] - yc, 2))) *
                     sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) /
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         3.0 / 4.0) +
                 2.0 * signx *
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         1.0 / 4.0) *
                     (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                          (x[1] - yc) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2)) +
                      0.5 * (2.0 * x[1] - 2.0 * yc) *
                          (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                           pow(x[1] - yc, 2)) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2))) *
                     cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2)))) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[1] - 0.5 * yc) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc) -
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 *
                                atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                              pow(x[1] - yc, 2) +
                          (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                              (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                               0.25 * pow(x[1] - yc, 2))) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) /
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             3.0 / 4.0) -
                     4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                              (x[1] - yc) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2)) +
                          0.5 * (2.0 * x[1] - 2.0 * yc) *
                              (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                               pow(x[1] - yc, 2)) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2))) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0) -
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 *
                                atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc) *
                    (1.0 * signx *
                         ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                              pow(x[1] - yc, 2) +
                          (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                              (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                               0.25 * pow(x[1] - yc, 2))) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) /
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             3.0 / 4.0) -
                     1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                              (x[1] - yc) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2)) +
                          0.5 * (2.0 * x[1] - 2.0 * yc) *
                              (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                               pow(x[1] - yc, 2)) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2))) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5)) /
               pow(pow(-pow(b, 2) -
                           pow(1.0 * signx *
                                       pow(0.25 * pow(x[0] - xc, 2) *
                                                   pow(x[1] - yc, 2) +
                                               pow(-pow(b, 2) +
                                                       0.25 *
                                                           pow(x[0] - xc, 2) -
                                                       0.25 * pow(x[1] - yc, 2),
                                                   2),
                                           1.0 / 4.0) *
                                       sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                           (x[1] - yc),
                                                       -4.0 * pow(b, 2) +
                                                           pow(x[0] - xc, 2) -
                                                           pow(x[1] - yc, 2))) +
                                   0.5 * x[1] - 0.5 * yc,
                               2) +
                           pow(1.0 * signx *
                                       pow(0.25 * pow(x[0] - xc, 2) *
                                                   pow(x[1] - yc, 2) +
                                               pow(-pow(b, 2) +
                                                       0.25 *
                                                           pow(x[0] - xc, 2) -
                                                       0.25 * pow(x[1] - yc, 2),
                                                   2),
                                           1.0 / 4.0) *
                                       cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                           (x[1] - yc),
                                                       -4.0 * pow(b, 2) +
                                                           pow(x[0] - xc, 2) -
                                                           pow(x[1] - yc, 2))) +
                                   0.5 * x[0] - 0.5 * xc,
                               2),
                       2) +
                       pow(1.0 * signx *
                                   pow(0.25 * pow(x[0] - xc, 2) *
                                               pow(x[1] - yc, 2) +
                                           pow(-pow(b, 2) +
                                                   0.25 * pow(x[0] - xc, 2) -
                                                   0.25 * pow(x[1] - yc, 2),
                                               2),
                                       1.0 / 4.0) *
                                   sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                       (x[1] - yc),
                                                   -4.0 * pow(b, 2) +
                                                       pow(x[0] - xc, 2) -
                                                       pow(x[1] - yc, 2))) +
                               0.5 * x[1] - 0.5 * yc,
                           2) *
                           (1.0 * signx *
                                pow(0.25 * pow(x[0] - xc, 2) *
                                            pow(x[1] - yc, 2) +
                                        pow(-pow(b, 2) +
                                                0.25 * pow(x[0] - xc, 2) -
                                                0.25 * pow(x[1] - yc, 2),
                                            2),
                                    1.0 / 4.0) *
                                cos(0.5 *
                                    atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                            0.5 * x[0] - 0.5 * xc) *
                           (4.0 * signx *
                                pow(0.25 * pow(x[0] - xc, 2) *
                                            pow(x[1] - yc, 2) +
                                        pow(-pow(b, 2) +
                                                0.25 * pow(x[0] - xc, 2) -
                                                0.25 * pow(x[1] - yc, 2),
                                            2),
                                    1.0 / 4.0) *
                                cos(0.5 *
                                    atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                            2.0 * x[0] - 2.0 * xc),
                   2) +
           1.0 * Ma *
               ((-(2.0 * signx *
                       ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                            pow(x[1] - yc, 2) +
                        (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                            (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                             0.25 * pow(x[1] - yc, 2))) *
                       sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2))) /
                       pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                       0.25 * pow(x[1] - yc, 2),
                                   2),
                           3.0 / 4.0) +
                   2.0 * signx *
                       pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                       0.25 * pow(x[1] - yc, 2),
                                   2),
                           1.0 / 4.0) *
                       (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                            (x[1] - yc) /
                            (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2)) +
                        0.5 * (2.0 * x[1] - 2.0 * yc) *
                            (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                             pow(x[1] - yc, 2)) /
                            (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2))) *
                       cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2)))) *
                     (1.0 * signx *
                          pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                  pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2),
                              1.0 / 4.0) *
                          sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                      0.5 * x[1] - 0.5 * yc) +
                 (1.0 * signx *
                      pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                              pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                      0.25 * pow(x[1] - yc, 2),
                                  2),
                          1.0 / 4.0) *
                      cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                  0.5 * x[0] - 0.5 * xc) *
                     (2.0 * signx *
                          ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                               pow(x[1] - yc, 2) +
                           (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                               (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                0.25 * pow(x[1] - yc, 2))) *
                          cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) /
                          pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                  pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2),
                              3.0 / 4.0) -
                      2.0 * signx *
                          pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                  pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2),
                              1.0 / 4.0) *
                          (-0.5 * (2 * x[0] - 2 * xc) *
                               (2.0 * x[0] - 2.0 * xc) * (x[1] - yc) /
                               (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                16.0 *
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2)) +
                           0.5 * (2.0 * x[1] - 2.0 * yc) *
                               (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                pow(x[1] - yc, 2)) /
                               (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                16.0 *
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2))) *
                          sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                      1.0)) *
                    (-pow(a, 2) -
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 sin(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[1] - 0.5 * yc,
                         2) +
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 cos(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[0] - 0.5 * xc,
                         2)) +
                (-(2.0 * signx *
                       ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                            pow(x[1] - yc, 2) +
                        (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                            (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                             0.25 * pow(x[1] - yc, 2))) *
                       sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2))) /
                       pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                       0.25 * pow(x[1] - yc, 2),
                                   2),
                           3.0 / 4.0) +
                   2.0 * signx *
                       pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                       0.25 * pow(x[1] - yc, 2),
                                   2),
                           1.0 / 4.0) *
                       (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                            (x[1] - yc) /
                            (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2)) +
                        0.5 * (2.0 * x[1] - 2.0 * yc) *
                            (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                             pow(x[1] - yc, 2)) /
                            (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2))) *
                       cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                       -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                           pow(x[1] - yc, 2)))) *
                     (1.0 * signx *
                          pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                  pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2),
                              1.0 / 4.0) *
                          sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                      0.5 * x[1] - 0.5 * yc) +
                 (1.0 * signx *
                      pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                              pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                      0.25 * pow(x[1] - yc, 2),
                                  2),
                          1.0 / 4.0) *
                      cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                  0.5 * x[0] - 0.5 * xc) *
                     (2.0 * signx *
                          ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                               pow(x[1] - yc, 2) +
                           (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                               (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                0.25 * pow(x[1] - yc, 2))) *
                          cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) /
                          pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                  pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2),
                              3.0 / 4.0) -
                      2.0 * signx *
                          pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                  pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2),
                              1.0 / 4.0) *
                          (-0.5 * (2 * x[0] - 2 * xc) *
                               (2.0 * x[0] - 2.0 * xc) * (x[1] - yc) /
                               (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                16.0 *
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2)) +
                           0.5 * (2.0 * x[1] - 2.0 * yc) *
                               (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                pow(x[1] - yc, 2)) /
                               (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                16.0 *
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2))) *
                          sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                          -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                              pow(x[1] - yc, 2))) +
                      1.0)) *
                    (-pow(b, 2) -
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 sin(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[1] - 0.5 * yc,
                         2) +
                     pow(1.0 * signx *
                                 pow(0.25 * pow(x[0] - xc, 2) *
                                             pow(x[1] - yc, 2) +
                                         pow(-pow(b, 2) +
                                                 0.25 * pow(x[0] - xc, 2) -
                                                 0.25 * pow(x[1] - yc, 2),
                                             2),
                                     1.0 / 4.0) *
                                 cos(0.5 *
                                     atan2(
                                         (2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                             0.5 * x[0] - 0.5 * xc,
                         2)) +
                (2.0 * signx *
                     ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                          pow(x[1] - yc, 2) +
                      (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                          (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                           0.25 * pow(x[1] - yc, 2))) *
                     sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2))) /
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         3.0 / 4.0) +
                 2.0 * signx *
                     pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                             pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                     0.25 * pow(x[1] - yc, 2),
                                 2),
                         1.0 / 4.0) *
                     (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                          (x[1] - yc) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2)) +
                      0.5 * (2.0 * x[1] - 2.0 * yc) *
                          (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                           pow(x[1] - yc, 2)) /
                          (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                           16.0 * pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                          0.25 * pow(x[1] - yc, 2),
                                      2))) *
                     cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                     -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                         pow(x[1] - yc, 2)))) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[1] - 0.5 * yc) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc) +
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 *
                                atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                              pow(x[1] - yc, 2) +
                          (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                              (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                               0.25 * pow(x[1] - yc, 2))) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) /
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             3.0 / 4.0) -
                     4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                              (x[1] - yc) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2)) +
                          0.5 * (2.0 * x[1] - 2.0 * yc) *
                              (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                               pow(x[1] - yc, 2)) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2))) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0) +
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 *
                                atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc) *
                    (1.0 * signx *
                         ((1.0 / 4.0) * (0.5 * x[0] - 0.5 * xc) *
                              pow(x[1] - yc, 2) +
                          (1.0 / 4.0) * (x[0] - 1.0 * xc) *
                              (-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                               0.25 * pow(x[1] - yc, 2))) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) /
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             3.0 / 4.0) -
                     1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         (-0.5 * (2 * x[0] - 2 * xc) * (2.0 * x[0] - 2.0 * xc) *
                              (x[1] - yc) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2)) +
                          0.5 * (2.0 * x[1] - 2.0 * yc) *
                              (-4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                               pow(x[1] - yc, 2)) /
                              (4.0 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                               16.0 *
                                   pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                           0.25 * pow(x[1] - yc, 2),
                                       2))) *
                         sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5)) /
               (pow(-pow(b, 2) -
                        pow(1.0 * signx *
                                    pow(0.25 * pow(x[0] - xc, 2) *
                                                pow(x[1] - yc, 2) +
                                            pow(-pow(b, 2) +
                                                    0.25 * pow(x[0] - xc, 2) -
                                                    0.25 * pow(x[1] - yc, 2),
                                                2),
                                        1.0 / 4.0) *
                                    sin(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                        (x[1] - yc),
                                                    -4.0 * pow(b, 2) +
                                                        pow(x[0] - xc, 2) -
                                                        pow(x[1] - yc, 2))) +
                                0.5 * x[1] - 0.5 * yc,
                            2) +
                        pow(1.0 * signx *
                                    pow(0.25 * pow(x[0] - xc, 2) *
                                                pow(x[1] - yc, 2) +
                                            pow(-pow(b, 2) +
                                                    0.25 * pow(x[0] - xc, 2) -
                                                    0.25 * pow(x[1] - yc, 2),
                                                2),
                                        1.0 / 4.0) *
                                    cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) *
                                                        (x[1] - yc),
                                                    -4.0 * pow(b, 2) +
                                                        pow(x[0] - xc, 2) -
                                                        pow(x[1] - yc, 2))) +
                                0.5 * x[0] - 0.5 * xc,
                            2),
                    2) +
                pow(1.0 * signx *
                            pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                    pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                            0.25 * pow(x[1] - yc, 2),
                                        2),
                                1.0 / 4.0) *
                            sin(0.5 *
                                atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                      -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                          pow(x[1] - yc, 2))) +
                        0.5 * x[1] - 0.5 * yc,
                    2) *
                    (1.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     0.5 * x[0] - 0.5 * xc) *
                    (4.0 * signx *
                         pow(0.25 * pow(x[0] - xc, 2) * pow(x[1] - yc, 2) +
                                 pow(-pow(b, 2) + 0.25 * pow(x[0] - xc, 2) -
                                         0.25 * pow(x[1] - yc, 2),
                                     2),
                             1.0 / 4.0) *
                         cos(0.5 * atan2((2.0 * x[0] - 2.0 * xc) * (x[1] - yc),
                                         -4.0 * pow(b, 2) + pow(x[0] - xc, 2) -
                                             pow(x[1] - yc, 2))) +
                     2.0 * x[0] - 2.0 * xc));
src[1] = -2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
+ pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
+ pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2)
 + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2)
  + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
  + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
  + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) 
  + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) 
  - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2)
   - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
   + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
   - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
   + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2)
    - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
    + 2.0*x[0] - 2.0*xc))*(-1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
    + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) 
    + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
    + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
    + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
    + 0.5*x[1] - 0.5*yc)/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
    + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
    - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
    + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2)
     - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
     + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
     - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))
      + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 
      0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
      + 2.0*x[0] - 2.0*xc), 2) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
      + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
      + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) 
      + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2)
       - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) 
       + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
       + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
       + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
       + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) 
       + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2)
        - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
        - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
        + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
        - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) 
        - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
        - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))
         + 0.5*x[0] - 0.5*xc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
         - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
         + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 0.5)/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
         + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) 
         - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2)
          + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) 
         - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
         + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(-2*(2*(-2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) 
         + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
         + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))
          + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) - 2*(-4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 4.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) 
          + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - 2*(-1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
          + 2.0*x[0] - 2.0*xc) - 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 3) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2)
           + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(((-2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))*(-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
         + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + ((-2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + (-4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 4.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc)
          + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) + (-1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) 
         + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) + 1.0*pow(Ma, 2)*pow((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2)*(-2*(-2*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) 
         + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) - 2*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) - 2*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0) - 2*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) 
         + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 3) + 1.0*pow(Ma, 2)*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) 
         - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(2*(-(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) 
         + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + 2*(-(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + 2*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) + 2*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0) + 2*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2)
          - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) 
         + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) ;
src[2] = 4.0*pow(Ma, 2)*pow(pow(a, 2) - pow(b, 2), 2)*(-2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) + 4.0*pow(Ma, 2)*pow(pow(a, 2) - pow(b, 2), 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0)/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) + 4.0*pow(Ma, 2)*pow(pow(a, 2) - pow(b, 2), 2)*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)*(-2*(2*(-2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) - 2*(-4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 4.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - 2*(-1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) - 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 3) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5)/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(-2*(-2*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) - 2*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) - 2*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0) - 2*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 3) - 2.0*pow(Ma, 2)*(pow(a, 2) - pow(b, 2))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*((-(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + (-(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + (2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) ;
src[3] = -2.0*Ma*(pow(a, 2) - pow(b, 2))*(-1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(0.5*pow(Ma, 2) + 1.0/gamma + 1.0/(gami*gamma))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)/(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)) - 2.0*Ma*(pow(a, 2) - pow(b, 2))*(0.5*pow(Ma, 2) + 1.0/gamma + 1.0/(gami*gamma))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 0.5)/(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)) - 2.0*Ma*(pow(a, 2) - pow(b, 2))*(0.5*pow(Ma, 2) + 1.0/gamma + 1.0/(gami*gamma))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(-(2*(-2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) - (-4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 4.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc) - (-1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0))*pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) - (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2.0*x[0] - 2.0*xc)*(-2*x[1] + 2*yc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[0] - 2.0*xc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*signx*(0.0625*pow(x[0] - xc, 2)*(2*x[1] - 2*yc) + (1.0/4.0)*(-1.0*x[1] + yc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 1.0))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) + Ma*((-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc))*(0.5*pow(Ma, 2) + 1.0/gamma + 1.0/(gami*gamma))*(-(-2*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + 2*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) - (2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5))/pow(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc), 2) + Ma*(0.5*pow(Ma, 2) + 1.0/gamma + 1.0/(gami*gamma))*((-(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(a, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + (-(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc) + (1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 1.0))*(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2)) + (2.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) + 2.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))))*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)*(1.0*signx*((1.0/4.0)*(0.5*x[0] - 0.5*xc)*pow(x[1] - yc, 2) + (1.0/4.0)*(x[0] - 1.0*xc)*(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2)))*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2)))/pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 3.0/4.0) - 1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*(-0.5*(2*x[0] - 2*xc)*(2.0*x[0] - 2.0*xc)*(x[1] - yc)/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)) + 0.5*(2.0*x[1] - 2.0*yc)*(-4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))/(4.0*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + 16.0*pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2)))*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5))/(pow(-pow(b, 2) - pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc, 2), 2) + pow(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*sin(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[1] - 0.5*yc, 2)*(1.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 0.5*x[0] - 0.5*xc)*(4.0*signx*pow(0.25*pow(x[0] - xc, 2)*pow(x[1] - yc, 2) + pow(-pow(b, 2) + 0.25*pow(x[0] - xc, 2) - 0.25*pow(x[1] - yc, 2), 2), 1.0/4.0)*cos(0.5*atan2((2.0*x[0] - 2.0*xc)*(x[1] - yc), -4.0*pow(b, 2) + pow(x[0] - xc, 2) - pow(x[1] - yc, 2))) + 2.0*x[0] - 2.0*xc)) ;
}
#endif

}  // namespace mach
#endif