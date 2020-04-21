#ifndef MACH_EULER_SENS_INTEG
#define MACH_EULER_SENS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "mesh_sens_integ.hpp"
#include "euler_fluxes.hpp"

using adept::adouble;

namespace mach
{

/// Integrator for the two-point entropy conservative Ismail-Roe flux
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class IsmailRoeMeshSensIntegrator : public DyadicMeshSensIntegrator<
                                IsmailRoeMeshSensIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for the Ismail-Roe flux over domains
   /// \param[in] a - factor, usually used to move terms to rhs
   IsmailRoeMeshSensIntegrator(const mfem::GridFunction &state_vec,
                               const mfem::GridFunction &adjoint_vec,
                               int num_state_vars = 1, double a = 1.0)
       : DyadicMeshSensIntegrator<IsmailRoeMeshSensIntegrator<dim, entvar>>(
             state_vec, adjoint_vec, num_state_vars, a) {}

   /// Ismail-Roe two-point (dyadic) entropy conservative flux function
   /// \param[in] di - physical coordinate direction in which flux is wanted
   /// \param[in] qL - state variables at "left" state
   /// \param[in] qR - state variables at "right" state
   /// \param[out] flux - fluxes in the direction `di`
   /// \note This is simply a wrapper for the function in `euler_fluxes.hpp`
   void calcFlux(int di, const mfem::Vector &qL,
                 const mfem::Vector &qR, mfem::Vector &flux);
};

#include "euler_sens_integ_def.hpp"

} // namespace mach

#endif