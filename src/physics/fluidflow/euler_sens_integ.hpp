#ifndef MACH_EULER_SENS_INTEG
#define MACH_EULER_SENS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "mesh_sens_integ.hpp"
#include "euler_fluxes.hpp"

namespace mach
{
/// Integrator for the mesh sensitivity of the Ismail-Roe domain integrator
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class IsmailRoeMeshSensIntegrator
 : public DyadicMeshSensIntegrator<IsmailRoeMeshSensIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for the Ismail-Roe flux over domains
   /// \param[in] a - factor, usually used to move terms to rhs
   IsmailRoeMeshSensIntegrator(const mfem::GridFunction &state_vec,
                               const mfem::GridFunction &adjoint_vec,
                               int num_state_vars = 1,
                               double a = 1.0)
    : DyadicMeshSensIntegrator<IsmailRoeMeshSensIntegrator<dim, entvar>>(
          state_vec,
          adjoint_vec,
          num_state_vars,
          a)
   { }

   /// Ismail-Roe two-point (dyadic) entropy conservative flux function
   /// \param[in] di - physical coordinate direction in which flux is wanted
   /// \param[in] qL - state variables at "left" state
   /// \param[in] qR - state variables at "right" state
   /// \param[out] flux - fluxes in the direction `di`
   /// \note This is simply a wrapper for the function in `euler_fluxes.hpp`
   void calcFlux(int di,
                 const mfem::Vector &qL,
                 const mfem::Vector &qR,
                 mfem::Vector &flux);
};

/// Integrator for inviscid slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class SlipWallBCMeshSens
 : public BoundaryMeshSensIntegrator<SlipWallBCMeshSens<dim, entvar>>
{
public:
   /// Constructs an integrator for a slip-wall boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SlipWallBCMeshSens(adept::Stack &diff_stack,
                      const mfem::GridFunction &state_vec,
                      const mfem::GridFunction &adjoint_vec,
                      int num_state_vars = 1,
                      double a = 1.0)
    : BoundaryMeshSensIntegrator<SlipWallBCMeshSens<dim, entvar>>(state_vec,
                                                                  adjoint_vec,
                                                                  dim + 2,
                                                                  a),
      stack(diff_stack)
   { }

   /// Compute ther derivative of flux_bar*flux w.r.t. `dir`
   /// \param[in] x - location at which the derivative is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[in] flux_bar - flux weighting (e.g. the adjoint)
   /// \param[out] dir_bar - derivative with respect to `dir`
   void calcFluxBar(const mfem::Vector &x,
                    const mfem::Vector &dir,
                    const mfem::Vector &u,
                    const mfem::Vector &flux_bar,
                    mfem::Vector &dir_bar);

protected:
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
};

}  // namespace mach

#include "euler_sens_integ_def.hpp"

#endif
