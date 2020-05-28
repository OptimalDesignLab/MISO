#ifndef MACH_EULER_SENS_INTEG
#define MACH_EULER_SENS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "mesh_sens_integ.hpp"
#include "euler_fluxes.hpp"

using adept::adouble;

namespace mach
{

/// Integrator for the mesh sensitivity of the Ismail-Roe domain integrator
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

/// Integrator for inviscid slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class SlipWallBCMeshSens : public BoundaryMeshSensIntegrator<
                               SlipWallBCMeshSens<dim, entvar>>
{
public:
   /// Constructs an integrator for a slip-wall boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SlipWallBCMeshSens(adept::Stack &diff_stack,
                      const mfem::GridFunction &state_vec,
                      const mfem::GridFunction &adjoint_vec,
                      int num_state_vars = 1, double a = 1.0)
       : BoundaryMeshSensIntegrator<SlipWallBCMeshSens<dim, entvar>>(
             state_vec, adjoint_vec, dim + 2, a), stack(diff_stack) {}

   /// Compute ther derivative of flux_bar*flux w.r.t. `dir`
   /// \param[in] x - location at which the derivative is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[in] flux_bar - flux weighting (e.g. the adjoint)
   /// \param[out] dir_bar - derivative with respect to `dir`
   void calcFluxBar(const mfem::Vector &x, const mfem::Vector &dir,
                    const mfem::Vector &u, const mfem::Vector &flux_bar,
                    mfem::Vector &dir_bar);

protected:
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
};

/// Integrator for differentiating inviscid far-field boundary condition
/// Provides differentiation of adjoint weighted residual w.r.t. mach number
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class FarFieldBCDiff : public BoundaryMeshSensIntegrator<FarFieldBCDiff<dim, entvar>>
{
public:
    /// Constructs an integrator for far-field boundary flux differentiation
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] state - state vector
   /// \param[in] adj - adjoint vector
   /// \param[in] q_far - state at the far-field
   /// \param[in] mach - mach number, needed for differentiation
   /// \param[in] aoa - angle of attack, needed for differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   FarFieldBCDiff(adept::Stack &diff_stack,
              const mfem::GridFunction &state,
              const mfem::GridFunction &adj,
              const mfem::Vector q_far, 
              double mach, double aoa,
              double a = 1.0)
       : BoundaryMeshSensIntegrator<FarFieldBCDiff<dim, entvar>>(
             state, adj, dim+2, a), stack(diff_stack), qfs(q_far), 
             work_vec(dim+2), mach_fs(mach), aoa_fs(aoa) {}

   /// Not used (or, rather, *do not use*!)
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q) { return 0.0; }

   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec);

   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac){}

   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac){}

   /// Construct the contribution to the element local dF/dX
   /// \param[in] el_bnd - the finite element whose dF/dX we want to update
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[out] elvect - element local dF/dX
   virtual void AssembleRHSElementVect(const mfem::FiniteElement &el_bnd,
                                       mfem::FaceElementTransformations &trans,
                                       mfem::Vector &elvect) override;
private:
   /// Stores the far-field state
   mfem::Vector qfs;
   /// Work vector for boundary flux computation
   mfem::Vector work_vec;
   /// Far field flow parameters
   double mach_fs, aoa_fs;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   mfem::Vector u_face, flux_face;
};

/// Integrator for differentiating pressure force w.r.t. mach number
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class PressureForceDiff : public InviscidBoundaryIntegrator<PressureForceDiff<dim, entvar>>
{
public:
    /// Constructs an integrator for far-field boundary flux differentiation
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] q_far - state at the far-field
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   PressureForceDiff(adept::Stack &diff_stack,
              const mfem::GridFunction &state,
              const mfem::GridFunction &adj,
              const mfem::Vector force_dir, 
              double mach, double aoa,
              double a = 1.0)
       : InviscidBoundaryIntegrator<PressureForceDiff<dim, entvar>>(
             diff_stack, adj.FESpace()->FEColl(), dim+2, a), 
             stack(diff_stack), force_nrm(force_dir), 
             work_vec(dim+2), mach_fs(mach), aoa_fs(aoa) {}

   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec){}

   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac){}

   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac){}

private:
   /// `dim` entry unit normal vector specifying the direction of the force
   mfem::Vector force_nrm;
   /// Stores the far-field state
   mfem::Vector qfs;
   /// Work vector for boundary flux computation
   mfem::Vector work_vec;
   /// Far field flow parameters
   double mach_fs, aoa_fs;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   mfem::Vector u_face, flux_face;
};

#include "euler_sens_integ_def.hpp"

} // namespace mach

#endif