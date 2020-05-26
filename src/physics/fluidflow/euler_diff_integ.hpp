#ifndef MACH_EULER_DIFF_INTEG
#define MACH_EULER_DIFF_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "inviscid_integ.hpp"
#include "euler_fluxes.hpp"

using adept::adouble;

namespace mach
{

/// Integrator for differentiating inviscid far-field boundary condition
/// Provides differentiation of adjoint weighted residual w.r.t. mach number
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class FarFieldBCDiff : public InviscidBoundaryIntegrator<FarFieldBCDiff<dim, entvar>>
{
public:
    /// Constructs an integrator for far-field boundary flux differentiation
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] q_far - state at the far-field
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   FarFieldBCDiff(adept::Stack &diff_stack,
              mfem::GridFunction *adj,
              const mfem::FiniteElementCollection *fe_coll,
              const mfem::Vector q_far, 
              double mach, double aoa,
              double a = 1.0)
       : InviscidBoundaryIntegrator<FarFieldBCDiff<dim, entvar>>(
             diff_stack, fe_coll, dim+2, a), adjoint(adj), qfs(q_far), work_vec(dim+2),
             mach_fs(mach), aoa_fs(aoa) {}

   /// Not used (or, rather, *do not use*!)
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q) { return 0.0; }

   /// Compute an adjoint-consistent slip-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec);

      /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac){}

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac){}

    /// Compute derivative w.r.t. mach number parameter (AssembleFaceVector)
    virtual double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                const mfem::FiniteElement &el_unused,
                                mfem::FaceElementTransformations &trans,
                                const mfem::Vector &elfun) override;
private:
   /// Stores the far-field state
   mfem::Vector qfs;
   /// Work vector for boundary flux computation
   mfem::Vector work_vec;
   /// Stores the given adjoint vector
   mfem::GridFunction *adjoint;
   /// Far field flow parameters
   double mach_fs, aoa_fs;
};

#include "euler_diff_integ_def.hpp"

}

#endif