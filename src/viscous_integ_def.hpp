//To do: All Assemble stuff will go here
#ifndef MACH_VISCOUS_INTEG_DEF
#define MACH_VISCOUS_INTEG_DEF

#include "mfem.hpp"
#include "adept.h"
#include "viscous_integ.hpp"
#include "navier_stokes_fluxes.hpp"
using adept::adouble;

namespace mach
{
/// Integrator for viscous terms
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ViscousIntegrator : public SymmetricViscousIntegrator<ViscousIntegrator<dim>>
{
public:
   /// Construct a Viscous integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : SymmetricViscousIntegrator<ViscousIntegrator<dim>>(
             diff_stack, dim + 2, a) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w)
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

   /// applies symmetric matrix `C(u)` to input `v`
   /// \param[in] i - index `i` in `Cij` matrix
   /// \param[in] j - index `j` in `Cij` matrix
   /// \param[in] u - state at which the symmetric matrix `C` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Cv - product of the multiplication
   /// \note This uses the CRTP, so it wraps call to `applyScaling` in Derived.
   void applyScaling(const int i, const int j,
                     const mfem::Vector &u, const mfem::Vector &v, mfem::Vector &Cv)
   {
      applyViscousScaling<double, dim>(i, j, u.GetData(),
                                       v.GetData(), Cv.GetData());
   }

   /// Computes the Jacobian of the product `C(u)*v` w.r.t. `u`
   /// \param[in] u - state at which the symmetric matrix `C` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Cv_jac - Jacobian of product w.r.t. `u`
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void applyScalingJacState(const mfem::Vector &u,
                             const mfem::Vector &v,
                             mfem::DenseMatrix &Cv_jac);

   /// Computes the Jacobian of the product `C(u)*v` w.r.t. `v`
   /// \param[in] u - state at which the symmetric matrix `C` is evaluated
   /// \param[out] Cv_jac - Jacobian of product w.r.t. `v` (i.e. `C`)
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void applyScalingJacV(const mfem::Vector &u, mfem::DenseMatrix &Cv_jac);
};

} // namespace mach
#endif