#ifndef MACH_NAVIER_STOKES_INTEG
#define MACH_NAVIER_STOKES_INTEG

#include "mfem.hpp"
#include "adept.h"
#include "viscous_integ.hpp"
#include "euler_fluxes.hpp"
#include "navier_stokes_fluxes.hpp"
using adept::adouble;

namespace mach
{

/// Entropy-stable volume integrator for Navier-Stokes viscous terms
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ESViscousIntegrator : public SymmetricViscousIntegrator<ESViscousIntegrator<dim>>
{
public:
   /// Construct a Viscous integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ESViscousIntegrator(adept::Stack &diff_stack, double Re_num, double Pr_num,
                       double a = 1.0)
       : SymmetricViscousIntegrator<ESViscousIntegrator<dim>>(
             diff_stack, dim + 2, a), Re(Re_num), Pr(Pr_num) {}

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

   /// applies symmetric matrices \f$ C_{d,:}(u) \f$ to input `Du`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices
   /// \param[in] u - state at which the symmetric matrices `C` are evaluated
   /// \param[in] Du - `Du[:,d2]` stores derivative of `u` in direction `d2`. 
   /// \param[out] CDu - product of the multiplication between the `C` and `Du`.
   void applyScaling(int d, const mfem::Vector &u, const mfem::DenseMatrix &Du,
                     mfem::Vector &CDu)
   {
      applyViscousScaling<double, dim>(d, Re, Pr, u.GetData(), Du.GetData(), 
                                       CDu.GetData());
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

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
};

/// Integrator for no-slip adiabatic-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class NoSlipAdiabaticWallBC : public ViscousBoundaryIntegrator<NoSlipAdiabaticWallBC<dim>>
{
public:
   /// Constructs an integrator for a no-slip, adiabatic boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   NoSlipAdiabaticWallBC(adept::Stack &diff_stack,
                         const mfem::FiniteElementCollection *fe_coll,
                         double Re_num, double Pr_num,
                         double a = 1.0)
       : ViscousBoundaryIntegrator<NoSlipAdiabaticWallBC<dim>>(
             diff_stack, fe_coll, dim + 2, a), Re(Re_num), Pr(Pr_num) {}

   /// Compute entropy stable no-slip, adiabatic wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dq,
                 mfem::Vector &flux_vec)
   {
      throw(-1);
      //calcSlipWallFlux<double,dim>(x.GetData(), dir.GetData(), q.GetData(),
      //                             flux_vec.GetData());
   }

};

} // namespace mach 

#endif