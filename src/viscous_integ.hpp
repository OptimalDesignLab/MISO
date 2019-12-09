#ifndef MACH_VISCOUS_INTEG
#define MACH_VISCOUS_INTEG

#include "mfem.hpp"
#include "solver.hpp"
#include "sbp_fe.hpp" // needed in viscous_integ_def.hpp

namespace mach
{

/// Integrator for symmetric viscous terms
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class SymmetricViscousIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct a SymmetricViscousIntegrator integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SymmetricViscousIntegrator(adept::Stack &diff_stack, int num_state_vars = 1,
                              double a = 1.0)
       : num_states(num_state_vars), alpha(a), stack(diff_stack) {}

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &Trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
#ifndef MFEM_THREAD_SAFE
   /// used to reference the states at node i
   mfem::Vector ui;
   /// used to reference the entropy variables at node j
   mfem::Vector wj;
   /// used to reference the entropy variables at node j
   mfem::Vector uj;
   /// stores the derivative w.r.t space direction at node i
   mfem::Vector Qwi;
   /// stores the product of c_{hat} matrices with Qwi
   mfem::Vector CQwd1d2;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJ_i;
   /// used to store the adjugate of the mapping Jacobian at node j
   mfem::DenseMatrix adjJ_j;
   /// used to store the adjugate of the mapping Jacobian at node j
   mfem::DenseMatrix adjJ_k;
   /// used to store the converted variables (for example)
   mfem::DenseMatrix w;
#endif

   /// converts working variables to another set (e.g. conservative to entropy)
   /// \param[in] u - working states that are to be converted
   /// \param[out] w - transformed variables
   /// \note This uses the CRTP, so it wraps a call to `convertVars` in Derived.
   void convert(const mfem::Vector &u, mfem::Vector &w)
   {
      static_cast<Derived *>(this)->convertVars(u, w);
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] u - working states that are to be converted
   /// \param[out] dwdu - Jacobian of transformed variables w.r.t. `u`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void convertJacState(const mfem::Vector &u, mfem::DenseMatrix &dwdu)
   {
      static_cast<Derived *>(this)->convertVarsJacState(u, dwdu);
   }

   /// applies symmetric matrix `C(u)` to input `v`
   /// \param[in] i - index `i` in `Cij` matrix
   /// \param[in] j - index `j` in `Cij` matrix
   /// \param[in] u - state at which the symmetric matrix `C` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Cv - product of the multiplication
   /// \note This uses the CRTP, so it wraps call to `applyScaling` in Derived.
   void scale(int i, int j, const mfem::Vector &u, const mfem::Vector &v, mfem::Vector &Cv)
   {
      static_cast<Derived *>(this)->applyScaling(i, j, u, v, Cv);
   }

   /// Computes the Jacobian of the product `C(u)*v` w.r.t. `u`
   /// \param[in] u - state at which the symmetric matrix `C` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Cv_jac - Jacobian of product w.r.t. `u`
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void scaleJacState(const mfem::Vector &u, const mfem::Vector &v,
                      mfem::DenseMatrix &Cv_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacState(u, v, Cv_jac);
   }

   /// Computes the Jacobian of the product `C(u)*v` w.r.t. `v`
   /// \param[in] u - state at which the symmetric matrix `C` is evaluated
   /// \param[out] Cv_jac - Jacobian of product w.r.t. `v` (i.e. `C`)
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void scaleJacV(const mfem::Vector &u, mfem::DenseMatrix &Cv_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacV(u, Cv_jac);
   }
};

#include "viscous_integ_def.hpp"

} // namespace mach

#endif