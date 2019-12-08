#ifndef MACH_VISCOUS_INTEG
#define MACH_VISCOUS_INTEG

#include "mfem.hpp"
#include "solver.hpp"
#include "sbp_fe.hpp" // needed in inviscid_integ_def.hpp

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
   /// stores the derivative w.r.t direction
   mfem::Vector Qwi;
   /// stores the product of c_{hat} with Qwi
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

// To do: we may want to move it to separate file
template <typename Derived>
void SymmetricViscousIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el, mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun, mfem::Vector &elvect)
{
   using namespace mfem;
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, wj, uj, Qwi, CQwd1d2;
   DenseMatrix w, adjJ_i, adjJ_j, adjJ_k;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   wj.SetSize(num_states);
   uj.SetSize(num_states);
   Qwi.SetSize(num_states);
   CQwd1d2.SetSize(num_states);
   w.SetSize(num_states, num_nodes);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   adjJ_k.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      Qwi = 0;
      // get the Jacobian (Trans.Weight) and cubature weight (node.weight)
      const IntegrationRule &ir = el.GetNodes();
      const IntegrationPoint &node = ir.IntPoint(i);
      Trans.SetIntPoint(&node);
      double norm = node.weight * Trans.Weight();
      double H = 1 / norm;
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      for (int d2 = 0; d2 < dim; ++d2)
      {
         for (int j = 0; j < num_nodes; ++j)
         {
            Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
            CalcAdjugate(Trans.Jacobian(), adjJ_j);
            double Qij = sbp.getQEntry(d2, i, j, adjJ_i, adjJ_j);
            u.GetRow(j, uj);
            w.GetColumnReference(j, wj);
            // Step 1: convert to entropy variables
            convert(uj, wj);
            // Step 2: find the derivative in `d2` direction
            for (int s = 0; s < num_states; ++s)
            {
               Qwi(s) += Qij * wj(s);
            }
         } // j node loop
         u.GetRow(i, ui);
         for (int d1 = 0; d1 < dim; ++d1)
         {
            // Step 3: apply the viscous coefficients' scaling
            scale(d1, d2, ui, Qwi, CQwd1d2);
            for (int k = 0; k < num_nodes; ++k)
            {
               Trans.SetIntPoint(&el.GetNodes().IntPoint(k));
               CalcAdjugate(Trans.Jacobian(), adjJ_k);
               double Qik = sbp.getQEntry(d1, i, k, adjJ_i, adjJ_k);
               // Step 4: apply derivative in `d1` direction
               // this evaluates Qd1'*C*(H^-1)*Qd2
               for (int s = 0; s < num_states; ++s)
               {
                  res(k, s) += alpha * Qik * H * CQwd1d2(s);
               }
            } // k loop
         }    // d1 loop
      }       //d2 loop
   }          // i node loop
}

#include "viscous_integ_def.hpp"

} // namespace mach

#endif