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
       : num_states(num_state_vars), alpha(a), stack(diff_stack),
         CDw_jac(Derived::ndim) {}

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
                                    mfem::DenseMatrix &elmat) {}

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
   /// used to store the (physical) space location of node i
   mfem::Vector xi;
   /// used to reference the entropy variables at node j
   mfem::Vector wj;
   /// used to reference the entropy variables at node j
   mfem::Vector uj;
   /// stores the product of c_{hat} matrices with Dwi
   mfem::Vector CDwi;
   /// stores the derivatives in all (physical) space directions at node i
   mfem::DenseMatrix Dwi;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJ_i;
   /// used to store the adjugate of the mapping Jacobian at node j
   mfem::DenseMatrix adjJ_j;
   /// used to store the adjugate of the mapping Jacobian at node j
   mfem::DenseMatrix adjJ_k;
   /// stores (num_state x num_state) Jacobian terms
   mfem::DenseMatrix jac_term;
   /// stores the derivative of scaled derivatives w.r.t. Dw
   std::vector<std::unique_ptr<mfem::DenseMatrix>> CDw_jac(dim);
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

   /// applies symmetric matrices \f$ C_{d,:}(u) \f$ to input `Dw`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices
   /// \param[in] x - coordinate location at which scaling is evaluated  
   /// \param[in] u - state at which the symmetric matrices `C` are evaluated
   /// \param[in] Dw - `Dw[:,d2]` stores derivative of `w` in direction `d2`. 
   /// \param[out] CDw - product of the multiplication between the `C` and `Dw`.
   /// \note This uses the CRTP, so it wraps call to `applyScaling` in Derived.
   void scale(int d, const mfem::Vector &x, const mfem::Vector &u,
              const mfem::DenseMatrix &Dw, mfem::Vector &CDw)
   {
      static_cast<Derived *>(this)->applyScaling(d, x, u, Dw, CDw);
   }

   void scaleJacState(int d, const mfem::Vector &x, const mfem::Vector &u,
                      const mfem::DenseMatrix &Dw, mfem::DenseMatrix &CDw_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacState(d, x, u, Dw, CDw_jac);
   }

   void scaleJacDw(int d, const mfem::Vector &x, const mfem::Vector &u,
                   const mfem::DenseMatrix &Dw,
                   mfem::Array<mfem::DenseMatrix> &CDw_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacDw(d, x, u, Dw, CDw_jac);
   }

};

/// Integrator for viscous boundary fluxes
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class ViscousBoundaryIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a boundary integrator based on a given boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousBoundaryIntegrator(adept::Stack &diff_stack,
                              const mfem::FiniteElementCollection *fe_coll,
                              int num_state_vars = 1, double a = 1.0)
       : num_states(num_state_vars), alpha(a), stack(diff_stack),
         fec(fe_coll) {}

	/// Construct the contribution to a functional from the boundary element
	/// \param[in] el_bnd - boundary element that contribute to the functional
	/// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \return element local contribution to functional
	virtual double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                const mfem::FiniteElement &el_unused,
                                mfem::FaceElementTransformations &trans,
                                const mfem::Vector &elfun) {}

   /// Construct the contribution to the element local residual
   /// \param[in] el_bnd - the finite element whose residual we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleFaceVector(const mfem::FiniteElement &el_bnd,
                                   const mfem::FiniteElement &el_unused,
                                   mfem::FaceElementTransformations &trans,
                                   const mfem::Vector &elfun,
                                   mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el_bnd - the finite element whose residual we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleFaceGrad(const mfem::FiniteElement &el_bnd,
                                 const mfem::FiniteElement &el_unused,
                                 mfem::FaceElementTransformations &trans,
                                 const mfem::Vector &elfun,
                                 mfem::DenseMatrix &elmat) {}

protected: 
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// used to select the appropriate face element
   const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
   /// used to reference the state at face node
   mfem::Vector u_face;
   /// stores the state at an arbitrary element node j
   mfem::Vector uj;
   /// stores the converted variables based on `uj` (usually entropy vars)
   mfem::Vector wj;
   /// store the physical location of a node
   mfem::Vector x;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// stores the flux evaluated by `bnd_flux`
   mfem::Vector flux_face;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJ_i;
   /// used to store the adjugate of the mapping Jacobian at node j
   mfem::DenseMatrix adjJ_j;
   /// stores the derivatives in all (physical) space directions at node i
   mfem::DenseMatrix Dwi;
   /// stores the jacobian of the flux with respect to the state at `u_face`
   mfem::DenseMatrix flux_jac_face;
#endif

   /// converts working variables to another set (e.g. conservative to entropy)
   /// \param[in] u - working states that are to be converted
   /// \param[out] w - transformed variables
   /// \note This uses the CRTP, so it wraps a call to `convertVars` in Derived.
   void convert(const mfem::Vector &u, mfem::Vector &w)
   {
      static_cast<Derived *>(this)->convertVars(u, w);
   }

   /// Compute a boundary flux function
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the flux
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \param[out] flux_vec - value of the flux
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   double flux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &u, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec)
   {
      static_cast<Derived*>(this)->calcFlux(x, dir, jac, u, Dw, flux_vec);
   }

#if 0
   /// Compute a scalar boundary function
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note `x` can be ignored depending on the function
   /// \note This uses the CRTP, so it wraps a call to `calcFunction` in
   /// Derived.
   double bndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                   const mfem::Vector &u)
   {
      return static_cast<Derived*>(this)->calcBndryFun(x, dir, u);
   }

   /// Compute a boundary flux function
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &x, const mfem::Vector &dir,
             const mfem::Vector &u, mfem::Vector &flux_vec)
   {
      static_cast<Derived*>(this)->calcFlux(x, dir, u, flux_vec);
   }

   /// Compute the Jacobian of the boundary flux function w.r.t. `u`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `u`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void fluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                     const mfem::Vector &u, mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived*>(this)->calcFluxJacState(x, dir, u, flux_jac);
   }
   
   /// Compute the Jacobian of the boundary flux function w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &x, const mfem::Vector &nrm,
                   const mfem::Vector &u, mfem::DenseMatrix &flux_dir)
   {
      static_cast<Derived*>(this)->calcFluxJacDir(x, nrm, u, flux_dir);
   }
#endif

};

#include "viscous_integ_def.hpp"

} // namespace mach

#endif