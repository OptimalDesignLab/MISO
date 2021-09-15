#ifndef MACH_INVISCID_INTEG
#define MACH_INVISCID_INTEG

#include "mfem.hpp"

#include "sbp_fe.hpp"  // needed in inviscid_integ_def.hpp
#include "utils.hpp"

namespace mach
{
/// Integrator for one-point inviscid flux functions
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class InviscidIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for "inviscid" type fluxes
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - factor, usually used to move terms to rhs
   /// \note `num_state_vars` is not necessarily the same as the number of
   /// states used by, nor the number of fluxes returned by, `flux`.
   /// For example, there may be 5 states for the 2D RANS equations, but
   /// `flux` may use only the first 4.
   InviscidIntegrator(adept::Stack &diff_stack,
                      int num_state_vars = 1,
                      double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack)
   { }

   /// Get the contribution of this element to a functional
   /// \param[in] el - the finite element whose contribution we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                   mfem::ElementTransformation &trans,
                                   const mfem::Vector &elfun);

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
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
   /// the coordinates of node i
   mfem::Vector x_i;
   /// used to reference the states at node i
   mfem::Vector ui;
   /// used to reference the residual at node i
   mfem::Vector resi;
   /// stores a row of the adjugate of the mapping Jacobian
   mfem::Vector dxidx;
   /// stores the result of calling the flux function
   mfem::Vector fluxi;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJ_i;
   /// used to store the flux Jacobian at node i
   mfem::DenseMatrix flux_jaci;
   /// used to store the flux at each node
   mfem::DenseMatrix elflux;
   /// used to store the residual in (num_states, Dof) format
   mfem::DenseMatrix elres;
#endif

   /// Compute a scalar domain functional
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] u - state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note `x` can be ignored depending on the function
   /// \note This uses the CRTP, so it wraps a call to `calcVolFun` in Derived.
   double volFun(const mfem::Vector &x, const mfem::Vector &u)
   {
      return static_cast<Derived *>(this)->calcVolFun(x, u);
   }

   /// An inviscid flux function
   /// \param[in] dir - desired direction for the flux
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_vec - flux evaluated at `u` in direction `dir`
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &dir,
             const mfem::Vector &u,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(dir, u, flux_vec);
   }

   /// Compute the Jacobian of the flux function `flux` w.r.t. `u`
   /// \parma[in] dir - desired direction for the flux
   /// \param[in] u - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `u`
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void fluxJacState(const mfem::Vector &dir,
                     const mfem::Vector &u,
                     mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived *>(this)->calcFluxJacState(dir, u, flux_jac);
   }

   /// Compute the Jacobian of the flux function `flux` w.r.t. `dir`
   /// \parma[in] dir - desired direction for the flux
   /// \param[in] u - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &dir,
                   const mfem::Vector &u,
                   mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived *>(this)->calcFluxJacDir(dir, u, flux_jac);
   }
};

/// Integrator for two-point (dyadic) fluxes (e.g. Entropy Stable)
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class DyadicFluxIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct a two-point flux integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - factor, usually used to move terms to rhs
   /// \note `num_state_vars` is not necessarily the same as the number of
   /// states used by, nor the number of fluxes returned by, `flux`.
   /// For example, there may be 5 states for the 2D RANS equations, but
   /// `flux` may use only the first 4.
   DyadicFluxIntegrator(adept::Stack &diff_stack,
                        int num_state_vars = 1,
                        double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// two point flux function
   // void (*flux)(int di, const mfem::Vector &u_left, const mfem::Vector
   // &u_right,
   //                 mfem::Vector &flux_vec);
#ifndef MFEM_THREAD_SAFE
   /// used to reference the states at node i
   mfem::Vector ui;
   /// used to reference the states at node j
   mfem::Vector uj;
   /// stores the result of calling the flux function
   mfem::Vector fluxij;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJ_i;
   /// used to store the adjugate of the mapping Jacobian at node j
   mfem::DenseMatrix adjJ_j;
   /// stores a row of the adjugate of the mapping Jacobian
   mfem::Vector dxidx;
   /// stores the jacobian w.r.t left state
   mfem::DenseMatrix flux_jaci;
   /// stores the jacobian w.r.t left state
   mfem::DenseMatrix flux_jacj;

#endif

   /// A two point (i.e. dyadic) flux function
   /// \param[in] di - desired coordinate direction for flux
   /// \param[in] u_left - the "left" state
   /// \param[in] u_right - the "right" state
   /// \param[out] flux_vec - flux evaluated at `u_left` and `u_right`
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(int di,
             const mfem::Vector &u_left,
             const mfem::Vector &u_right,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(di, u_left, u_right, flux_vec);
   }

   /// Compute the Jacobians of `flux` with respect to `u_left` and `u_right`
   /// \param[in] di - desired coordinate direction for flux
   /// \param[in] u_left - the "left" state
   /// \param[in] u_right - the "right" state
   /// \param[out] jac_left - Jacobian of `flux` w.r.t. `u_left`
   /// \param[out] jac_right - Jacobian of `flux` w.r.t. `u_right`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacStates(int di,
                      const mfem::Vector &u_left,
                      const mfem::Vector &u_right,
                      mfem::DenseMatrix &jac_left,
                      mfem::DenseMatrix &jac_right)
   {
      static_cast<Derived *>(this)->calcFluxJacStates(
          di, u_left, u_right, jac_left, jac_right);
   }
};

/// Integrator for local projection stabilization
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class LPSIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an LPS integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] coeff - the LPS coefficient
   LPSIntegrator(adept::Stack &diff_stack,
                 int num_state_vars = 1,
                 double a = 1.0,
                 double coeff = 1.0)
    : num_states(num_state_vars), alpha(a), lps_coeff(coeff), stack(diff_stack)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// the LPS coefficient
   double lps_coeff;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
#ifndef MFEM_THREAD_SAFE
   /// used to reference the states at node i
   mfem::Vector ui;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJt;
   /// used to store the converted variables (for example)
   mfem::DenseMatrix w;
   /// used to store the projected converted variables (for example)
   mfem::DenseMatrix Pw;
   /// used to store the Jacobian of scale or convert
   mfem::DenseMatrix jac_term;
   /// used to hold a nodewise block for insertion in the element Jacobian
   mfem::DenseMatrix jac_node;
   /// used to hold the (i,j)th LPS matrix operator block entry
   mfem::DenseMatrix Lij;
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

   /// applies symmetric matrix `A(adjJ,u)` to input `v` to scale dissipation
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] u - state at which the symmetric matrix `A` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Av - product of the multiplication
   /// \note should not include LPS coefficient, which is applied separately
   /// \note This uses the CRTP, so it wraps call to `applyScaling` in Derived.
   void scale(const mfem::DenseMatrix &adjJ,
              const mfem::Vector &u,
              const mfem::Vector &v,
              mfem::Vector &Av)
   {
      static_cast<Derived *>(this)->applyScaling(adjJ, u, v, Av);
   }

   /// Computes the Jacobian of the product `A(adjJ,u)*v` w.r.t. `u`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] u - state at which the symmetric matrix `A` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Av_jac - Jacobian of product w.r.t. `u`
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void scaleJacState(const mfem::DenseMatrix &adjJ,
                      const mfem::Vector &u,
                      const mfem::Vector &v,
                      mfem::DenseMatrix &Av_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacState(adjJ, u, v, Av_jac);
   }

   /// Computes the Jacobian of the product `A(adjJ,u)*v` w.r.t. `adjJ`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] u - state at which the symmetric matrix `A` is evaluated
   /// \param[in] v - vector that is being multiplied
   /// \param[out] Av_jac - Jacobian of product w.r.t. `adjJ`
   /// \note `Av_jac` stores derivatives treating `adjJ` is a 1d array.
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void scaleJacAdjJ(const mfem::DenseMatrix &adjJ,
                     const mfem::Vector &u,
                     const mfem::Vector &v,
                     mfem::DenseMatrix &Av_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacAdjJ(adjJ, u, v, Av_jac);
   }

   /// Computes the Jacobian of the product `A(adjJ,u)*v` w.r.t. `v`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] u - state at which the symmetric matrix `A` is evaluated
   /// \param[out] Av_jac - Jacobian of product w.r.t. `v` (i.e. `A`)
   /// \note `Av_jac` stores derivatives treating `adjJ` is a 1d array.
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void scaleJacV(const mfem::DenseMatrix &adjJ,
                  const mfem::Vector &u,
                  mfem::DenseMatrix &Av_jac)
   {
      static_cast<Derived *>(this)->applyScalingJacV(adjJ, u, Av_jac);
   }
};

/// Integrator for inviscid boundary fluxes (fluxes that do not need gradient)
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class InviscidBoundaryIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a boundary integrator based on a given boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   InviscidBoundaryIntegrator(adept::Stack &diff_stack,
                              const mfem::FiniteElementCollection *fe_coll,
                              int num_state_vars = 1,
                              double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack), fec(fe_coll)
   { }

   /// Construct the contribution to a functional from the boundary element
   /// \param[in] el_bnd - boundary element that contribute to the functional
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \return element local contribution to functional
   virtual double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                const mfem::FiniteElement &el_unused,
                                mfem::FaceElementTransformations &trans,
                                const mfem::Vector &elfun);

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
                                 mfem::DenseMatrix &elmat);

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
   /// store the physical location of a node
   mfem::Vector x;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// stores the flux evaluated by `bnd_flux`
   mfem::Vector flux_face;
   /// stores the jacobian of the flux with respect to the state at `u_face`
   mfem::DenseMatrix flux_jac_face;
#endif

   /// Compute a scalar boundary function
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note `x` can be ignored depending on the function
   /// \note This uses the CRTP, so it wraps a call to `calcFunction` in
   /// Derived.
   double bndryFun(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &u)
   {
      return static_cast<Derived *>(this)->calcBndryFun(x, dir, u);
   }

   /// Compute a boundary flux function
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &x,
             const mfem::Vector &dir,
             const mfem::Vector &u,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(x, dir, u, flux_vec);
   }

   /// Compute the Jacobian of the boundary flux function w.r.t. `u`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `u`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void fluxJacState(const mfem::Vector &x,
                     const mfem::Vector &dir,
                     const mfem::Vector &u,
                     mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived *>(this)->calcFluxJacState(x, dir, u, flux_jac);
   }

   /// Compute the Jacobian of the boundary flux function w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &x,
                   const mfem::Vector &nrm,
                   const mfem::Vector &u,
                   mfem::DenseMatrix &flux_dir)
   {
      static_cast<Derived *>(this)->calcFluxJacDir(x, nrm, u, flux_dir);
   }
};

/// Integrator for inviscid interface fluxes (fluxes that do not need gradient)
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class InviscidFaceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a face integrator based on a given interface flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   InviscidFaceIntegrator(adept::Stack &diff_stack,
                          const mfem::FiniteElementCollection *fe_coll,
                          int num_state_vars = 1,
                          double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack), fec(fe_coll)
   { }

   /// Get the contribution from the interface to a functional
   /// \param[in] el_left - "left" element for functional contribution
   /// \param[in] el_right - "right" element for functional contribution
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - holds the solution on the adjacent elements
   virtual double GetFaceEnergy(const mfem::FiniteElement &el_left,
                                const mfem::FiniteElement &el_right,
                                mfem::FaceElementTransformations &trans,
                                const mfem::Vector &elfun);

   /// Construct the contribution to the element local residuals
   /// \param[in] el_left - "left" element whose residual we want to update
   /// \param[in] el_right - "right" element whose residual we want to update
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleFaceVector(const mfem::FiniteElement &el_left,
                                   const mfem::FiniteElement &el_right,
                                   mfem::FaceElementTransformations &trans,
                                   const mfem::Vector &elfun,
                                   mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el_left - "left" element whose residual we want to update
   /// \param[in] el_right - "right" element whose residual we want to update
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleFaceGrad(const mfem::FiniteElement &el_left,
                                 const mfem::FiniteElement &el_right,
                                 mfem::FaceElementTransformations &trans,
                                 const mfem::Vector &elfun,
                                 mfem::DenseMatrix &elmat);

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
   /// used to reference the left state at face node
   mfem::Vector u_face_left;
   /// used to reference the right state at face node
   mfem::Vector u_face_right;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// stores the flux evaluated by `bnd_flux`
   mfem::Vector flux_face;
   /// stores the jacobian of the flux with respect to the left state
   mfem::DenseMatrix flux_jac_left;
   /// stores the jacobian of the flux with respect to the right state
   mfem::DenseMatrix flux_jac_right;
#endif

   /// Compute a scalar interface function
   /// \param[in] dir - vector normal to the face
   /// \param[in] u_left - "left" state at which to evaluate the function
   /// \param[in] u_right - "right" state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note This uses the CRTP, so it wraps a call to `calcIFaceFun` in
   /// Derived.
   double iFaceFun(const mfem::Vector &dir,
                   const mfem::Vector &u_left,
                   const mfem::Vector &u_right)
   {
      return static_cast<Derived *>(this)->calcIFaceFun(dir, u_left, u_right);
   }

   /// Compute an interface flux function
   /// \param[in] dir - vector normal to the face
   /// \param[in] u_left - "left" state at which to evaluate the flux
   /// \param[in] u_right - "right" state at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &dir,
             const mfem::Vector &u_left,
             const mfem::Vector &u_right,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(dir, u_left, u_right, flux_vec);
   }

   /// Compute the Jacobian of the interface flux function w.r.t. states
   /// \param[in] dir - vector normal to the face
   /// \param[in] u_left - "left" state at which to evaluate the flux
   /// \param[in] u_right - "right" state at which to evaluate the flux
   /// \param[out] jac_left - Jacobian of `flux` w.r.t. `u_left`
   /// \param[out] jac_right - Jacobian of `flux` w.r.t. `u_right`
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void fluxJacStates(const mfem::Vector &dir,
                      const mfem::Vector &u_left,
                      const mfem::Vector &u_right,
                      mfem::DenseMatrix &jac_left,
                      mfem::DenseMatrix &jac_right)
   {
      static_cast<Derived *>(this)->calcFluxJacState(
          dir, u_left, u_right, jac_left, jac_right);
   }

   /// Compute the Jacobian of the interface flux function w.r.t. `dir`
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u_left - "left" state at which to evaluate the flux
   /// \param[in] u_right - "right" state at which to evaluate the flux
   /// \param[out] flux_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &dir,
                   const mfem::Vector &u_left,
                   const mfem::Vector &u_right,
                   mfem::DenseMatrix &flux_dir)
   {
      static_cast<Derived *>(this)->calcFluxJacDir(
          dir, u_left, u_right, flux_dir);
   }
};

/// Integrator for nonlinear temporal terms
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class NonlinearMassIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for nonlinear mass matrices
   /// \param[in] num_state_vars - the number of state variables (redundant)
   /// \param[in] a - factor, usually used to move terms to rhs
   NonlinearMassIntegrator(int num_state_vars = 1, double a = 1.0)
    : num_states(num_state_vars), alpha(a)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
#ifndef MFEM_THREAD_SAFE
   /// state at which conservative variables are to evaluated
   mfem::Vector u_i;
   /// holds the conservative variables at node i
   mfem::Vector q_i;
   /// stores the (dq/du) matrix for the Jacobian calculation
   mfem::DenseMatrix A_i;
#endif

   /// converts working variables to conservative variables
   /// \param[in] u - working states that are to be converted
   /// \param[out] q - conservative variables
   /// \note This uses the CRTP, so it wraps a call to `convertVars` in Derived.
   void convert(const mfem::Vector &u, mfem::Vector &q)
   {
      static_cast<Derived *>(this)->convertVars(u, q);
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] u - working states that are to be converted
   /// \param[out] dqdu - Jacobian of transformed variables w.r.t. `u`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void convertJacState(const mfem::Vector &u, mfem::DenseMatrix &dqdu)
   {
      static_cast<Derived *>(this)->convertVarsJacState(u, dqdu);
   }
};

#include "inviscid_integ_def.hpp"

}  // namespace mach

#endif
