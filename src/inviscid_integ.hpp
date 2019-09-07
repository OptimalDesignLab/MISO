#ifndef MACH_INVISCID_INTEG
#define MACH_INVISCID_INTEG

#include "mfem.hpp"
#include "solver.hpp"

namespace mach
{

/// Integrator for one-point inviscid flux functions
class InviscidIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for "inviscid" type fluxes
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fluxFun - the flux function
   /// \param[in] num_state_vars - the number of state variables
   /// \note `num_state_vars` is not necessarily the same as the number of
   /// states used by, nor the number of fluxes returned by, `flux_function`.
   /// For example, there may be 5 states for the 2D RANS equations, but 
   /// `flux_function` may use only the first 4.
   InviscidIntegrator(adept::Stack &diff_stack,
                      void (*fluxFun)(const mfem::Vector &nrm,
                           const mfem::Vector &u, mfem::Vector &flux_vec),
                      int num_state_vars = 1, double a = 1.0)
      : num_states(num_state_vars), alpha(a), stack(diff_stack), flux(fluxFun)
   { }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   // virtual void AssembleElementGrad(const mfem::FiniteElement &el,
   //                                  mfem::ElementTransformation &Ttr,
   //                                  const mfem::Vector &elfun,
   //                                  mfem::DenseMatrix &elmat);

private:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// flux function
   void (*flux)(const mfem::Vector &nrm, const mfem::Vector &u,
         mfem::Vector &flux_vec);
#ifndef MFEM_THREAD_SAFE
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
   /// used to store the flux at each node
   mfem::DenseMatrix elflux;
   /// used to store the residual in (num_states, Dof) format
   mfem::DenseMatrix elres;
#endif
};


/// Integrator for two-point (dyadic) fluxes (e.g. Entropy Stable)
class DyadicFluxIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct a two-point flux integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fluxFun - the two-point (dyadic) flux function
   /// \param[in] num_state_vars - the number of state variables
   /// \note `num_state_vars` is not necessarily the same as the number of
   /// states used by, nor the number of fluxes returned by, `flux_function`.
   /// For example, there may be 5 states for the 2D RANS equations, but 
   /// `flux_function` may use only the first 4.
   DyadicFluxIntegrator(adept::Stack &diff_stack,
                        void (*fluxFun)(int di, const mfem::Vector &u_left,
                                        const mfem::Vector &u_right,
                                       mfem::Vector &flux_vec),
                        int num_state_vars = 1, double a = 1.0)
       : num_states(num_state_vars), alpha(a), stack(diff_stack),
         flux(fluxFun) {}

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   // virtual void AssembleElementGrad(const mfem::FiniteElement &el,
   //                                  mfem::ElementTransformation &Ttr,
   //                                  const mfem::Vector &elfun,
   //                                  mfem::DenseMatrix &elmat);

private:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// two point flux function
   void (*flux)(int di, const mfem::Vector &u_left, const mfem::Vector &u_right,
                    mfem::Vector &flux_vec);
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
#endif
};


/// Integrator for local projection stabilization
class LPSIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an LPS integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] convertVarsFun - maps working variables to new variables
   /// \param[in] applyScalingFun - performs matrix-scaling operation
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] coeff - the LPS coefficient
   LPSIntegrator(adept::Stack &diff_stack,
                 void (*convertVarsFun)(const double *u, double *w),
                 void (*applyScalingFun)(const double *adjJ, const double *u,
                                         const double *v, double *Av),
                 int num_state_vars = 1, double a = 1.0, double coeff = 1.0)
       : num_states(num_state_vars), alpha(a), lps_coeff(coeff),
         stack(diff_stack), convertVars(convertVarsFun),
         applyScaling(applyScalingFun) {}

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   // virtual void AssembleElementGrad(const mfem::FiniteElement &el,
   //                                  mfem::ElementTransformation &Ttr,
   //                                  const mfem::Vector &elfun,
   //                                  mfem::DenseMatrix &elmat);

private:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// the LPS coefficient
   double lps_coeff;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// converts working variables to another set (e.g. conservative to entropy)
   void (*convertVars)(const double *u, double *w);
   /// applies symmetric matrix `A(adjJ,u)` to input `v` to scale dissipation
   /// \note The input array `v` is multiplied by a symmetric matrix `A`, which
   /// may depend on the state `u` and adjugate `adjJ`.  The result is returned
   /// in `Av`.
   /// \note should note include LPS coefficient, which is stored separately
   void (*applyScaling)(const double *adjJ, const double *u, const double *v,
                        double *Av);
#ifndef MFEM_THREAD_SAFE
   /// used to reference the states at node i 
   mfem::Vector ui;
   /// used to store the adjugate of the mapping Jacobian at node i
   mfem::DenseMatrix adjJt;
   /// used to store the converted variables (for example)
   mfem::DenseMatrix w;
   /// used to store the projected converted variables (for example)
   mfem::DenseMatrix Pw;
#endif
};

/// Integrator for inviscid boundary fluxes (fluxes that do not need gradient)
class InviscidBoundaryIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a boundary integrator based on a given boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fluxFun - boundary flux function
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   InviscidBoundaryIntegrator(adept::Stack &diff_stack,
                              void (*fluxFun)(const double *x,
                                              const double *nrm,
                                              const double *u,
                                              double *flux_vec),
                              const mfem::FiniteElementCollection *fe_coll,
                              int num_state_vars = 1, double a = 1.0)
       : num_states(num_state_vars), alpha(a), stack(diff_stack),
         bnd_flux(fluxFun), fec(fe_coll) {}

   /// Construct the contribution to the element local residual
   /// \param[in] el_bnd - the finite element whose residual we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleFaceVector(const mfem::FiniteElement &el_bnd,
                                   const mfem::FiniteElement &el_unused,
                                   mfem::FaceElementTransformations &trans,
                                   const mfem::Vector &elfun,
                                   mfem::Vector &elvect);

private: 
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// flux function used on the given boundary
   void (*bnd_flux)(const double *x, const double *nrm, const double *u,
                    double *flux_vec);
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
#endif
};

} // namespace mach

#endif