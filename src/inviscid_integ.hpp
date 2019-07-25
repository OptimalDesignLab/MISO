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
                      void (*fluxFun)(const double *nrm, const double *u,
                                      double *flux_vec),
                      int num_state_vars = 1, double a = 1.0)
       : stack(diff_stack), flux(fluxFun), num_states(num_state_vars),
         alpha(a) {}

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
   void (*flux)(const double *nrm, const double *u, double *flux_vec);
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
                        void (*fluxFun)(int di, const double *u_left,
                                        const double *u_right,
                                        double *flux_vec),
                        int num_state_vars = 1, double a = 1.0)
       : stack(diff_stack), flux(fluxFun), num_states(num_state_vars),
         alpha(a) {}

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
   void (*flux)(int di, const double *u_left, const double *u_right,
                    double *flux_vec);
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
   ///
   LPSIntegrator(adept::Stack &diff_stack,
                 void (*convertVarsFun)(const double *u, double *w),
                 void (*applyScalingFun)(const double *adjJ, const double *u,
                                         const double *v, double *Av),
                 int num_state_vars = 1, double a = 1.0, double coeff = 1.0)
       : stack(diff_stack), convertVars(convertVarsFun),
         applyScaling(applyScalingFun), num_states(num_state_vars), alpha(a),
         lps_coeff(coeff) {}

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

} // namespace mach

#endif