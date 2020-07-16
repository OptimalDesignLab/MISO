#ifndef MACH_RANS_INTEG
#define MACH_RANS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "navier_stokes_integ.hpp"
#include "euler_fluxes.hpp"
#include "navier_stokes_fluxes.hpp"
#include "rans_fluxes.hpp"

using adept::adouble;
using namespace std; /// TODO: this is polluting other headers!

namespace mach
{

/// Entropy-stable volume integrator for Navier-Stokes viscous terms
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ESViscousSAIntegrator : public ESViscousIntegrator <dim>
{
public:
   /// Construct an entropy-stable viscous integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] vis - nondimensional dynamic viscosity (use Sutherland if neg)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ESViscousSAIntegrator(adept::Stack &diff_stack, double Re_num, double Pr_num,
                       double vis = -1.0, double a = 1.0)
       : ESViscousIntegrator <dim>(
             diff_stack, Re_num, Pr_num, vis, a) {}

private:

};

/// Volume integrator for inviscid terms, including SA variable
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class SAInviscidIntegrator : public DyadicFluxIntegrator<
                                 SAInviscidIntegrator<dim, entvar>>
{
public:
   /// Construct an inviscid integrator with SA terms
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAInviscidIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : DyadicFluxIntegrator<SAInviscidIntegrator<dim, entvar>>(
            diff_stack, dim+3, a) {}


   /// Ismail-Roe two-point (dyadic) flux function with additional variable
   /// \param[in] di - physical coordinate direction in which flux is wanted
   /// \param[in] qL - state variables at "left" state
   /// \param[in] qR - state variables at "right" state
   /// \param[out] flux - fluxes in the direction `di`
   /// \note This is simply a wrapper for the function in `euler_fluxes.hpp`
   void calcFlux(int di, const mfem::Vector &qL,
                 const mfem::Vector &qR, mfem::Vector &flux);

   /// Compute the Jacobians of `flux` with respect to `u_left` and `u_right`
   /// \param[in] di - desired coordinate direction for flux
   /// \param[in] qL - the "left" state
   /// \param[in] qR - the "right" state
   /// \param[out] jacL - Jacobian of `flux` w.r.t. `qL`
   /// \param[out] jacR - Jacobian of `flux` w.r.t. `qR`
   void calcFluxJacStates(int di, const mfem::Vector &qL,
                          const mfem::Vector &qR,
                          mfem::DenseMatrix &jacL,
                          mfem::DenseMatrix &jacR);
private:

};

/// Integrator for RANS SA Production term 
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SASourceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for RANS SA Production term 
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] 
   /// \param[in] vis - nondimensional dynamic viscosity (use Sutherland if neg)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SASourceIntegrator(adept::Stack &diff_stack, mfem::GridFunction dist,
                          mfem::Vector sa_params, double vis = -1.0, double a = 1.0)
       : alpha(a), mu(vis), stack(diff_stack), num_states(dim+3), sacs(sa_params) {}

   /// Construct the element local residual
   /// \param[in] fe - the finite element whose residual we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &fe,
                                      mfem::ElementTransformation &Trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] fe - the finite element whose Jacobian we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &fe,
                                    mfem::ElementTransformation &Trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

   // Compute vorticity on an SBP element, needed for SA model terms
   /// \param[in] q - the state over the element
   /// \param[in] sbp - the sbp element whose shape functions we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[out] curl - the curl of the velocity field at each node/int point
   void calcVorticitySBP(const mfem::DenseMatrix &q, 
                         const mfem::FiniteElement &fe, 
                         mfem::ElementTransformation &Trans, 
                         mfem::DenseMatrix curl);

   // Compute gradient for the turbulence variable on an SBP element, 
   // needed for SA model terms
   /// \param[in] q - the state over the element
   /// \param[in] sbp - the sbp element whose shape functions we want
   /// \param[in] Trans - defines the reference to physical element mapping
   /// \param[out] grad - the gradient of the turbulence variable at each node
   void calcGradSBP(const mfem::DenseMatrix &q, 
                    const mfem::FiniteElement &fe, 
                    mfem::ElementTransformation &Trans, 
                    mfem::DenseMatrix grad);

private:

protected:
   /// nondimensional dynamic viscosity
   double mu;
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// vector of SA model parameters
   mfem::Vector sacs;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
#ifndef MFEM_THREAD_SAFE
   /// the coordinates of node i
   mfem::Vector xi;
   /// used to reference the states at node i 
   mfem::Vector ui;
   /// used to reference the gradient at node i
   mfem::Vector grad_i;
   /// used to reference the curl at node i
   mfem::Vector curl_i;
   /// stores the result of calling the flux function
   mfem::Vector fluxi;
   /// used to store the gradient on the element
   mfem::DenseMatrix grad;
   /// used to store the gradient on the element
   mfem::DenseMatrix curl;
   /// used to store the flux Jacobian at node i
   mfem::DenseMatrix flux_jaci;
   /// used to store the flux at each node
   mfem::DenseMatrix elflux;
   /// used to store the residual in (num_states, Dof) format
   mfem::DenseMatrix elres;
#endif
};

#if 0
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
   /// \param[in] q_ref - a reference state (needed by penalty)
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   NoSlipAdiabaticWallBC(adept::Stack &diff_stack,
                         const mfem::FiniteElementCollection *fe_coll,
                         double Re_num, double Pr_num,
                         const mfem::Vector &q_ref, double vis = -1.0,
                         double a = 1.0)
       : ViscousBoundaryIntegrator<NoSlipAdiabaticWallBC<dim>>(
             diff_stack, fe_coll, dim + 2, a),
         Re(Re_num), Pr(Pr_num),
         qfs(q_ref), mu(vis), work_vec(dim + 2) {}

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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      convertVarsJac<dim>(q, this->stack, dwdu);
   }

   /// Compute entropy-stable, no-slip, adiabatic-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute Jacobian of entropy-stable, no-slip, adiabatic-wall boundary flux
   /// w.r.t states
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux w.r.t. states
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                         const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac);

   /// Compute Jacobian of entropy-stable, no-slip, adiabatic-wall boundary flux
   /// w.r.t vector of entropy-variables' derivatives
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux w.r.t. entropy-variables' derivatives
   void calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                      const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// Fixed state used to compute no-slip penalty matrix
   mfem::Vector qfs;
   /// work space for flux computations
   mfem::Vector work_vec;
};

/// Integrator for viscous slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
/// \note This is the same as the inviscid slip wall, but it provides the
/// necessary entropy-variable gradient flux.
template <int dim>
class ViscousSlipWallBC : public ViscousBoundaryIntegrator<ViscousSlipWallBC<dim>>
{
public:
   /// Constructs an integrator for a viscous inflow boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousSlipWallBC(adept::Stack &diff_stack,
                     const mfem::FiniteElementCollection *fe_coll,
                     double Re_num, double Pr_num, double vis = -1.0,
                     double a = 1.0)
       : ViscousBoundaryIntegrator<ViscousSlipWallBC<dim>>(
             diff_stack, fe_coll, dim + 2, a),
         Re(Re_num), Pr(Pr_num), mu(vis), work_vec(dim + 2) {}

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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      convertVarsJac<dim>(q, this->stack, dwdu);
   }
   /// Compute flux corresponding to a viscous inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute jacobian of flux corresponding to a viscous inflow boundary
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated 
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         double jac, const mfem::Vector &q,
                         const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac);

   /// Compute jacobian of flux corresponding to a viscous inflow boundary
   /// w.r.t `entrpy-variables' derivatives`
   /// \param[in] x - coordinate location at which flux is evaluated 
   /// \param[in] dir - vector normal to the boundary at `x` 
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables 
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir,
                      double jac, const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// work space for flux computations
   mfem::Vector work_vec;
};

/// Integrator for viscous far-field boundary conditions
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ViscousFarFieldBC : public ViscousBoundaryIntegrator<ViscousFarFieldBC<dim>>
{
public:
   /// Constructs an integrator for a viscous far-field boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_far - state at the far-field
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousFarFieldBC(adept::Stack &diff_stack,
                     const mfem::FiniteElementCollection *fe_coll,
                     double Re_num, double Pr_num,
                     const mfem::Vector &q_far, double vis = -1.0,
                     double a = 1.0)
       : ViscousBoundaryIntegrator<ViscousFarFieldBC<dim>>(
             diff_stack, fe_coll, dim + 2, a),
         Re(Re_num), Pr(Pr_num),
         qfs(q_far), mu(vis), work_vec(dim + 2) {}

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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      convertVarsJac<dim>(q, this->stack, dwdu);
   }

   /// Compute flux corresponding to a viscous inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec)
   {
      calcBoundaryFlux<double, dim>(dir.GetData(), qfs.GetData(), q.GetData(),
                                    work_vec.GetData(), flux_vec.GetData());
   }

   /// Compute jacobian of flux corresponding to a viscous far-field boundary
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated 
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         double jac, const mfem::Vector &q,
                         const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac);

   /// Compute jacobian of flux corresponding to a viscous far-field boundary
   /// w.r.t `entrpy-variables' derivatives`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir,
                      double jac, const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// far-field boundary state
   mfem::Vector qfs;
   /// work space for flux computations
   mfem::Vector work_vec;
};
#endif

#include "rans_integ_def.hpp"

} // namespace mach

#endif