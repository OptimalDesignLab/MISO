#ifndef MACH_RANS_INTEG
#define MACH_RANS_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "navier_stokes_integ.hpp"
#include "euler_fluxes.hpp"
#include "navier_stokes_fluxes.hpp"
#include "rans_fluxes.hpp"

using adept::adouble;
using namespace std; 

namespace mach
{

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
   SAInviscidIntegrator(adept::Stack &diff_stack,  double a = 1.0)
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
   mfem::Vector qfs;
};

/// Integrator for RANS SA Production, Destruction, Viscous-Like terms 
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SASourceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for RANS SA Production term 
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] distance - wall distance function projected onto a space
   /// \param[in] re_fs - freestream reynolds number
   /// \param[in] sa_params - Spalart-Allmaras model constants
   /// \param[in] vis - nondimensional dynamic viscosity (use Sutherland if neg)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] P, D - control production and destruction terms for debugging
   SASourceIntegrator(adept::Stack &diff_stack, mfem::GridFunction distance, double re_fs, 
                          mfem::Vector sa_params, double vis = -1.0, double a = 1.0,  double P = 1.0, double D = 1.0, double dmin = 1e-4)
       : alpha(a), d0(dmin), prod(P), dest(D), mu(vis), stack(diff_stack), num_states(dim+3), Re(re_fs), sacs(sa_params),
       dist(distance) {}

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


private:

protected:
   /// nondimensional dynamic viscosity
   double mu;
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// Freestream Reynolds number
   double Re;
   /// 0-wall distance to evaluate
   double d0;
   /// vector of SA model parameters
   mfem::Vector sacs;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// activate production and destruction terms
   double prod; double dest; 
#ifndef MFEM_THREAD_SAFE
   /// the coordinates of node i
   mfem::Vector xi;
   /// used to reference the states at node i 
   mfem::Vector ui;
   /// used to reference the conservative states at node i 
   mfem::Vector uci;
   /// used to reference the gradient of nu at node i
   mfem::Vector grad_nu_i;
   /// used to reference the gradient of rho at node i
   mfem::Vector grad_rho_i;
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
   /// wall distance function
   mfem::GridFunction dist;
#endif
};

/// Integrator for SA no-slip adiabatic-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SANoSlipAdiabaticWallBC : public ViscousBoundaryIntegrator<SANoSlipAdiabaticWallBC<dim>>
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
   SANoSlipAdiabaticWallBC(adept::Stack &diff_stack,
                         const mfem::FiniteElementCollection *fe_coll,
                         double Re_num, double Pr_num, mfem::Vector sa_params,
                         const mfem::Vector &q_ref, double vis = -1.0,
                         double a = 1.0)
       : ViscousBoundaryIntegrator<SANoSlipAdiabaticWallBC<dim>>(
             diff_stack, fe_coll, dim + 3, a),
         Re(Re_num), Pr(Pr_num), sacs(sa_params),
         qfs(q_ref), mu(vis),  work_vec(dim + 3) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w)
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
      w(dim+2) = q(dim+2);
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u' 
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      dwdu = 0.0;
      convertVarsJac<dim>(q, this->stack, dwdu);
      dwdu(dim+2,dim+2) = 1.0;
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

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x, const mfem::Vector &dir,
                   const mfem::Vector &q, mfem::DenseMatrix &flux_mat);
   // {
   //    flux_mat = 0.0;
   // }

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x, const mfem::Vector dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac);
   // {
   //    for (int d = 0; d < dim; ++d)
   //       flux_jac[d] = 0.0;
   // }

   /// Computes boundary node contribution to the surface force
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - stress at given point
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       double jac, const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw) {return 0.0;}

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
   /// vector of SA model parameters
   mfem::Vector sacs;
};

/// Integrator for SA viscous slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
/// \note This is the same as the inviscid slip wall, but it provides the
/// necessary entropy-variable gradient flux.
template <int dim>
class SAViscousSlipWallBC : public ViscousBoundaryIntegrator<SAViscousSlipWallBC<dim>>
{
public:
   /// Constructs an integrator for a viscous slip-wall boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAViscousSlipWallBC(adept::Stack &diff_stack,
                     const mfem::FiniteElementCollection *fe_coll,
                     double Re_num, double Pr_num, mfem::Vector sa_params,
                     double vis = -1.0, double a = 1.0)
       : ViscousBoundaryIntegrator<SAViscousSlipWallBC<dim>>(
             diff_stack, fe_coll, dim + 3, a),
         Re(Re_num), Pr(Pr_num), mu(vis),
         sacs(sa_params), work_vec(dim + 3) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w)
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
      w(dim+2) = q(dim+2);
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      dwdu = 0.0;
      convertVarsJac<dim>(q, this->stack, dwdu);
      dwdu(dim+2,dim+2) = 1.0;
   }
   /// Compute flux corresponding to a viscous slip-wall boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute jacobian of flux corresponding to a viscous slip-wall boundary
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

   /// Compute jacobian of flux corresponding to a viscous slip-wall boundary
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

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x, const mfem::Vector &dir,
                   const mfem::Vector &q, mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x, const mfem::Vector dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
         flux_jac[d] = 0.0;
   }

   /// Computes boundary node contribution to the surface force
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - stress at given point
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       double jac, const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw) {return 0.0;}

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// work space for flux computations
   mfem::Vector work_vec;
   /// vector of SA model parameters
   mfem::Vector sacs;

   mfem::Vector qfs;
};            

/// Integrator for SA far-field boundary conditions
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SAFarFieldBC : public ViscousBoundaryIntegrator<SAFarFieldBC<dim>>
{
public:
   /// Constructs an integrator for a SA far-field boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_far - state at the far-field
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAFarFieldBC(adept::Stack &diff_stack,
                     const mfem::FiniteElementCollection *fe_coll,
                     double Re_num, double Pr_num,
                     const mfem::Vector &q_far, double vis = -1.0,
                     double a = 1.0)
       : ViscousBoundaryIntegrator<SAFarFieldBC<dim>>(
             diff_stack, fe_coll, dim + 3, a),
         Re(Re_num), Pr(Pr_num), qfs(q_far), mu(vis), work_vec(dim + 2) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w)
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
      w(dim+2) = q(dim+2);
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      dwdu = 0.0;
      convertVarsJac<dim>(q, this->stack, dwdu);
      dwdu(dim+2,dim+2) = 1.0;
   }

   /// Compute flux corresponding to a SA inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);


   /// Compute jacobian of flux corresponding to a SA far-field boundary
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

   /// Compute jacobian of flux corresponding to a SA far-field boundary
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

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x, const mfem::Vector &dir,
                   const mfem::Vector &q, mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x, const mfem::Vector dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
         flux_jac[d] = 0.0;
   }

   /// Computes boundary node contribution to the surface force
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - stress at given point
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       double jac, const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw) {return 0.0;}

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


/// Integrator for SA inflow boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SAInflowBC : public ViscousBoundaryIntegrator<SAInflowBC<dim>>
{
public:
   /// Constructs an integrator for a SA inflow boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_inflow - state at the inflow
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAInflowBC(adept::Stack &diff_stack,
                   const mfem::FiniteElementCollection *fe_coll,
                   double Re_num, double Pr_num,
                   const mfem::Vector &q_inflow, double vis = -1.0,
                   double a = 1.0)
       : ViscousBoundaryIntegrator<SAInflowBC<dim>>(
             diff_stack, fe_coll, dim + 2, a),
         Re(Re_num), Pr(Pr_num),
         q_in(q_inflow), mu(vis), work_vec(dim + 2) {}

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

   /// Compute flux corresponding to a SA inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute jacobian of flux corresponding to a SA inflow boundary
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

   /// Compute jacobian of flux corresponding to a SA inflow boundary
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
                      vector<mfem::DenseMatrix> &flux_jac);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x, const mfem::Vector &dir,
                   const mfem::Vector &q, mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x, const mfem::Vector dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
         flux_jac[d] = 0.0;
   }

   /// Computes boundary node contribution to the surface force
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - stress at given point
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       double jac, const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw) {return 0.0;}

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// Inflow boundary state
   mfem::Vector q_in;
   /// work space for flux computations
   mfem::Vector work_vec;
};

/// Integrator for SA outflow boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SAOutflowBC : public ViscousBoundaryIntegrator<SAOutflowBC<dim>>
{
public:
   /// Constructs an integrator for a SA outflow boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_outflow - state at the outflow
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAOutflowBC(adept::Stack &diff_stack,
                    const mfem::FiniteElementCollection *fe_coll,
                    double Re_num, double Pr_num,
                    const mfem::Vector &q_outflow, double vis = -1.0,
                    double a = 1.0)
       : ViscousBoundaryIntegrator<SAOutflowBC<dim>>(
             diff_stack, fe_coll, dim + 2, a),
         Re(Re_num), Pr(Pr_num),
         q_out(q_outflow), mu(vis), work_vec(dim + 2) {}

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

   /// Compute flux corresponding to a SA inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir, double jac,
                 const mfem::Vector &q, const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute jacobian of flux corresponding to a SA inflow boundary
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

   /// Compute jacobian of flux corresponding to a SA inflow boundary
   /// w.r.t `entrpy-variables' derivatives`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDw(const mfem::Vector &x, const mfem::Vector &dir,
                      double jac, const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x, const mfem::Vector &dir,
                   const mfem::Vector &q, mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x, const mfem::Vector dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
         flux_jac[d] = 0.0;
   }

   /// Computes boundary node contribution to the surface force
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - stress at given point
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       double jac, const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw) {return 0.0;}

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// Outflow boundary state
   mfem::Vector q_out;
   /// work space for flux computations
   mfem::Vector work_vec;
};

/// Integrator for local-projection stabilization with SA variable
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class SALPSIntegrator : public LPSIntegrator<
                                   SALPSIntegrator<dim, entvar>>
{
public:
   /// Construct an LPS integrator with the SA variable
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] coeff - the LPS coefficient
   SALPSIntegrator(adept::Stack &diff_stack, double a = 1.0,
                          double coeff = 1.0)
       : LPSIntegrator<SALPSIntegrator<dim,entvar>>(
             diff_stack, dim + 3, a, coeff) {}

   /// converts state variables to entropy variables, if necessary
   /// \param[in] q - state variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w);

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

   /// Applies the matrix `dQ/dW` to `vec`, and scales by the avg. spectral radius
   /// \param[in] adjJ - the adjugate of the mapping Jacobian
   /// \param[in] q - the state at which `dQ/dW` and radius are to be evaluated
   /// \param[in] vec - the vector being multiplied
   /// \param[out] mat_vec - the result of the operation
   /// \warning adjJ must be supplied transposed from its `mfem` storage format,
   /// so we can use pointer arithmetic to access its rows.
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void applyScaling(const mfem::DenseMatrix &adjJ, const mfem::Vector &q,
                     const mfem::Vector &vec, mfem::Vector &mat_vec);

   /// Computes the Jacobian of the product `A(adjJ,q)*v` w.r.t. `q`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] q - state at which `dQ/dW` and radius are evaluated
   /// \param[in] vec - vector that is being multiplied
   /// \param[out] mat_vec_jac - Jacobian of product w.r.t. `q`
   /// \warning adjJ must be supplied transposed from its `mfem` storage format,
   /// so we can use pointer arithmetic to access its rows.
   void applyScalingJacState(const mfem::DenseMatrix &adjJ,
                             const mfem::Vector &q,
                             const mfem::Vector &vec,
                             mfem::DenseMatrix &mat_vec_jac);

   /// Computes the Jacobian of the product `A(adjJ,u)*v` w.r.t. `vec`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] q - state at which the symmetric matrix `A` is evaluated
   /// \param[out] mat_vec_jac - Jacobian of product w.r.t. `vec`
   /// \note `mat_vec_jac` stores derivatives treating `adjJ` is a 1d array.
   /// \note The size of `mat_vec_jac` must be set before calling this function
   void applyScalingJacV(const mfem::DenseMatrix &adjJ,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &mat_vec_jac);

};

/// Volume integrator for Navier-Stokes viscous terms with SA variable
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SAViscousIntegrator : public SymmetricViscousIntegrator<SAViscousIntegrator<dim>>
{
public:
   /// Construct an SA viscous integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] vis - nondimensional dynamic viscosity (use Sutherland if neg)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAViscousIntegrator(adept::Stack &diff_stack, double Re_num, double Pr_num,
                       mfem::Vector sa_params, double vis = -1.0, double a = 1.0)
       : SymmetricViscousIntegrator<SAViscousIntegrator<dim>>(
             diff_stack, dim + 3, a),
         Re(Re_num), Pr(Pr_num), mu(vis), sacs(sa_params) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w)
   {
      calcEntropyVars<double, dim>(q.GetData(), w.GetData());
      w(dim+2) = q(dim+2);
   }

   /// Compute the Jacobian of the mapping `convert` w.r.t. `u`
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] dwdu - Jacobian of entropy variables w.r.t. `u`
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu)
   {
      dwdu = 0.0;
      convertVarsJac<dim>(q, this->stack, dwdu);
      dwdu(dim+2, dim+2) = 1.0;
   }

   /// applies symmetric matrices \f$ C_{d,:}(q) \f$ to input `Dw`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices
   /// \param[in] x - coordinate location at which scaling evaluated (not used)
   /// \param[in] q - state at which the symmetric matrices `C` are evaluated
   /// \param[in] Dw - `Du[:,d2]` stores derivative of `w` in direction `d2`.
   /// \param[out] CDw - product of the multiplication between the `C` and `Dw`.
   void applyScaling(int d, const mfem::Vector &x, const mfem::Vector &q,
                     const mfem::DenseMatrix &Dw, mfem::Vector &CDw);

   /// Computes the Jacobian of the product `C(q)*Dw` w.r.t. `q`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices   
   /// \param[in] x - coordinate location at which scaling evaluated (not used)
   /// \param[in] q - state at which the symmetric matrix `C` is evaluated
   /// \param[in] Dw - vector that is being multiplied
   /// \param[out] CDw_jac - Jacobian of product w.r.t. `q`
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void applyScalingJacState(int d, const mfem::Vector &x,
                             const mfem::Vector &q,
                             const mfem::DenseMatrix &Dw,
                             mfem::DenseMatrix &CDw_jac);

   /// Computes the Jacobian of the product `C(q)*Dw` w.r.t. `Dw`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices   
   /// \param[in] x - coordinate location at which scaling evaluated (not used)
   /// \param[in] q - state at which the symmetric matrix `C` is evaluated
   /// \param[in] Dw - vector that is being multiplied
   /// \param[out] CDw_jac - Jacobian of product w.r.t. `Dw` (i.e. `C`)
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void applyScalingJacDw(int d, const mfem::Vector &x, const mfem::Vector &q,
                          const mfem::DenseMatrix &Dw, 
                          vector<mfem::DenseMatrix> &CDw_jac);

   /// This allows the base class to access the number of dimensions
   static const int ndim = dim;

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensional dynamic viscosity
   double mu;
   /// vector of SA model parameters
   mfem::Vector sacs;

   mfem::Vector qfs;
};


/// Source-term integrator for a 2D SA MMS problem
/// \note For details on the MMS problem, see the file rans_mms.py
class SAMMSIntegrator : public MMSIntegrator<SAMMSIntegrator>
{
public:
   /// Construct an integrator for a 2D RANS SA MMS source
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SAMMSIntegrator(double Re_num, double Pr_num, double a = -1.0)
       : MMSIntegrator<SAMMSIntegrator>(5, a),
         Re(Re_num), Pr(Pr_num) {}

   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcSource(const mfem::Vector &x, mfem::Vector &src)
   {
      double mu = 1.0/Re;
      calcSAMMS<double>(mu, Pr, x.GetData(), src.GetData());
   }

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
};

#include "rans_integ_def.hpp"

} // namespace mach

#endif