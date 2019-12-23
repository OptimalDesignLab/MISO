#ifndef MACH_NAVIER_STOKES_INTEG
#define MACH_NAVIER_STOKES_INTEG

#include "mfem.hpp"
#include "adept.h"
#include "viscous_integ.hpp"
#include "euler_fluxes.hpp"
#include "navier_stokes_fluxes.hpp"

using adept::adouble;
using namespace std; /// TODO: this is polluting other headers!

namespace mach
{

/// Entropy-stable volume integrator for Navier-Stokes viscous terms
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ESViscousIntegrator : public SymmetricViscousIntegrator<ESViscousIntegrator<dim>>
{
public:
   /// Construct an entropy-stable viscous integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] vis - nondimensional dynamic viscosity (use Sutherland if neg)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ESViscousIntegrator(adept::Stack &diff_stack, double Re_num, double Pr_num,
                       double vis = -1.0, double a = 1.0)
       : SymmetricViscousIntegrator<ESViscousIntegrator<dim>>(
             diff_stack, dim + 2, a),
         Re(Re_num), Pr(Pr_num), mu(vis) {}

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

   /// applies symmetric matrices \f$ C_{d,:}(q) \f$ to input `Dw`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices
   /// \param[in] x - coordinate location at which scaling evaluated (not used)
   /// \param[in] q - state at which the symmetric matrices `C` are evaluated
   /// \param[in] Dw - `Du[:,d2]` stores derivative of `w` in direction `d2`.
   /// \param[out] CDw - product of the multiplication between the `C` and `Dw`.
   void applyScaling(int d, const mfem::Vector &x, const mfem::Vector &q,
                     const mfem::DenseMatrix &Dw, mfem::Vector &CDw)
   {
      double mu_Re = mu;
      if (mu < 0.0)
      {
         mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
      }
      mu_Re /= Re;
      applyViscousScaling<double, dim>(d, mu_Re, Pr, q.GetData(), Dw.GetData(),
                                       CDw.GetData());
   }

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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

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
   /// \param[in] q_inflow - state at the inflow
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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

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

/// Integrator for viscous inflow boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ViscousInflowBC : public ViscousBoundaryIntegrator<ViscousInflowBC<dim>>
{
public:
   /// Constructs an integrator for a viscous inflow boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_inflow - state at the inflow
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousInflowBC(adept::Stack &diff_stack,
                   const mfem::FiniteElementCollection *fe_coll,
                   double Re_num, double Pr_num,
                   const mfem::Vector &q_inflow, double vis = -1.0,
                   double a = 1.0)
       : ViscousBoundaryIntegrator<ViscousInflowBC<dim>>(
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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

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
                      vector<mfem::DenseMatrix> &flux_jac);

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

/// Integrator for viscous outflow boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ViscousOutflowBC : public ViscousBoundaryIntegrator<ViscousOutflowBC<dim>>
{
public:
   /// Constructs an integrator for a viscous outflow boundary
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_outflow - state at the outflow
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousOutflowBC(adept::Stack &diff_stack,
                    const mfem::FiniteElementCollection *fe_coll,
                    double Re_num, double Pr_num,
                    const mfem::Vector &q_outflow, double vis = -1.0,
                    double a = 1.0)
       : ViscousBoundaryIntegrator<ViscousOutflowBC<dim>>(
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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

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
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
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
   /// Outflow boundary state
   mfem::Vector q_out;
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
   void convertVarsJacState(const mfem::Vector &q, mfem::DenseMatrix &dwdu);

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

#include "navier_stokes_integ_def.hpp"

} // namespace mach

#endif