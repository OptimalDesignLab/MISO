#ifndef MACH_NAVIER_STOKES_INTEG
#define MACH_NAVIER_STOKES_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "viscous_integ.hpp"
#include "mms_integ.hpp"
#include "euler_fluxes.hpp"
#include "navier_stokes_fluxes.hpp"

using adept::adouble;
using namespace std;  /// TODO: this is polluting other headers!

namespace mach
{
/// Source-term integrator for a 2D Navier-Stokes MMS problem
/// \note For details on the MMS problem, see the file viscous_mms.py
class NavierStokesMMSIntegrator
 : public MMSIntegrator<NavierStokesMMSIntegrator>
{
public:
   /// Construct an integrator for a 2D Navier-Stokes MMS source
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   NavierStokesMMSIntegrator(double Re_num, double Pr_num, double a = -1.0)
    : MMSIntegrator<NavierStokesMMSIntegrator>(4, a), Re(Re_num), Pr(Pr_num)
   { }

   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcSource(const mfem::Vector &x, mfem::Vector &src) const
   {
      double mu = 1.0 / Re;
      calcViscousMMS<double>(mu, Pr, x.GetData(), src.GetData());
   }

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
};

/// Entropy-stable volume integrator for Navier-Stokes viscous terms
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ESViscousIntegrator
 : public SymmetricViscousIntegrator<ESViscousIntegrator<dim>>
{
public:
   /// Construct an entropy-stable viscous integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] vis - nondimensional dynamic viscosity (use Sutherland if neg)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ESViscousIntegrator(adept::Stack &diff_stack,
                       double Re_num,
                       double Pr_num,
                       double vis = -1.0,
                       double a = 1.0)
    : SymmetricViscousIntegrator<ESViscousIntegrator<dim>>(diff_stack,
                                                           dim + 2,
                                                           a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis)
   { }

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

   /// applies symmetric matrices \f$ C_{d,:}(q) \f$ to input `Dw`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices
   /// \param[in] x - coordinate location at which scaling evaluated (not used)
   /// \param[in] q - state at which the symmetric matrices `C` are evaluated
   /// \param[in] Dw - `Du[:,d2]` stores derivative of `w` in direction `d2`.
   /// \param[out] CDw - product of the multiplication between the `C` and `Dw`.
   void applyScaling(int d,
                     const mfem::Vector &x,
                     const mfem::Vector &q,
                     const mfem::DenseMatrix &Dw,
                     mfem::Vector &CDw)
   {
      double mu_Re = mu;
      if (mu < 0.0)
      {
         mu_Re = calcSutherlandViscosity<double, dim>(q.GetData());
      }
      mu_Re /= Re;
      applyViscousScaling<double, dim>(
          d, mu_Re, Pr, q.GetData(), Dw.GetData(), CDw.GetData());
   }

   /// Computes the Jacobian of the product `C(q)*Dw` w.r.t. `q`
   /// \param[in] d - index `d` in \f$ C_{d,:} \f$ matrices
   /// \param[in] x - coordinate location at which scaling evaluated (not used)
   /// \param[in] q - state at which the symmetric matrix `C` is evaluated
   /// \param[in] Dw - vector that is being multiplied
   /// \param[out] CDw_jac - Jacobian of product w.r.t. `q`
   /// \note This uses the CRTP, so it wraps call to a func. in Derived.
   void applyScalingJacState(int d,
                             const mfem::Vector &x,
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
   void applyScalingJacDw(int d,
                          const mfem::Vector &x,
                          const mfem::Vector &q,
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
class NoSlipAdiabaticWallBC
 : public ViscousBoundaryIntegrator<NoSlipAdiabaticWallBC<dim>>
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
                         double Re_num,
                         double Pr_num,
                         const mfem::Vector &q_ref,
                         double vis = -1.0,
                         double a = 1.0)
    : ViscousBoundaryIntegrator<NoSlipAdiabaticWallBC<dim>>(diff_stack,
                                                            fe_coll,
                                                            dim + 2,
                                                            a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      qfs(q_ref),
      work_vec(dim + 2)
   { }

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

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw);

   /// Compute entropy-stable, no-slip, adiabatic-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 double jac,
                 const mfem::Vector &q,
                 const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &q,
                   mfem::DenseMatrix &flux_mat);

   /// Compute Jacobian of entropy-stable, no-slip, adiabatic-wall boundary flux
   /// w.r.t states
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux w.r.t. states
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         double jac,
                         const mfem::Vector &q,
                         const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac);

   /// Compute Jacobian of entropy-stable, no-slip, adiabatic-wall boundary flux
   /// w.r.t vector of entropy-variables' derivatives
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux w.r.t. entropy-variables'
   /// derivatives
   void calcFluxJacDw(const mfem::Vector &x,
                      const mfem::Vector &dir,
                      double jac,
                      const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &q,
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
class ViscousSlipWallBC
 : public ViscousBoundaryIntegrator<ViscousSlipWallBC<dim>>
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
                     double Re_num,
                     double Pr_num,
                     double vis = -1.0,
                     double a = 1.0)
    : ViscousBoundaryIntegrator<ViscousSlipWallBC<dim>>(diff_stack,
                                                        fe_coll,
                                                        dim + 2,
                                                        a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      work_vec(dim + 2)
   { }

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

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw);

   /// Compute flux corresponding to a viscous inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 double jac,
                 const mfem::Vector &q,
                 const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &q,
                   mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute jacobian of flux corresponding to a viscous inflow boundary
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         double jac,
                         const mfem::Vector &q,
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
   void calcFluxJacDw(const mfem::Vector &x,
                      const mfem::Vector &dir,
                      double jac,
                      const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
      {
         flux_jac[d] = 0.0;
      }
   }

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
                   double Re_num,
                   double Pr_num,
                   const mfem::Vector &q_inflow,
                   double vis = -1.0,
                   double a = 1.0)
    : ViscousBoundaryIntegrator<ViscousInflowBC<dim>>(diff_stack,
                                                      fe_coll,
                                                      dim + 2,
                                                      a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      q_in(q_inflow),
      work_vec(dim + 2)
   { }

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

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw);

   /// Compute flux corresponding to a viscous inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 double jac,
                 const mfem::Vector &q,
                 const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &q,
                   mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute jacobian of flux corresponding to a viscous inflow boundary
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         double jac,
                         const mfem::Vector &q,
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
   void calcFluxJacDw(const mfem::Vector &x,
                      const mfem::Vector &dir,
                      double jac,
                      const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      vector<mfem::DenseMatrix> &flux_jac);

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
      {
         flux_jac[d] = 0.0;
      }
   }

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
                    double Re_num,
                    double Pr_num,
                    const mfem::Vector &q_outflow,
                    double vis = -1.0,
                    double a = 1.0)
    : ViscousBoundaryIntegrator<ViscousOutflowBC<dim>>(diff_stack,
                                                       fe_coll,
                                                       dim + 2,
                                                       a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      q_out(q_outflow),
      work_vec(dim + 2)
   { }

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

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw);

   /// Compute flux corresponding to a viscous inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 double jac,
                 const mfem::Vector &q,
                 const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &q,
                   mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute jacobian of flux corresponding to a viscous inflow boundary
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         double jac,
                         const mfem::Vector &q,
                         const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac);

   /// Compute jacobian of flux corresponding to a viscous inflow boundary
   /// w.r.t `entrpy-variables' derivatives`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDw(const mfem::Vector &x,
                      const mfem::Vector &dir,
                      double jac,
                      const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
      {
         flux_jac[d] = 0.0;
      }
   }

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
class ViscousFarFieldBC
 : public ViscousBoundaryIntegrator<ViscousFarFieldBC<dim>>
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
                     double Re_num,
                     double Pr_num,
                     const mfem::Vector &q_far,
                     double vis = -1.0,
                     double a = 1.0)
    : ViscousBoundaryIntegrator<ViscousFarFieldBC<dim>>(diff_stack,
                                                        fe_coll,
                                                        dim + 2,
                                                        a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      qfs(q_far),
      work_vec(dim + 2)
   { }

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

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw);

   /// Compute flux corresponding to a viscous inflow boundary
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 double jac,
                 const mfem::Vector &q,
                 const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec)
   {
      calcBoundaryFlux<double, dim>(dir.GetData(),
                                    qfs.GetData(),
                                    q.GetData(),
                                    work_vec.GetData(),
                                    flux_vec.GetData());
   }

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &q,
                   mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute jacobian of flux corresponding to a viscous far-field boundary
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         double jac,
                         const mfem::Vector &q,
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
   void calcFluxJacDw(const mfem::Vector &x,
                      const mfem::Vector &dir,
                      double jac,
                      const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
      {
         flux_jac[d] = 0.0;
      }
   }

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

/// Integrator for exact, prescribed BCs (with zero normal derivative)
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class ViscousExactBC : public ViscousBoundaryIntegrator<ViscousExactBC<dim>>
{
public:
   /// Constructs an integrator for a viscous exact BCs
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_far - state at the far-field
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   ViscousExactBC(adept::Stack &diff_stack,
                  const mfem::FiniteElementCollection *fe_coll,
                  double Re_num,
                  double Pr_num,
                  void (*fun)(const mfem::Vector &, mfem::Vector &),
                  double vis = -1.0,
                  double a = 1.0)
    : ViscousBoundaryIntegrator<ViscousExactBC<dim>>(diff_stack,
                                                     fe_coll,
                                                     dim + 2,
                                                     a),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      qexact(dim + 2),
      work_vec(dim + 2)
   {
      exactSolution = fun;
   }

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

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &u,
                       const mfem::DenseMatrix &Dw);

   /// Compute flux corresponding to an exact solution
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 double jac,
                 const mfem::Vector &q,
                 const mfem::DenseMatrix &Dw,
                 mfem::Vector &flux_vec);

   /// Compute flux terms that are multiplied by test-function derivative
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[out] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcFluxDv(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &q,
                   mfem::DenseMatrix &flux_mat)
   {
      flux_mat = 0.0;
   }

   /// Compute jacobian of flux corresponding to an exact solution
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables (not used yet)
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         double jac,
                         const mfem::Vector &q,
                         const mfem::DenseMatrix &Dw,
                         mfem::DenseMatrix &flux_jac);

   /// Compute jacobian of flux corresponding to an exact solution
   /// w.r.t `entrpy-variables' derivatives`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDw(const mfem::Vector &x,
                      const mfem::Vector &dir,
                      double jac,
                      const mfem::Vector &q,
                      const mfem::DenseMatrix &Dw,
                      std::vector<mfem::DenseMatrix> &flux_jac);

   /// Compute the Jacobian of calcFluxDv w.r.t. state
   /// \param[in] x - coordinate location at which fluxes are evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] flux_jac[di] - Jacobian of calcFluxDv[di] with respect to `q`
   void calcFluxDvJacState(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &u,
                           std::vector<mfem::DenseMatrix> &flux_jac)
   {
      for (int d = 0; d < dim; ++d)
      {
         flux_jac[d] = 0.0;
      }
   }

private:
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// Function to evaluate the exact solution at a given x value
   void (*exactSolution)(const mfem::Vector &, mfem::Vector &);
   /// far-field boundary state
   mfem::Vector qexact;
   /// work space for flux computations
   mfem::Vector work_vec;
};

/// Integrator for surface-force at no-slip adiabatic-wall boundary
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
template <int dim>
class SurfaceForce : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs an integrator for surface force at a no-slip, adiabatic wall
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - number of state variables at each node
   /// \param[in] Re_num - Reynolds number
   /// \param[in] Pr_num - Prandtl number
   /// \param[in] q_ref - a reference state (needed by penalty)
   /// \param[in] force_dir - unit vector specifying the direction of the force
   /// \param[in] vis - viscosity (if negative use Sutherland's law)
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SurfaceForce(adept::Stack &diff_stack,
                const mfem::FiniteElementCollection *fe_coll,
                int num_state_vars,
                double Re_num,
                double Pr_num,
                const mfem::Vector &q_ref,
                const mfem::Vector &force_dir,
                double vis = -1.0,
                double a = 1.0)
    : num_states(num_state_vars),
      alpha(a),
      stack(diff_stack),
      fec(fe_coll),
      Re(Re_num),
      Pr(Pr_num),
      mu(vis),
      qfs(q_ref),
      force_nrm(force_dir),
      work_vec(dim + 2)
   { }

   /// Construct the contribution to functional from the boundary element
   /// \param[in] el_bnd - boundary element that contribute to the functional
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \return element local contribution to functional
   double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                        const mfem::FiniteElement &el_unused,
                        mfem::FaceElementTransformations &trans,
                        const mfem::Vector &elfun) override;

   /// Construct the contribution to the element local dJ/dq
   /// \param[in] el_bnd - the finite element whose residual we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   void AssembleFaceVector(const mfem::FiniteElement &el_bnd,
                           const mfem::FiniteElement &el_unused,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   /// Construct the element local Jacobian (not used)
   /// \param[in] el_bnd - the finite element whose residual we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   void AssembleFaceGrad(const mfem::FiniteElement &el_bnd,
                         const mfem::FiniteElement &el_unused,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override
   { }

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

   /// Computes boundary node contribution to the surface force
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian determinant (needed by some fluxes)
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - stress at given point
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       double jac,
                       const mfem::Vector &q,
                       const mfem::DenseMatrix &Dw);

   /// Returns the gradient of the stress with respect to `q`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[out] flux_vec - value of the flux
   void calcBndryFunJacState(const mfem::Vector &x,
                             const mfem::Vector &dir,
                             double jac,
                             const mfem::Vector &q,
                             const mfem::DenseMatrix &Dw,
                             mfem::Vector &flux_vec);

   /// Compute the gradient of the stress with respect to `Dw`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] jac - mapping Jacobian (needed by no-slip penalty)
   /// \param[in] q - conservative variables at which to evaluate the fluxes
   /// \param[in] Dw - space derivatives of the entropy variables
   /// \param[in] flux_mat[:,di] - to be scaled by derivative `D_[di] v`
   void calcBndryFunJacDw(const mfem::Vector &x,
                          const mfem::Vector &dir,
                          double jac,
                          const mfem::Vector &q,
                          const mfem::DenseMatrix &Dw,
                          mfem::DenseMatrix &flux_mat);

private:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// used to select the appropriate face element
   const mfem::FiniteElementCollection *fec;
   /// Reynolds number
   double Re;
   /// Prandtl number
   double Pr;
   /// nondimensionalized dynamic viscosity
   double mu;
   /// Fixed state used to compute no-slip penalty matrix
   mfem::Vector qfs;
   /// `dim` entry unit normal vector specifying the direction of the force
   mfem::Vector force_nrm;
   /// work space for flux computations
   mfem::Vector work_vec;
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
   /// stores various Jacobian terms
   mfem::DenseMatrix flux_mat;
   /// Jacobian of w variables with respect to states u at node j
   mfem::DenseMatrix dwduj;
#endif
};

#include "navier_stokes_integ_def.hpp"

}  // namespace mach

#endif