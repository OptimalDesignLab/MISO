#ifndef MACH_EULER_INTEG
#define MACH_EULER_INTEG

#include "adept.h"
#include "mfem.hpp"

#include "inviscid_integ.hpp"
#include "euler_fluxes.hpp"

using adept::adouble;

namespace mach
{

/// Integrator for the Euler flux over an element
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class EulerIntegrator : public InviscidIntegrator<EulerIntegrator<dim>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - factor, usually used to move terms to rhs
   EulerIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : InviscidIntegrator<EulerIntegrator<dim>>(diff_stack, dim + 2, a) {}

   /// Not used by this integrator
   double calcVolFun(const mfem::Vector &x, const mfem::Vector &u) {}

   /// Euler flux function in a given (scaled) direction
   /// \param[in] dir - direction in which the flux is desired
   /// \param[in] q - conservative variables
   /// \param[out] flux - fluxes in the direction `dir`
   /// \note wrapper for the relevant function in `euler_fluxes.hpp`
   void calcFlux(const mfem::Vector &dir, const mfem::Vector &q,
                 mfem::Vector &flux)
   {
      calcEulerFlux<double,dim>(dir.GetData(), q.GetData(), flux.GetData());
   }

   /// Compute the Jacobian of the Euler flux w.r.t. `q`
   /// \param[in] dir - desired direction (scaled) for the flux
   /// \param[in] q - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &dir, const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the flux function `flux` w.r.t. `dir`
   /// \parma[in] dir - desired direction for the flux 
   /// \param[in] q - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void calcFluxJacDir(const mfem::Vector &dir, const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);
};

/// Integrator for the two-point entropy conservative Ismail-Roe flux
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class IsmailRoeIntegrator : public DyadicFluxIntegrator<
                                IsmailRoeIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for the Ismail-Roe flux over domains
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - factor, usually used to move terms to rhs
   IsmailRoeIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : DyadicFluxIntegrator<IsmailRoeIntegrator<dim, entvar>>(
             diff_stack, dim + 2, a) {}

   /// Ismail-Roe two-point (dyadic) entropy conservative flux function
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
};

/// Integrator for entropy stable local-projection stabilization
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class EntStableLPSIntegrator : public LPSIntegrator<
                                   EntStableLPSIntegrator<dim, entvar>>
{
public:
   /// Construct an entropy-stable LPS integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] coeff - the LPS coefficient
   EntStableLPSIntegrator(adept::Stack &diff_stack, double a = 1.0,
                          double coeff = 1.0)
       : LPSIntegrator<EntStableLPSIntegrator<dim,entvar>>(
             diff_stack, dim + 2, a, coeff) {}

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

   /// Computes the Jacobian of the product `A(adjJ,u)*v` w.r.t. `adjJ`
   /// \param[in] adjJ - adjugate of the mapping Jacobian
   /// \param[in] q - state at which the symmetric matrix `A` is evaluated
   /// \param[in] vec - vector that is being multiplied
   /// \param[out] mat_vec_jac - Jacobian of product w.r.t. `adjJ`
   /// \note `mat_vec_jac` stores derivatives treating `adjJ` is a 1d array.
   /// \note The size of `mat_vec_jac` must be set before calling this function
   void applyScalingJacAdjJ(const mfem::DenseMatrix &adjJ,
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

/// Integrator for the time term in an entropy-stable discretization
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the state variables are the entropy variables
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class MassIntegrator : public NonlinearMassIntegrator<
                           MassIntegrator<dim, entvar>>
{
public:
   /// Construct the nonlinear mass matrix integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] u_old - the state at the previous time step
   /// \param[in] delta_t - used to define state where mass matrix is evaluated
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   MassIntegrator(adept::Stack &diff_stack, const mfem::GridFunction &u_old,
                  double delta_t, double a = 1.0)
       : NonlinearMassIntegrator<MassIntegrator<dim, entvar>>(u_old, delta_t,
                                                              dim + 2, a),
         stack(diff_stack), q_work(dim + 2) {}

   /// converts state variables to entropy variables, if necessary
   /// \param[in] w - entropy variables that will be converted to conservative variables
   /// \param[out] q - conservative variables corresponding to `w`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertToConserv(const mfem::Vector &w, mfem::Vector &q);

   /// applies symmetric matrix `dq/du` to input `k`
   /// \param[in] u - state at which the symmetric matrix `dq/du` is evaluated
   /// \param[in] k - vector that is being multiplied
   /// \param[out] Ak - product of the multiplication
   void calcMatVec(const mfem::Vector &u, const mfem::Vector &k,
                   mfem::Vector &Ak);

   /// Compute the Jacobian of function `matVec` w.r.t. `u`
   /// \param[in] u - state at which to evaluate the Jacobian
   /// \param[in] k - vector that is being multiplied by `A = dq/du`
   /// \param[out] jac - Jacobian of the product w.r.t. `u`
   void calcMatVecJacState(const mfem::Vector &u, const mfem::Vector &k,
                           mfem::DenseMatrix &jac);
   
   /// Compute the Jacobian \partial u / \partial w
   /// \param[in] q - the input state variables 
   /// \param[out] jac - the Jacobian of conservative variable w.r.t state variables
   void calcToConservJacState(const mfem::Vector &q, mfem::DenseMatrix &jac);
   
   /// Computes the matrix (dq/du)
   /// \param[in] u - state at which to evaluate the entropy inverse Hessian
   /// \param[out] jac - stores the entropy inverse Hessian
   void calcMatVecJacK(const mfem::Vector &u, mfem::DenseMatrix &jac);

protected:
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// work vector used to store conservative variables
   mfem::Vector q_work;
};

/// Integrator for the steady isentropic-vortex boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class IsentropicVortexBC : public InviscidBoundaryIntegrator<
                               IsentropicVortexBC<dim, entvar>>
{
public:
   /// Constructs an integrator for isentropic vortex boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   IsentropicVortexBC(adept::Stack &diff_stack,
                      const mfem::FiniteElementCollection *fe_coll,
                      double a = 1.0)
       : InviscidBoundaryIntegrator<IsentropicVortexBC<dim, entvar>>(
             diff_stack, fe_coll, 4, a) {}

   /// Not used (or, rather, *do not use*!)
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q) { return 0.0; }

   /// Compute a characteristic boundary flux for the isentropic vortex
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec);

   /// Compute the Jacobian of the isentropic vortex boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the isentropic vortex boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac);
};

/// Integrator for inviscid slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class SlipWallBC : public InviscidBoundaryIntegrator<SlipWallBC<dim, entvar>>
{
public:
   /// Constructs an integrator for a slip-wall boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SlipWallBC(adept::Stack &diff_stack,
              const mfem::FiniteElementCollection *fe_coll,
              double a = 1.0)
       : InviscidBoundaryIntegrator<SlipWallBC<dim, entvar>>(
             diff_stack, fe_coll, dim+2, a) {}

   /// Not used (or, rather, *do not use*!)
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q) { return 0.0; }

   /// Compute an adjoint-consistent slip-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac);
};

/// Integrator for inviscid far-field boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class FarFieldBC : public InviscidBoundaryIntegrator<FarFieldBC<dim, entvar>>
{
public:
   /// Constructs an integrator for a far-field boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] q_far - state at the far-field
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   FarFieldBC(adept::Stack &diff_stack,
              const mfem::FiniteElementCollection *fe_coll,
              const mfem::Vector q_far, 
              double a = 1.0)
       : InviscidBoundaryIntegrator<FarFieldBC<dim, entvar>>(
             diff_stack, fe_coll, dim+2, a), qfs(q_far), work_vec(dim+2) {}

   /// Not used (or, rather, *do not use*!)
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q) { return 0.0; }

   /// Compute an adjoint-consistent slip-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac);

private:
   /// Stores the far-field state
   mfem::Vector qfs;
   /// Work vector for boundary flux computation
   mfem::Vector work_vec;
};

/// Interface integrator for the DG method
/// \tparam dim - number of spatial dimension (1, 2 or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
template <int dim, bool entvar = false>
class InterfaceIntegrator : public InviscidFaceIntegrator<
                                InterfaceIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] coeff - scales the dissipation (must be non-negative!)
   /// \param[in] fe_coll - pointer to a finite element collection
   /// \param[in] a - factor, usually used to move terms to rhs
   InterfaceIntegrator(adept::Stack &diff_stack, double coeff, 
                       const mfem::FiniteElementCollection *fe_coll,
                       double a = 1.0);

   /// Compute the interface function at a given (scaled) direction
   /// \param[in] dir - vector normal to the interface
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qR - "right" state at which to evaluate the flu 
   /// \param[out] flux - value of the flux
   /// \note wrapper for the relevant function in `euler_fluxes.hpp`
   void calcFlux(const mfem::Vector &dir, const mfem::Vector &qL,
                 const mfem::Vector &qR, mfem::Vector &flux);

   /// Compute the Jacobian of the interface flux function w.r.t. states
   /// \param[in] dir - vector normal to the face
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qL - "right" state at which to evaluate the flux
   /// \param[out] jacL - Jacobian of `flux` w.r.t. `qL`
   /// \param[out] jacR - Jacobian of `flux` w.r.t. `qR`
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void calcFluxJacState(const mfem::Vector &dir, const mfem::Vector &qL,
                         const mfem::Vector &qR, mfem::DenseMatrix &jacL,
                         mfem::DenseMatrix &jacR);

   /// Compute the Jacobian of the interface flux function w.r.t. `dir`
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qR - "right" state at which to evaluate the flux
   /// \param[out] jac_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void calcFluxJacDir(const mfem::Vector &dir, const mfem::Vector &qL,
                       const mfem::Vector &qR, mfem::DenseMatrix &jac_dir);

protected:
   /// Scalar that controls the amount of dissipation
   double diss_coeff;
};

/// Integrator for forces due to pressure
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class PressureForce : public InviscidBoundaryIntegrator<
                          PressureForce<dim, entvar>>
{
public:
   /// Constructs an integrator that computes pressure contribution to force
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] force_dir - unit vector specifying the direction of the force
   PressureForce(adept::Stack &diff_stack,
                 const mfem::FiniteElementCollection *fe_coll,
                 const mfem::Vector &force_dir)
       : InviscidBoundaryIntegrator<PressureForce<dim, entvar>>(
             diff_stack, fe_coll, dim+2, 1.0), force_nrm(force_dir),
             work_vec(dim+2) {}

   /// Return an adjoint-consistent slip-wall normal (pressure) stress term
   /// \param[in] x - coordinate location at which stress is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the stress
   /// \returns conmponent of stress due to pressure in `force_nrm` direction
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Returns the gradient of the stress with respect to `q`
   /// \param[in] x - coordinate location at which stress is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the stress
   /// \param[out] flux_vec - derivative of stress with respect to `q`
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec);

   /// Not used
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac) {}

   /// Not used
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac) {}

private:
   /// `dim` entry unit normal vector specifying the direction of the force
   mfem::Vector force_nrm;
   /// work vector used to stored the flux
   mfem::Vector work_vec;
};

/// Integrator for total (mathematical) entropy over an element
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class EntropyIntegrator : public InviscidIntegrator<
                              EntropyIntegrator<dim, entvar>>
{
public:
   /// Constructs an integrator that computes integral of entropy for an element
   /// \param[in] diff_stack - for algorithmic differentiation
   EntropyIntegrator(adept::Stack &diff_stack)
       : InviscidIntegrator<EntropyIntegrator<dim, entvar>>(
             diff_stack, dim + 2, 1.0) {}

   /// Return the entropy for the state `u` 
   /// \param[in] x - coordinate location at which stress is evaluated (not used)
   /// \param[in] u - state variables at which to evaluate the entropy
   /// \returns mathematical entropy based on `u`
   double calcVolFun(const mfem::Vector &x, const mfem::Vector &u);

   /// Not used
   void calcFlux(const mfem::Vector &dir, const mfem::Vector &u,
                 mfem::Vector &flux) {}

   /// Not used
   void calcFluxJacState(const mfem::Vector &dir, const mfem::Vector &u, 
                         mfem::DenseMatrix &flux_jac) {}

   /// Not used
   void calcFluxJacDir(const mfem::Vector &dir, const mfem::Vector &u,
                       mfem::DenseMatrix &flux_jac) {}
};

#include "euler_integ_def.hpp"

} // namespace mach

#endif