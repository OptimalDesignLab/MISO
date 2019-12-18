#ifndef MACH_EULER_INTEG
#define MACH_EULER_INTEG

#include "mfem.hpp"
#include "adept.h"
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
/// \note This derived class uses the CRTP
template <int dim>
class IsmailRoeIntegrator : public DyadicFluxIntegrator<IsmailRoeIntegrator<dim>>
{
public:
   /// Construct an integrator for the Ismail-Roe flux over domains
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - factor, usually used to move terms to rhs
   IsmailRoeIntegrator(adept::Stack &diff_stack, double a = 1.0)
       : DyadicFluxIntegrator<IsmailRoeIntegrator<dim>>(
             diff_stack, dim+2, a) {}

   /// Ismail-Roe two-point (dyadic) entropy conservative flux function
   /// \param[in] di - physical coordinate direction in which flux is wanted
   /// \param[in] qL - conservative variables at "left" state
   /// \param[in] qR - conservative variables at "right" state
   /// \param[out] flux - fluxes in the direction `di`
   /// \note This is simply a wrapper for the function in `euler_fluxes.hpp`
   void calcFlux(int di, const mfem::Vector &qL,
                 const mfem::Vector &qR, mfem::Vector &flux)
   {
      calcIsmailRoeFlux<double,dim>(di, qL.GetData(), qR.GetData(),
                                 flux.GetData());
   }

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
/// \note This derived class uses the CRTP
template <int dim>
class EntStableLPSIntegrator : public LPSIntegrator<EntStableLPSIntegrator<dim>>
{
public:
   /// Construct an entropy-stable LPS integrator
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   /// \param[in] coeff - the LPS coefficient
   EntStableLPSIntegrator(adept::Stack &diff_stack, double a = 1.0,
                          double coeff = 1.0)
       : LPSIntegrator<EntStableLPSIntegrator<dim>>(
             diff_stack, dim + 2, a, coeff) {}

   /// converts conservative variables to entropy variables
   /// \param[in] q - conservative variables that are to be converted
   /// \param[out] w - entropy variables corresponding to `q`
   /// \note a wrapper for the relevant function in `euler_fluxes.hpp`
   void convertVars(const mfem::Vector &q, mfem::Vector &w)
   {
      calcEntropyVars<double,dim>(q.GetData(), w.GetData());
   }

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
                     const mfem::Vector &vec, mfem::Vector &mat_vec)
   {
      applyLPSScaling<double,dim>(adjJ.GetData(), q.GetData(), vec.GetData(),
                                  mat_vec.GetData());
   }

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

/// Integrator for the steady isentropic-vortex boundary condition
/// \note This derived class uses the CRTP
class IsentropicVortexBC : public InviscidBoundaryIntegrator<IsentropicVortexBC>
{
public:
   /// Constructs an integrator for isentropic vortex boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   IsentropicVortexBC(adept::Stack &diff_stack,
                      const mfem::FiniteElementCollection *fe_coll,
                      double a = 1.0)
       : InviscidBoundaryIntegrator<IsentropicVortexBC>(
             diff_stack, fe_coll, 4, a) {}

   /// Note used (or, rather, *do not use*!)
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q) { return 0.0; }

   /// Compute a characteristic boundary flux for the isentropic vortex
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec)
   {
      calcIsentropicVortexFlux<double>(x.GetData(), dir.GetData(), q.GetData(),
                                       flux_vec.GetData());
   }

   /// Compute the Jacobian of the isentropic vortex boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x, const mfem::Vector &dir,
                         const mfem::Vector &q, mfem::DenseMatrix &flux_jac)
   {
      throw MachException("Not implemented!");
   }

   /// Compute the Jacobian of the isentropic vortex boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q, mfem::DenseMatrix &flux_jac)
   {
      throw MachException("Not implemented!");
   }
};

/// Integrator for inviscid slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class SlipWallBC : public InviscidBoundaryIntegrator<SlipWallBC<dim>>
{
public:
   /// Constructs an integrator for a slip-wall boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   SlipWallBC(adept::Stack &diff_stack,
              const mfem::FiniteElementCollection *fe_coll,
              double a = 1.0)
       : InviscidBoundaryIntegrator<SlipWallBC<dim>>(
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
                 const mfem::Vector &q, mfem::Vector &flux_vec)
   {
      calcSlipWallFlux<double,dim>(x.GetData(), dir.GetData(), q.GetData(),
                                   flux_vec.GetData());
   }

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
/// \note This derived class uses the CRTP
template <int dim>
class FarFieldBC : public InviscidBoundaryIntegrator<FarFieldBC<dim>>
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
       : InviscidBoundaryIntegrator<FarFieldBC<dim>>(
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
                 const mfem::Vector &q, mfem::Vector &flux_vec)
   {
      calcBoundaryFlux<double, dim>(dir.GetData(), qfs.GetData(), q.GetData(),
                                    work_vec.GetData(), flux_vec.GetData());
   }

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
template<int dim>
class InterfaceIntegrator : public InviscidFaceIntegrator<InterfaceIntegrator<dim>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - pointer to a finite element collection
   /// \param[in] a - factor, usually used to move terms to rhs
   InterfaceIntegrator(adept::Stack &diff_stack,
                       const mfem::FiniteElementCollection *fe_coll,
                       double a = 1.0)
      : InviscidFaceIntegrator<InterfaceIntegrator<dim>>(diff_stack, fe_coll,
         dim+2, a) { }
   
   /// Compute the interface function at a given (scaled) direction
   /// \param[in] dir - vector normal to the interface
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qR - "right" state at which to evaluate the flu 
   /// \param[out] flux - value of the flux
   /// \note wrapper for the relevant function in `euler_fluxes.hpp`
   void calcFlux(const mfem::Vector &dir, const mfem::Vector &qL,
                 const mfem::Vector &qR, mfem::Vector &flux)
   {
      calcIsmailRoeFaceFlux<double, dim>(dir.GetData(), qL.GetData(),
                                         qR.GetData(), flux.GetData());
   }

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
};

/// Integrator for forces due to pressure
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class PressureForce : public InviscidBoundaryIntegrator<PressureForce<dim>>
{
public:
   /// Constructs an integrator that computes pressure contribution to force
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] force_dir - unit vector specifying the direction of the force
   PressureForce(adept::Stack &diff_stack,
                 const mfem::FiniteElementCollection *fe_coll,
                 const mfem::Vector &force_dir)
       : InviscidBoundaryIntegrator<PressureForce<dim>>(
             diff_stack, fe_coll, dim+2, 1.0), force_nrm(force_dir),
             work_vec(dim+2) {}

   /// Return an adjoint-consistent slip-wall normal (pressure) stress term
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \returns conmponent of stress due to pressure in `force_nrm` direction
   double calcBndryFun(const mfem::Vector &x, const mfem::Vector &dir,
                       const mfem::Vector &q)
   {
      calcSlipWallFlux<double,dim>(x.GetData(), dir.GetData(), q.GetData(),
                                   work_vec.GetData());
      return dot<double,dim>(force_nrm.GetData(), work_vec.GetData()+1);
   }

   /// Not used
   void calcFlux(const mfem::Vector &x, const mfem::Vector &dir,
                 const mfem::Vector &q, mfem::Vector &flux_vec) {}

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

#include "euler_integ_def.hpp"

} // namespace mach

#endif