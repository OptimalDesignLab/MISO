#ifndef MACH_EULER_INTEG_DG
#define MACH_EULER_INTEG_DG

#include "adept.h"
#include "mfem.hpp"

#include "inviscid_integ_dg.hpp"
#include "euler_fluxes.hpp"
#include "mms_integ_dg.hpp"

namespace mach
{
/// Integrator for the Euler flux over an element
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class EulerDGIntegrator : public DGInviscidIntegrator<EulerDGIntegrator<dim>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - factor, usually used to move terms to rhs
   EulerDGIntegrator(adept::Stack &diff_stack, double a = 1.0)
    : DGInviscidIntegrator<EulerDGIntegrator<dim>>(diff_stack, dim + 2, a)
   { }

   /// Not used by this integrator
   double calcVolFun(const mfem::Vector &x, const mfem::Vector &u)
   {
      return 0.0;
   }

   /// Euler flux function in a given (scaled) direction
   /// \param[in] dir - direction in which the flux is desired
   /// \param[in] q - conservative variables
   /// \param[out] flux - fluxes in the direction `dir`
   /// \note wrapper for the relevant function in `euler_fluxes.hpp`
   void calcFlux(const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux)
   {
      calcEulerFlux<double, dim>(dir.GetData(), q.GetData(), flux.GetData());
   }

   /// Compute the Jacobian of the Euler flux w.r.t. `q`
   /// \param[in] dir - desired direction (scaled) for the flux
   /// \param[in] q - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the flux function `flux` w.r.t. `dir`
   /// \parma[in] dir - desired direction for the flux
   /// \param[in] q - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void calcFluxJacDir(const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);
};

/// Integrator for the steady isentropic-vortex boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class DGIsentropicVortexBC
 : public DGInviscidBoundaryIntegrator<DGIsentropicVortexBC<dim, entvar>>
{
public:
   /// Constructs an integrator for isentropic vortex boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   DGIsentropicVortexBC(adept::Stack &diff_stack,
                        const mfem::FiniteElementCollection *fe_coll,
                        double a = 1.0)
    : DGInviscidBoundaryIntegrator<DGIsentropicVortexBC<dim, entvar>>(
          diff_stack,
          fe_coll,
          dim + 2,
          a)
   { }

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - state variable at which to evaluate the flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Compute a characteristic boundary flux for the isentropic vortex
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - state variable at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux_vec);

   /// Compute the Jacobian of the isentropic vortex boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - state variable at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the isentropic vortex boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - state variable at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);
};

/// Integrator for inviscid slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class DGSlipWallBC
 : public DGInviscidBoundaryIntegrator<DGSlipWallBC<dim, entvar>>
{
public:
   /// Constructs an integrator for a slip-wall boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   DGSlipWallBC(adept::Stack &diff_stack,
                const mfem::FiniteElementCollection *fe_coll,
                double a = 1.0)
    : DGInviscidBoundaryIntegrator<DGSlipWallBC<dim, entvar>>(diff_stack,
                                                              fe_coll,
                                                              dim + 2,
                                                              a)
   { }

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - state variable at which to evaluate the flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Compute an adjoint-consistent slip-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux_vec);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);
};

/// Integrator for inviscid far-field boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class DGFarFieldBC
 : public DGInviscidBoundaryIntegrator<DGFarFieldBC<dim, entvar>>
{
public:
   /// Constructs an integrator for a far-field boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] q_far - state at the far-field
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   DGFarFieldBC(adept::Stack &diff_stack,
                const mfem::FiniteElementCollection *fe_coll,
                const mfem::Vector &q_far,
                double a = 1.0)
    : DGInviscidBoundaryIntegrator<DGFarFieldBC<dim, entvar>>(diff_stack,
                                                              fe_coll,
                                                              dim + 2,
                                                              a),
      qfs(q_far),
      work_vec(dim + 2)
   { }

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - state variable at which to evaluate the flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Compute an adjoint-consistent slip-wall boundary flux
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux_vec);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `q`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `q`
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute the Jacobian of the slip-wall boundary flux w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `dir`
   void calcFluxJacDir(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);

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
class DGInterfaceIntegrator
 : public DGInviscidFaceIntegrator<DGInterfaceIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] coeff - scales the dissipation (must be non-negative!)
   /// \param[in] fe_coll - pointer to a finite element collection
   /// \param[in] a - factor, usually used to move terms to rhs
   DGInterfaceIntegrator(adept::Stack &diff_stack,
                         double coeff,
                         const mfem::FiniteElementCollection *fe_coll,
                         double a = 1.0);

   /// Contracts flux with the entropy variables
   /// \param[in] dir - vector normal to the interface
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qR - "right" state at which to evaluate the flux
   double calcIFaceFun(const mfem::Vector &dir,
                       const mfem::Vector &qL,
                       const mfem::Vector &qR);

   /// Compute the interface function at a given (scaled) direction
   /// \param[in] dir - vector normal to the interface
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qR - "right" state at which to evaluate the flux
   /// \param[out] flux - value of the flux
   /// \note wrapper for the relevant function in `euler_fluxes.hpp`
   void calcFlux(const mfem::Vector &dir,
                 const mfem::Vector &qL,
                 const mfem::Vector &qR,
                 mfem::Vector &flux);

   /// Compute the Jacobian of the interface flux function w.r.t. states
   /// \param[in] dir - vector normal to the face
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qL - "right" state at which to evaluate the flux
   /// \param[out] jacL - Jacobian of `flux` w.r.t. `qL`
   /// \param[out] jacR - Jacobian of `flux` w.r.t. `qR`
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void calcFluxJacState(const mfem::Vector &dir,
                         const mfem::Vector &qL,
                         const mfem::Vector &qR,
                         mfem::DenseMatrix &jacL,
                         mfem::DenseMatrix &jacR);

   /// Compute the Jacobian of the interface flux function w.r.t. `dir`
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] qL - "left" state at which to evaluate the flux
   /// \param[in] qR - "right" state at which to evaluate the flux
   /// \param[out] jac_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void calcFluxJacDir(const mfem::Vector &dir,
                       const mfem::Vector &qL,
                       const mfem::Vector &qR,
                       mfem::DenseMatrix &jac_dir);

protected:
   /// Scalar that controls the amount of dissipation
   double diss_coeff;
};

/// Integrator for mass matrix
class DGMassIntegrator : public mfem::BilinearFormIntegrator
{
public:
   /// Constructs a diagonal-mass matrix integrator.
   /// \param[in] nvar - number of state variables
   DGMassIntegrator(int nvar = 1) : num_state(nvar) { }

   /// Finds the mass matrix for the given element.
   /// \param[in] el - the element for which the mass matrix is desired
   /// \param[in,out] trans -  transformation
   /// \param[out] elmat - the element mass matrix
   void AssembleElementMatrix(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              mfem::DenseMatrix &elmat)
   {
      using namespace mfem;
      int num_nodes = el.GetDof();
      double w;
#ifdef MFEM_THREAD_SAFE
      Vector shape;
#endif
      elmat.SetSize(num_nodes * num_state);
      shape.SetSize(num_nodes);
      DenseMatrix elmat1;
      elmat1.SetSize(num_nodes);
      const IntegrationRule *ir = IntRule;
      if (ir == NULL)
      {
         int order = 2 * el.GetOrder() + trans.OrderW();
         if (el.Space() == FunctionSpace::rQk)
         {
            ir = &RefinedIntRules.Get(el.GetGeomType(), order);
         }
         else
         {
            ir = &IntRules.Get(el.GetGeomType(), order);
         }
      }
      elmat = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         el.CalcShape(ip, shape);
         w = trans.Weight() * ip.weight;
         AddMult_a_VVt(w, shape, elmat1);
         for (int k = 0; k < num_state; k++)
         {
            elmat.AddMatrix(elmat1, num_nodes * k, num_nodes * k);
         }
      }
   }

protected:
   mfem::Vector shape;
   mfem::DenseMatrix elmat;
   int num_state;
};
/// Integrator for forces due to pressure
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class DGPressureForce
 : public DGInviscidBoundaryIntegrator<DGPressureForce<dim, entvar>>
{
public:
   /// Constructs an integrator that computes pressure contribution to force
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] force_dir - unit vector specifying the direction of the force
   DGPressureForce(adept::Stack &diff_stack,
                   const mfem::FiniteElementCollection *fe_coll,
                   const mfem::Vector &force_dir)
    : DGInviscidBoundaryIntegrator<DGPressureForce<dim, entvar>>(diff_stack,
                                                                 fe_coll,
                                                                 dim + 2,
                                                                 1.0),
      force_nrm(force_dir),
      work_vec(dim + 2)
   { }

   /// Return an adjoint-consistent slip-wall normal (pressure) stress term
   /// \param[in] x - coordinate location at which stress is evaluated (not
   /// used) \param[in] dir - vector normal to the boundary at `x` \param[in] q
   /// - conservative variables at which to evaluate the stress \returns
   /// conmponent of stress due to pressure in `force_nrm` direction
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Returns the gradient of the stress with respect to `q`
   /// \param[in] x - coordinate location at which stress is evaluated (not
   /// used) \param[in] dir - vector normal to the boundary at `x` \param[in] q
   /// - conservative variables at which to evaluate the stress \param[out]
   /// flux_vec - derivative of stress with respect to `q`
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux_vec);

   /// Not used
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac)
   { }

   /// Not used
   void calcFluxJacDir(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac)
   { }

private:
   /// `dim` entry unit normal vector specifying the direction of the force
   mfem::Vector force_nrm;
   /// work vector used to stored the flux
   mfem::Vector work_vec;
};

/// Source-term integrator for a 2D Euler MMS problem
/// \note For details on the MMS problem, see the file viscous_mms.py
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class EulerMMSIntegrator
 : public InviscidMMSIntegrator<EulerMMSIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for a 2D Navier-Stokes MMS source
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   EulerMMSIntegrator(adept::Stack &diff_stack, double a = 1.0)
    : InviscidMMSIntegrator<EulerMMSIntegrator<dim, entvar>>(dim + 2, a)
   { }

   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcSource(const mfem::Vector &x, mfem::Vector &src) const
   {
      calcInviscidMMS<double>(x.GetData(), src.GetData());
   }

private:
};
/// Integrator for exact, prescribed BCs (with zero normal derivative)
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim, bool entvar>
class InviscidDGExactBC
 : public DGInviscidBoundaryIntegrator<InviscidDGExactBC<dim, entvar>>
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
   InviscidDGExactBC(adept::Stack &diff_stack,
                     const mfem::FiniteElementCollection *fe_coll,
                     void (*fun)(const mfem::Vector &, mfem::Vector &),
                     double a = 1.0)
    : DGInviscidBoundaryIntegrator<InviscidDGExactBC<dim, entvar>>(diff_stack,
                                                                   fe_coll,
                                                                   dim + 2,
                                                                   a),
      qexact(dim + 2),
      work_vec(dim + 2)
   {
      exactSolution = fun;
   }

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Compute flux corresponding to an exact solution
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux_vec);

   /// Compute jacobian of flux corresponding to an exact solution
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute jacobian of flux corresponding to an exact solution
   /// w.r.t `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDir(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);

private:
   /// Function to evaluate the exact solution at a given x value
   void (*exactSolution)(const mfem::Vector &, mfem::Vector &);
   /// far-field boundary state
   mfem::Vector qexact;
   /// work space for flux computations
   mfem::Vector work_vec;
};
// Integrator for exact, prescribed BCs (with zero normal derivative)
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim, bool entvar>
class DGPotentialFlowBC
 : public DGInviscidBoundaryIntegrator<DGPotentialFlowBC<dim, entvar>>
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
   DGPotentialFlowBC(adept::Stack &diff_stack,
                     const mfem::FiniteElementCollection *fe_coll,
                     double a = 1.0)
    : DGInviscidBoundaryIntegrator<DGPotentialFlowBC<dim, entvar>>(diff_stack,
                                                                   fe_coll,
                                                                   dim + 2,
                                                                   a)
   { }

   /// Contracts flux with the entropy variables
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the function
   /// \param[in] Dw - `Dw[:,di]` is the derivative of `w` in direction `di`
   /// \returns fun - w^T*flux
   double calcBndryFun(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q);

   /// Compute flux corresponding to an exact solution
   /// \param[in] x - coordinate location at which flux is evaluated (not used)
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   void calcFlux(const mfem::Vector &x,
                 const mfem::Vector &dir,
                 const mfem::Vector &q,
                 mfem::Vector &flux_vec);

   /// Compute jacobian of flux corresponding to an exact solution
   /// w.r.t `states`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacState(const mfem::Vector &x,
                         const mfem::Vector &dir,
                         const mfem::Vector &q,
                         mfem::DenseMatrix &flux_jac);

   /// Compute jacobian of flux corresponding to an exact solution
   /// w.r.t `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] q - conservative variables at which to evaluate the flux
   /// \param[out] flux_jac - jacobian of the flux
   void calcFluxJacDir(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::DenseMatrix &flux_jac);

private:
};

/// Source-term integrator for a 2D Euler (Potentail) MMS problem
/// \note For details on the MMS problem, see the file
/// viscous_mms_potential_flow.py \tparam dim - number of spatial dimensions (1,
/// 2, or 3) \tparam entvar - if true, states = ent. vars; otherwise, states =
/// conserv. \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class PotentialMMSIntegrator
 : public InviscidMMSIntegrator<PotentialMMSIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for a 2D Navier-Stokes MMS source
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   PotentialMMSIntegrator(adept::Stack &diff_stack, double a = 1.0)
    : InviscidMMSIntegrator<PotentialMMSIntegrator<dim, entvar>>(dim + 2, a)
   { }

   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcSource(const mfem::Vector &x, mfem::Vector &src) const
   {
      calcPotentialMMS<double>(x.GetData(), src.GetData());
   }

private:
};
}  // namespace mach

#include "euler_integ_def_dg.hpp"

#endif
