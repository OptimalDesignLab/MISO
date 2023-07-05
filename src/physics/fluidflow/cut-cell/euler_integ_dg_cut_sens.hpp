#ifndef MACH_EULER_INTEG_DG_CUT_SENS
#define MACH_EULER_INTEG_DG_CUT_SENS

#include "adept.h"
#include "mfem.hpp"
#include "inviscid_integ_dg_cut_sens.hpp"
#include "euler_fluxes.hpp"
#include "mms_integ_dg_cut.hpp"
namespace mach
{
/// Integrator for the Euler flux over an element
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \note This derived class uses the CRTP
template <int dim>
class CutEulerDGSensitivityIntegrator
 : public CutDGSensitivityInviscidIntegrator<
       CutEulerDGSensitivityIntegrator<dim>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] cutSquareIntRules - integration rule for cut cells
   /// \param[in] embeddedElements - elements completely inside the geometry
   /// \param[in] a - factor, usually used to move terms to rhs
   CutEulerDGSensitivityIntegrator(
       adept::Stack &diff_stack,
       std::map<int, IntegrationRule *> cutSquareIntRules,
       std::map<int, IntegrationRule *> cutSquareIntRules_sens,
       std::vector<bool> embeddedElements,
       double a = 1.0)
    : CutDGSensitivityInviscidIntegrator<CutEulerDGSensitivityIntegrator<dim>>(
          diff_stack,
          cutSquareIntRules,
          cutSquareIntRules_sens,
          embeddedElements,
          dim + 2,
          a)
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
/// Integrator for inviscid slip-wall boundary condition
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class CutDGSensitivitySlipWallBC
 : public CutDGSensitivityInviscidBoundaryIntegrator<CutDGSensitivitySlipWallBC<dim, entvar>>
{
public:
   /// Constructs an integrator for a slip-wall boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] cutSegmentIntRules - integration rule for cut segments
   /// \param[in] phi - level-set function
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   CutDGSensitivitySlipWallBC(adept::Stack &diff_stack,
                   const mfem::FiniteElementCollection *fe_coll,
                   std::map<int, IntegrationRule *> cutSegmentIntRules,
                    std::map<int, IntegrationRule *> cutSegmentIntRules_sens,
                   /*Algoim::LevelSet<2> */  LevelSetF<double, 2> phi,
                   double a = 1.0)
    : CutDGSensitivityInviscidBoundaryIntegrator<CutDGSensitivitySlipWallBC<dim, entvar>>(
          diff_stack,
          fe_coll,
          cutSegmentIntRules,
          cutSegmentIntRules_sens,
          phi,
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
  /// Not used
   void calcFluxJacNor(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::Vector &flux_jac);

   void calcFluxJacIntRule(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &q,
                           mfem::Vector &flux_jac);


private:
};
/// Interface integrator for the DG method
/// \tparam dim - number of spatial dimension (1, 2 or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
template <int dim, bool entvar = false>
class CutDGSensitivityInterfaceIntegrator
 : public CutDGSensitivityInviscidFaceIntegrator<CutDGSensitivityInterfaceIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for the Euler flux over elements
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] coeff - scales the dissipation (must be non-negative!)
   /// \param[in] fe_coll - pointer to a finite element collection
   /// \param[in] immersedFaces - interior faces completely inside the geometry
   /// \param[in] cutInteriorFaceIntRules - integration rule for the interior
   /// faces cut by the geometry \param[in] a - factor, usually used to move
   /// terms to rhs
   CutDGSensitivityInterfaceIntegrator(
       adept::Stack &diff_stack,
       double coeff,
       const mfem::FiniteElementCollection *fe_coll,
       std::map<int, bool> immersedFaces,
       std::map<int, IntegrationRule *> cutInteriorFaceIntRules,
       std::map<int, IntegrationRule *> cutInteriorFaceIntRules_sens,
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
/// Source-term integrator for a 2D Euler MMS problem
/// \note For details on the MMS problem, see the file viscous_mms.py
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class CutSensitivityPotentialMMSIntegrator
 : public CutSensitivityMMSIntegrator<CutSensitivityPotentialMMSIntegrator<dim, entvar>>
{
public:
   /// Construct an integrator for a 2D Navier-Stokes MMS source
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] cutSquareIntRules - integration rule for cut cells
   /// \param[in] embeddedElements - elements completely inside the geometry
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   CutSensitivityPotentialMMSIntegrator(adept::Stack &diff_stack,
                             std::map<int, IntegrationRule *> cutSquareIntRules,
                             std::map<int, IntegrationRule *> cutSquareIntRules_sens,
                             std::vector<bool> embeddedElements,
                             double a = 1.0)
    : CutSensitivityMMSIntegrator<CutSensitivityPotentialMMSIntegrator<dim, entvar>>(
          diff_stack,
          cutSquareIntRules,
          cutSquareIntRules_sens,
          embeddedElements,
          dim + 2,
          a)
   { }

   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcSource(const mfem::Vector &x, mfem::Vector &src) const
   {
      calcPotentialMMS<double>(x.GetData(), src.GetData());
   }
   /// Computes the MMS source term at a give point
   /// \param[in] x - spatial location at which to evaluate the source
   /// \param[out] src - source term evaluated at `x`
   void calcPotentialSourceJac(const mfem::Vector &xq,
                      mfem::DenseMatrix &source_jac);

private:
};

/// Integrator for forces due to pressure
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, states = ent. vars; otherwise, states = conserv.
/// \note This derived class uses the CRTP
template <int dim, bool entvar = false>
class CutDGSensitivityPressureForce
 : public CutDGSensitivityInviscidBoundaryIntegrator<CutDGSensitivityPressureForce<dim, entvar>>
{
public:
   /// Constructs an integrator that computes pressure contribution to force
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] force_dir - unit vector specifying the direction of the force
   /// \param[in] cutSegmentIntRules - integration rule for cut segments
   /// \param[in] phi - level-set function (required for normal vector
   /// calculations)
   CutDGSensitivityPressureForce(adept::Stack &diff_stack,
                      const mfem::FiniteElementCollection *fe_coll,
                      const mfem::Vector &force_dir,
                      std::map<int, IntegrationRule *> cutSegmentIntRules,
                      std::map<int, IntegrationRule *> cutSegmentIntRules_sens,
                      /*Algoim::LevelSet<2> */  LevelSetF<double, 2> phi)
    : CutDGSensitivityInviscidBoundaryIntegrator<CutDGSensitivityPressureForce<dim, entvar>>(
          diff_stack,
          fe_coll,
          cutSegmentIntRules,
          cutSegmentIntRules_sens,
          phi,
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
   void calcFluxJacNor(const mfem::Vector &x,
                       const mfem::Vector &dir,
                       const mfem::Vector &q,
                       mfem::Vector &flux_jac);

   void calcFluxJacIntRule(const mfem::Vector &x,
                           const mfem::Vector &dir,
                           const mfem::Vector &q,
                           mfem::Vector &flux_jac);

private:
   /// `dim` entry unit normal vector specifying the direction of the force
   mfem::Vector force_nrm;
   /// work vector used to stored the flux
  mfem::Vector work_vec;
};
}  // namespace mach

#include "euler_integ_def_dg_cut_sens.hpp"

#endif