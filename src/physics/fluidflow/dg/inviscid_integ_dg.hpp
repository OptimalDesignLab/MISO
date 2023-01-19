#ifndef MACH_INVISCID_INTEG_DG
#define MACH_INVISCID_INTEG_DG

#include "adept.h"
#include "mfem.hpp"
using namespace mfem;
using namespace std;
namespace mach
{
/// Integrator for one-point inviscid flux functions
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class DGInviscidIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for "inviscid" type fluxes
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - factor, usually used to move terms to rhs
   /// \note `num_state_vars` is not necessarily the same as the number of
   /// states used by, nor the number of fluxes returned by, `flux`.
   /// For example, there may be 5 states for the 2D RANS equations, but
   /// `flux` may use only the first 4.
   DGInviscidIntegrator(adept::Stack &diff_stack,
                        int num_state_vars = 1,
                        double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack)
   { }

   /// Get the contribution of this element to a functional
   /// \param[in] el - the finite element whose contribution we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun) override;

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect) override;

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   void AssembleElementGrad(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun,
                            mfem::DenseMatrix &elmat) override;

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
#ifndef MFEM_THREAD_SAFE
   /// the coordinates of node i
   mfem::Vector x_i;
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
   /// used to store the flux Jacobian at node i
   mfem::DenseMatrix flux_jaci;
   /// used to store the flux at each node
   mfem::DenseMatrix elflux;
   /// used to store the residual in (num_states, Dof) format
   mfem::DenseMatrix elres;
#endif

   /// Compute a scalar domain functional
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] u - state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note `x` can be ignored depending on the function
   /// \note This uses the CRTP, so it wraps a call to `calcVolFun` in Derived.
   double volFun(const mfem::Vector &x, const mfem::Vector &u)
   {
      return static_cast<Derived *>(this)->calcVolFun(x, u);
   }

   /// An inviscid flux function
   /// \param[in] dir - desired direction for the flux
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_vec - flux evaluated at `u` in direction `dir`
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &dir,
             const mfem::Vector &u,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(dir, u, flux_vec);
   }

   /// Compute the Jacobian of the flux function `flux` w.r.t. `u`
   /// \parma[in] dir - desired direction for the flux
   /// \param[in] u - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `u`
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void fluxJacState(const mfem::Vector &dir,
                     const mfem::Vector &u,
                     mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived *>(this)->calcFluxJacState(dir, u, flux_jac);
   }

   /// Compute the Jacobian of the flux function `flux` w.r.t. `dir`
   /// \parma[in] dir - desired direction for the flux
   /// \param[in] u - state at which to evaluate the flux Jacobian
   /// \param[out] flux_jac - Jacobian of the flux function w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &dir,
                   const mfem::Vector &u,
                   mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived *>(this)->calcFluxJacDir(dir, u, flux_jac);
   }
};

/// Integrator for inviscid boundary fluxes (fluxes that do not need gradient)
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class DGInviscidBoundaryIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a boundary integrator based on a given boundary flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   DGInviscidBoundaryIntegrator(adept::Stack &diff_stack,
                                const mfem::FiniteElementCollection *fe_coll,
                                int num_state_vars = 1,
                                double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack), fec(fe_coll)
   { }

   /// Construct the contribution to a functional from the boundary element
   /// \param[in] el_bnd - boundary element that contribute to the functional
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \return element local contribution to functional
   double GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                        const mfem::FiniteElement &el_unused,
                        mfem::FaceElementTransformations &trans,
                        const mfem::Vector &elfun) override;

   /// Construct the contribution to the element local residual
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

   /// Construct the element local Jacobian
   /// \param[in] el_bnd - the finite element whose residual we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - hold geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   void AssembleFaceGrad(const mfem::FiniteElement &el_bnd,
                         const mfem::FiniteElement &el_unused,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// used to select the appropriate face element
   const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
   /// used to reference the state at face node
   mfem::Vector u_face;
   /// store the physical location of a node
   mfem::Vector x;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// stores the shape vector
   mfem::Vector shape;
   /// stores the flux evaluated by `bnd_flux`
   mfem::Vector flux_face;
   /// stores the jacobian of the flux with respect to the state at `u_face`
   mfem::DenseMatrix flux_jac_face;
#endif

   /// Compute a scalar boundary function
   /// \param[in] x - coordinate location at which function is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note `x` can be ignored depending on the function
   /// \note This uses the CRTP, so it wraps a call to `calcFunction` in
   /// Derived.
   double bndryFun(const mfem::Vector &x,
                   const mfem::Vector &dir,
                   const mfem::Vector &u)
   {
      return static_cast<Derived *>(this)->calcBndryFun(x, dir, u);
   }

   /// Compute a boundary flux function
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &x,
             const mfem::Vector &dir,
             const mfem::Vector &u,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(x, dir, u, flux_vec);
   }

   /// Compute the Jacobian of the boundary flux function w.r.t. `u`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_jac - Jacobian of `flux` w.r.t. `u`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void fluxJacState(const mfem::Vector &x,
                     const mfem::Vector &dir,
                     const mfem::Vector &u,
                     mfem::DenseMatrix &flux_jac)
   {
      static_cast<Derived *>(this)->calcFluxJacState(x, dir, u, flux_jac);
   }

   /// Compute the Jacobian of the boundary flux function w.r.t. `dir`
   /// \param[in] x - coordinate location at which flux is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[out] flux_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &x,
                   const mfem::Vector &nrm,
                   const mfem::Vector &u,
                   mfem::DenseMatrix &flux_dir)
   {
      static_cast<Derived *>(this)->calcFluxJacDir(x, nrm, u, flux_dir);
   }
};

/// Integrator for inviscid interface fluxes (fluxes that do not need gradient)
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class DGInviscidFaceIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs a face integrator based on a given interface flux
   /// \param[in] diff_stack - for algorithmic differentiation
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   DGInviscidFaceIntegrator(adept::Stack &diff_stack,
                            const mfem::FiniteElementCollection *fe_coll,
                            int num_state_vars = 1,
                            double a = 1.0)
    : num_states(num_state_vars), alpha(a), stack(diff_stack), fec(fe_coll)
   { }

   /// Get the contribution from the interface to a functional
   /// \param[in] el_left - "left" element for functional contribution
   /// \param[in] el_right - "right" element for functional contribution
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - holds the solution on the adjacent elements
   double GetFaceEnergy(const mfem::FiniteElement &el_left,
                        const mfem::FiniteElement &el_right,
                        mfem::FaceElementTransformations &trans,
                        const mfem::Vector &elfun) override;

   /// Construct the contribution to the element local residuals
   /// \param[in] el_left - "left" element whose residual we want to update
   /// \param[in] el_right - "right" element whose residual we want to update
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   void AssembleFaceVector(const mfem::FiniteElement &el_left,
                           const mfem::FiniteElement &el_right,
                           mfem::FaceElementTransformations &trans,
                           const mfem::Vector &elfun,
                           mfem::Vector &elvect) override;

   /// Construct the element local Jacobian
   /// \param[in] el_left - "left" element whose residual we want to update
   /// \param[in] el_right - "right" element whose residual we want to update
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local state function
   /// \param[out] elmat - element local Jacobian
   void AssembleFaceGrad(const mfem::FiniteElement &el_left,
                         const mfem::FiniteElement &el_right,
                         mfem::FaceElementTransformations &trans,
                         const mfem::Vector &elfun,
                         mfem::DenseMatrix &elmat) override;

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// stack used for algorithmic differentiation
   adept::Stack &stack;
   /// used to select the appropriate face element
   const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE
   /// used to reference the left state at face node
   mfem::Vector u_face_left;
   /// used to reference the right state at face node
   mfem::Vector u_face_right;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// stores the flux evaluated by `bnd_flux`
   mfem::Vector flux_face;
   /// stores shape vectors and flux valuse
   mfem::Vector shape1, shape2, funval1, funval2, fluxN;
   /// stores the jacobian of the flux with respect to the left state
   mfem::DenseMatrix flux_jac_left;
   /// stores the jacobian of the flux with respect to the right state
   mfem::DenseMatrix flux_jac_right;
#endif

   /// Compute a scalar interface function
   /// \param[in] dir - vector normal to the face
   /// \param[in] u_left - "left" state at which to evaluate the function
   /// \param[in] u_right - "right" state at which to evaluate the function
   /// \returns fun - value of the function
   /// \note This uses the CRTP, so it wraps a call to `calcIFaceFun` in
   /// Derived.
   double iFaceFun(const mfem::Vector &dir,
                   const mfem::Vector &u_left,
                   const mfem::Vector &u_right)
   {
      return static_cast<Derived *>(this)->calcIFaceFun(dir, u_left, u_right);
   }

   /// Compute an interface flux function
   /// \param[in] dir - vector normal to the face
   /// \param[in] u_left - "left" state at which to evaluate the flux
   /// \param[in] u_right - "right" state at which to evaluate the flux
   /// \param[out] flux_vec - value of the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(const mfem::Vector &dir,
             const mfem::Vector &u_left,
             const mfem::Vector &u_right,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(dir, u_left, u_right, flux_vec);
   }

   /// Compute the Jacobian of the interface flux function w.r.t. states
   /// \param[in] dir - vector normal to the face
   /// \param[in] u_left - "left" state at which to evaluate the flux
   /// \param[in] u_right - "right" state at which to evaluate the flux
   /// \param[out] jac_left - Jacobian of `flux` w.r.t. `u_left`
   /// \param[out] jac_right - Jacobian of `flux` w.r.t. `u_right`
   /// \note This uses the CRTP, so it wraps a call a func. in Derived.
   void fluxJacStates(const mfem::Vector &dir,
                      const mfem::Vector &u_left,
                      const mfem::Vector &u_right,
                      mfem::DenseMatrix &jac_left,
                      mfem::DenseMatrix &jac_right)
   {
      static_cast<Derived *>(this)->calcFluxJacState(
          dir, u_left, u_right, jac_left, jac_right);
   }

   /// Compute the Jacobian of the interface flux function w.r.t. `dir`
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u_left - "left" state at which to evaluate the flux
   /// \param[in] u_right - "right" state at which to evaluate the flux
   /// \param[out] flux_dir - Jacobian of `flux` w.r.t. `dir`
   /// \note This uses the CRTP, so it wraps a call to a func. in Derived.
   void fluxJacDir(const mfem::Vector &dir,
                   const mfem::Vector &u_left,
                   const mfem::Vector &u_right,
                   mfem::DenseMatrix &flux_dir)
   {
      static_cast<Derived *>(this)->calcFluxJacDir(
          dir, u_left, u_right, flux_dir);
   }
};
/// Integrator for diffusion (artificial viscosity)
/// \tparam Derived - a class Derived from this one (needed for CRTP)
/// \note This probably does not need to be a nonlinear integrator, but this
/// makes it easier to incorporate directly into the nonlinear form.
template <int dim>
class EulerDiffusionIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Construct an integrator for MMS sources
   /// \param[in] a - factor, usually used to move terms to rhs
   EulerDiffusionIntegrator(double &visc_coeff , double a = 1.0)
    : num_states(dim + 2), 
    diff_coeff(visc_coeff),
    alpha(a)
   { }
   // cutSquareIntRules(_cutSquareIntRules),
   // embeddedElements(_embeddedElements)

   /// Get the contribution of this element to a functional
   /// \param[in] el - the finite element whose contribution we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   double GetElementEnergy(const mfem::FiniteElement &el,
                           mfem::ElementTransformation &trans,
                           const mfem::Vector &elfun)
   {
      return 0.0;
   }

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state function
   /// \param[out] elvect - element local residual
   void AssembleElementVector(const mfem::FiniteElement &el,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &elfun,
                              mfem::Vector &elvect)
   {
      using namespace mfem;
      using namespace std;
      const int num_nodes = el.GetDof();
      elvect.SetSize(num_states * num_nodes);
      elvect = 0.0;
         DenseMatrix u_mat(elfun.GetData(), num_nodes, num_states);
         DenseMatrix res(elvect.GetData(), num_nodes, num_states);
         dshape.SetSize(num_nodes, dim);
         pelmat.SetSize(num_states, dim);
         pelmat2.SetSize(num_states, dim);
         gshape.SetSize(dim);
         Jinv.SetSize(dim);
         const IntegrationRule *ir = NULL;
         if (ir == NULL)
         {
            ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));
         }
         for (int i = 0; i < ir->GetNPoints(); i++)
         {
            const IntegrationPoint &ip = ir->IntPoint(i);

            trans.SetIntPoint(&ip);
            CalcAdjugate(trans.Jacobian(), Jinv);
            double w = ip.weight / trans.Weight();
            w *= diff_coeff;

            MultAAt(Jinv, gshape);
            gshape *= w;

            el.CalcDShape(ip, dshape);

            MultAtB(u_mat, dshape, pelmat);
            MultABt(pelmat, gshape, pelmat2);
            AddMultABt(dshape, pelmat2, res);
         }
         res *= alpha;
   }

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element
   /// mapping \param[in] elfun - element local state function \param[out]
   /// elmat - element local Jacobian
   void AssembleElementGrad(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun,
                            mfem::DenseMatrix &elmat)
   {
      using namespace mfem;
      using namespace std;
      const int num_nodes = el.GetDof();
      int ndof = elfun.Size();
      elmat.SetSize(ndof);
      elmat = 0.0;
      dshape.SetSize(num_nodes, dim);
      pelmat.SetSize(num_nodes, dim);
      gshape.SetSize(dim);
      Jinv.SetSize(dim);
      elmat1.SetSize(num_nodes);
      const IntegrationRule *ir = NULL;
      if (ir == NULL)
      {
         int intorder = trans.OrderGrad(&el) + trans.Order() + el.GetOrder();
         ir = &IntRules.Get(el.GetGeomType(), intorder);
      }
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);
         CalcAdjugate(trans.Jacobian(), Jinv);
         double w = ip.weight / trans.Weight();
         w *= diff_coeff;
         MultAAt(Jinv, gshape);
         gshape *= w;
         el.CalcDShape(ip, dshape);
         MultABt(dshape, gshape, pelmat);
         MultABt(dshape, pelmat, elmat1);
         for (int m = 0; m < dim + 2; ++m)
         {
            elmat.AddMatrix(elmat1, m * num_nodes, m * num_nodes);
         }
      }
   }

protected:
   /// number of states
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// viscosity coefficient
   double &diff_coeff;
private:
   DenseMatrix dshape, dshapedxt, pelmat, pelmat2;
   DenseMatrix Jinv, gshape, elmat1;
};
}  // namespace mach

#include "inviscid_integ_def_dg.hpp"

#endif
