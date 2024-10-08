#ifndef MISO_MESH_SENS_INTEG
#define MISO_MESH_SENS_INTEG

#include "mfem.hpp"

namespace miso
{
/// Integrator for mesh sensitivity of dyadic domain integrators
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class DyadicMeshSensIntegrator : public mfem::LinearFormIntegrator
{
public:
   /// Constructs an integrator for dyadic-integrator mesh sensitivities
   /// \param[in] state_vec - the state at which to evaluate the senstivity
   /// \param[in] adjoint_vec - the adjoint that weights the residual
   /// \param[in] num_state_vars - the number of state variables per node
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   DyadicMeshSensIntegrator(const mfem::GridFunction &state_vec,
                            const mfem::GridFunction &adjoint_vec,
                            int num_state_vars = 1,
                            double a = 1.0)
    : state(state_vec),
      adjoint(adjoint_vec),
      num_states(num_state_vars),
      alpha(a)
   { }

   /// Construct the element local contribution to dF/dx
   /// \param[in] el - the finite element whose dF/dx contribution we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[out] elvect - element local dF/dx
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

protected:
   /// The state vector used to evaluate fluxes
   const mfem::GridFunction &state;
   /// The adjoint vector that weights the residuals
   const mfem::GridFunction &adjoint;
   /// number of states (could extract from state or adjoint)
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
#ifndef MFEM_THREAD_SAFE
   /// stores the result of calling the flux function
   mfem::Vector fluxij;
   /// stores derivatives w.r.t. adjugate Jacobian at node i
   mfem::DenseMatrix adjJ_i_bar;
   /// stores derivatives w.r.t. adjugate Jacobian at node j
   mfem::DenseMatrix adjJ_j_bar;
   /// stores derivatives w.r.t. mesh nodes
   mfem::DenseMatrix PointMat_bar;
#endif

   /// A two point (i.e. dyadic) flux function
   /// \param[in] di - desired coordinate direction for flux
   /// \param[in] u_left - the "left" state
   /// \param[in] u_right - the "right" state
   /// \param[out] flux_vec - flux evaluated at `u_left` and `u_right`
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void flux(int di,
             const mfem::Vector &u_left,
             const mfem::Vector &u_right,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived *>(this)->calcFlux(di, u_left, u_right, flux_vec);
   }
};

/// Integrator for mesh sensitivity associated with boundary integrators
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class BoundaryMeshSensIntegrator : public mfem::LinearFormIntegrator
{
public:
   /// Constructs an integrator for boundary-based mesh sensitivities
   /// \param[in] state_vec - the state at which to evaluate the senstivity
   /// \param[in] adjoint_vec - the adjoint that weights the residual
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   BoundaryMeshSensIntegrator(const mfem::GridFunction &state_vec,
                              const mfem::GridFunction &adjoint_vec,
                              int num_state_vars = 1,
                              double a = 1.0)
    : state(state_vec),
      adjoint(adjoint_vec),
      num_states(num_state_vars),
      alpha(a)
   { }

   /// **Do not use**: only included because it is required by base class
   void AssembleRHSElementVect(const mfem::FiniteElement &el,
                               mfem::ElementTransformation &trans,
                               mfem::Vector &elvect) override;

   /// Construct the contribution to the element local dF/dX
   /// \param[in] el_bnd - the finite element whose dF/dX we want to update
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[out] elvect - element local dF/dX
   void AssembleRHSElementVect(const mfem::FiniteElement &el_bnd,
                               mfem::FaceElementTransformations &trans,
                               mfem::Vector &elvect) override;

protected:
   /// The state vector used to evaluate fluxes
   const mfem::GridFunction &state;
   /// The adjoint vector that weights the residuals
   const mfem::GridFunction &adjoint;
   /// number of states (could extract from state or adjoint)
   int num_states;
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
#ifndef MFEM_THREAD_SAFE
   /// store the physical location of a node
   mfem::Vector x;
   /// the outward pointing (scaled) normal to the boundary at a node
   mfem::Vector nrm;
   /// the derivative with respect to the normal derivative
   mfem::Vector nrm_bar;
   /// linear transformation from the element Jacobian to the face Jacobian
   mfem::DenseMatrix Jac_map;
   /// derivative with respect to the element mapping Jacobian
   mfem::DenseMatrix Jac_bar;
   // derivatives with respect to the face mapping Jacobian
   mfem::DenseMatrix Jac_face_bar;
   /// stores derivatives w.r.t. mesh nodes
   mfem::DenseMatrix PointMat_bar;
#endif

   /// Compute the derivative of flux_bar^T * flux w.r.t. the vector `dir`
   /// \param[in] x - coordinate location at which the derivative is evaluated
   /// \param[in] dir - vector normal to the boundary at `x`
   /// \param[in] u - state at which to evaluate the flux
   /// \param[in] flux_bar - flux weighting (e.g. the adjoint)
   /// \param[out] dir_bar - derivative with respect to `dir`
   /// \note `x` can be ignored depending on the flux
   /// \note This uses the CRTP, so it wraps a call to `calcFlux` in Derived.
   void fluxBar(const mfem::Vector &x,
                const mfem::Vector &dir,
                const mfem::Vector &u,
                const mfem::Vector &flux_bar,
                mfem::Vector &dir_bar)
   {
      static_cast<Derived *>(this)->calcFluxBar(x, dir, u, flux_bar, dir_bar);
   }
};

}  // namespace miso

#include "mesh_sens_integ_def.hpp"

#endif
