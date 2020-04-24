#ifndef MACH_MESH_SENS_INTEG
#define MACH_MESH_SENS_INTEG

#include "mfem.hpp"

#include "sbp_fe.hpp"
#include "solver.hpp"

namespace mach
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
                            int num_state_vars = 1, double a = 1.0)
       : state(state_vec), adjoint(adjoint_vec), num_states(num_state_vars),
         alpha(a) {}

   /// Construct the element local contribution to dF/dx
   /// \param[in] el - the finite element whose dF/dx contribution we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[out] elvect - element local dF/dx
   virtual void AssembleRHSElementVect(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &trans,
                                       mfem::Vector &elvect);

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
   void flux(int di, const mfem::Vector &u_left, const mfem::Vector &u_right,
             mfem::Vector &flux_vec)
   {
      static_cast<Derived*>(this)->calcFlux(di, u_left, u_right, flux_vec);
   }
};

#if 0
/// Integrator for mesh sensitivity associated with boundary integrators
/// \tparam Derived - a class Derived from this one (needed for CRTP)
template <typename Derived>
class BoundaryMeshSensIntegrator : public mfem::NonlinearFormIntegrator
{
public:
   /// Constructs an integrator for boundary-based mesh sensitivities
   /// \param[in] state_vec - the state at which to evaluate the senstivity
   /// \param[in] adjoint_vec - the adjoint that weights the residual
   /// \param[in] fe_coll - used to determine the face elements
   /// \param[in] num_state_vars - the number of state variables
   /// \param[in] a - used to move residual to lhs (1.0) or rhs(-1.0)
   BoundaryMeshSensIntegrator(const mfem::GridFunction &state_vec,
                              const mfem::GridFunction &adjoint_vec,
                              const mfem::FiniteElementCollection *fe_coll,
                              int num_state_vars = 1, double a = 1.0)
       : state(state_vec), adjoint(adjoint_vec), num_states(num_state_vars),
         alpha(a), stack(diff_stack), fec(fe_coll) {}

   /// Construct the contribution to the element local dF/dX
   /// \param[in] el_bnd - the finite element whose dF/dX we want to update
   /// \param[in] el_unused - dummy element that is not used for boundaries
   /// \param[in] trans - holds geometry and mapping information about the face
   /// \param[in] elfun - element local nodes function
   /// \param[out] elvect - element local dF/dX
   virtual void AssembleFaceVector(const mfem::FiniteElement &el_bnd,
                                   const mfem::FiniteElement &el_unused,
                                   mfem::FaceElementTransformations &trans,
                                   const mfem::Vector &elfun,
                                   mfem::Vector &elvect);

protected: 
   /// scales the terms; can be used to move to rhs/lhs
   double alpha;
   /// used to select the appropriate face element
   const mfem::FiniteElementCollection *fec;
#ifndef MFEM_THREAD_SAFE

#endif

};
#endif

#include "mesh_sens_integ_def.hpp"

} // namespace mach

#endif
