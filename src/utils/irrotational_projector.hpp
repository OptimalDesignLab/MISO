#ifndef MACH_IRROTATIONAL_PROJECTOR
#define MACH_IRROTATIONAL_PROJECTOR

#include "mfem.hpp"

#include "mach_input.hpp"

namespace mach
{

/// Forward declarations of mesh sens integrators
class DiffusionIntegratorMeshSens;
class VectorFEWeakDivergenceIntegratorMeshSens;

/// The following are adapted from MFEM's pfem_extras.xpp
class DiscreteGradOperator final : public mfem::ParDiscreteLinearOperator
{
public:
   DiscreteGradOperator(mfem::ParFiniteElementSpace *dfes,
                        mfem::ParFiniteElementSpace *rfes)
      : ParDiscreteLinearOperator(dfes, rfes)
   { this->AddDomainInterpolator(new mfem::GradientInterpolator); }
};

class DiscreteCurlOperator final : public mfem::ParDiscreteLinearOperator
{
public:
   DiscreteCurlOperator(mfem::ParFiniteElementSpace *dfes,
                        mfem::ParFiniteElementSpace *rfes)
      : ParDiscreteLinearOperator(dfes, rfes)
   { this->AddDomainInterpolator(new mfem::CurlInterpolator); }
};

class DiscreteDivOperator final : public mfem::ParDiscreteLinearOperator
{
public:
   DiscreteDivOperator(mfem::ParFiniteElementSpace *dfes,
                       mfem::ParFiniteElementSpace *rfes)
      : ParDiscreteLinearOperator(dfes, rfes)
   { this->AddDomainInterpolator(new mfem::DivergenceInterpolator); }
};

/// This class computes the irrotational portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class IrrotationalProjector : public mfem::Operator
{
public:
   /// Used to set inputs in the operator
   friend void setInputs(IrrotationalProjector &op,
                         const MachInputs &inputs);

   // Given a GridFunction 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the irrotational portion, 'y', of
   // this vector field.  The resulting GridFunction will satisfy Curl y = 0
   // to machine precision.
   void Mult(const mfem::Vector &x, mfem::Vector &y) const override;

   /// \brief Reverse-mode differentiation of IrrotationalProjector::Mult
   /// \param[in] x - GridFunction 'x' that would be input to Mult
   /// \param[in] proj_bar - derivative of some output w.r.t. the projection
   /// \param[in] wrt - string indicating what to take the derivative w.r.t.
   /// \param[inout] wrt_bar - accumulated sensitivity of output w.r.t. @a wrt
   void vectorJacobianProduct(const mfem::Vector &x,
                              const mfem::Vector &proj_bar,
                              std::string wrt,
                              mfem::Vector &wrt_bar);

   IrrotationalProjector(mfem::ParFiniteElementSpace &h1_fes,
                         mfem::ParFiniteElementSpace &nd_fes,
                         const int &ir_order);

protected:
   /// Update the bilinear forms and reassemble them
   void update() const;
   /// flag indicating if the bilinear forms need to be reassembled
   mutable bool dirty;

private:
   mfem::ParFiniteElementSpace &h1_fes;
   mfem::ParFiniteElementSpace &nd_fes;

   mutable mfem::ParBilinearForm diffusion;
   mutable mfem::ParMixedBilinearForm weak_div;
   mutable DiscreteGradOperator grad;

   mfem::ParLinearForm mesh_sens;
   DiffusionIntegratorMeshSens *diff_mesh_sens;
   VectorFEWeakDivergenceIntegratorMeshSens *div_mesh_sens;

   mutable mfem::ParGridFunction psi;
   mutable mfem::ParGridFunction div_x;

   mutable mfem::Vector Psi;
   mutable mfem::Vector RHS;

   mutable mfem::HypreBoomerAMG amg;
   mutable mfem::HyprePCG pcg;

   mfem::Array<int> ess_bdr, ess_bdr_tdofs;
};

} // namespace mach

#endif
