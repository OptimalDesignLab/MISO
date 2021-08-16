#ifndef MACH_DIV_FREE_PROJECTOR
#define MACH_DIV_FREE_PROJECTOR

#include "mfem.hpp"

#include "irrotational_projector.hpp"

namespace mach
{

/// This class computes the divergence free portion of a vector field.
/// This vector field must be discretized using Nedelec basis
/// functions.
class DivergenceFreeProjector : public IrrotationalProjector
{
public:
   /// Used to set inputs in the operator
   friend void setInputs(DivergenceFreeProjector &op,
                         const MachInputs &inputs);

   // Given a GridFunction 'x' of Nedelec DoFs for an arbitrary vector field,
   // compute the Nedelec DoFs of the divergence free portion, 'y', of
   // this vector field.  The resulting GridFunction will satisfy Div y = 0
   // in a weak sense.
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   /// \brief Reverse-mode differentiation of DivergenceFreeProjector::Mult
   /// \param[in] x - GridFunction 'x' that would be input to Mult
   /// \param[in] proj_bar - derivative of some output w.r.t. the projection
   /// \param[in] wrt - string indicating what to take the derivative w.r.t.
   /// \param[inout] wrt_bar - accumulated sensitivity of output w.r.t. @a wrt
   void vectorJacobianProduct(const mfem::Vector &x,
                              const mfem::Vector &proj_bar,
                              std::string wrt,
                              mfem::Vector &wrt_bar);

   DivergenceFreeProjector(mfem::ParFiniteElementSpace &h1_fes,
                           mfem::ParFiniteElementSpace &nd_fes,
                           const int &ir_order)
      : IrrotationalProjector(h1_fes, nd_fes, ir_order), psi_irrot(&nd_fes)
   { }

private:
   mutable mfem::ParGridFunction psi_irrot;
};

} // namespace mach

#endif
