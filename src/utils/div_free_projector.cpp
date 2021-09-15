#include "div_free_projector.hpp"

using namespace mfem;

namespace mach
{
void setInputs(DivergenceFreeProjector &op, const MachInputs &inputs)
{
   auto it = inputs.find("mesh_coords");
   if (it != inputs.end())
   {
      op.dirty = true;
   }
}

void DivergenceFreeProjector::Mult(const Vector &x, Vector &y) const
{
   this->IrrotationalProjector::Mult(x, y);
   y -= x;
   y *= -1.0;
}

void DivergenceFreeProjector::vectorJacobianProduct(
    const mfem::Vector &x,
    const mfem::Vector &proj_bar,
    std::string wrt,
    mfem::Vector &wrt_bar)
{
   if (wrt == "in")
   {
      psi_irrot = proj_bar;
      psi_irrot *= -1.0;
      this->IrrotationalProjector::vectorJacobianProduct(
          x, psi_irrot, wrt, wrt_bar);
      wrt_bar += proj_bar;
   }
   else if (wrt == "mesh_coords")
   {
      psi_irrot = proj_bar;
      psi_irrot *= -1.0;
      this->IrrotationalProjector::vectorJacobianProduct(
          x, psi_irrot, wrt, wrt_bar);
   }
}

}  // namespace mach
