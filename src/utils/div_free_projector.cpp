#include "div_free_projector.hpp"

using namespace mfem;

namespace mach
{

DivergenceFreeProjector::DivergenceFreeProjector(ParFiniteElementSpace &h1_fes,
                                                 ParFiniteElementSpace &nd_fes,
                                                 const int &ir_order)
   : IrrotationalProjector(h1_fes, nd_fes, ir_order)
{}

void DivergenceFreeProjector::Mult(const Vector &x, Vector &y) const
{
   this->IrrotationalProjector::Mult(x, y);
   y  -= x;
   y *= -1.0;
}

void DivergenceFreeProjector::vectorJacobianProduct(
   const mfem::ParGridFunction &proj_bar,
   std::string wrt,
   mfem::ParGridFunction &wrt_bar)
{
   if (wrt == "wrt")
   {

   }
}

} // namespace mach
