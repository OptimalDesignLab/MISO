#include "finite_element_state.hpp"

namespace mach
{
double norm(const FiniteElementState &state, const double p)
{
   if (state.space().GetVDim() == 1)
   {
      mfem::ConstantCoefficient zero(0.0);
      return state.gridFunc().ComputeLpError(p, zero);
   }
   else
   {
      mfem::Vector zero(state.space().GetVDim());
      zero = 0.0;
      mfem::VectorConstantCoefficient zerovec(zero);
      return state.gridFunc().ComputeLpError(p, zerovec);
   }
}

}  // namespace mach
