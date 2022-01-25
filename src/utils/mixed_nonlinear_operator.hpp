#ifndef MACH_MIXED_NONLINEAR_OPERATOR
#define MACH_MIXED_NONLINEAR_OPERATOR

#include "finite_element_vector.hpp"

namespace mach
{
/// Class that handles the potentially nonlinear transformation from a field in
/// one function space to a field in another space
/// Currently only supports element based transformations
class MixedNonlinearOperator
{
public:
   /// compute the action of the operator
   void apply(const mfem::Vector &x, mfem::Vector &y);

   MixedNonlinearOperator(FiniteElementVector &domain,
                          FiniteElementVector &range);

private:
   FiniteElementVector &domain;
   FiniteElementVector &range;
};

}  // namespace mach

#endif
