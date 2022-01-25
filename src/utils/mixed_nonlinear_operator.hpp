#ifndef MACH_MIXED_NONLINEAR_OPERATOR
#define MACH_MIXED_NONLINEAR_OPERATOR

#include "mfem.hpp"

#include "finite_element_state.hpp"
#include "mach_input.hpp"

namespace mach
{
/// Class that handles the potentially nonlinear transformation from a field in
/// one function space to a field in another space
/// Currently only supports element based transformations
class MixedNonlinearOperator
{
public:
   void apply(const mfem::Vector in_vec, mfem::Vector &out_vec)
   {
      MachInputs inputs{{"state", in_vec}};
      apply(inputs, out_vec);
   }
   /// compute the action of the operator
   void apply(const MachInputs &inputs, mfem::Vector &out_vec);

   MixedNonlinearOperator(FiniteElementState &domain,
                          FiniteElementState &range,
                          std::function<void(const mfem::FiniteElement &,
                                             const mfem::FiniteElement &,
                                             mfem::ElementTransformation &,
                                             const mfem::Vector &,
                                             mfem::Vector &)> operation)
    : domain(domain), range(range), operation(std::move(operation))
   { }

private:
   FiniteElementState &domain;
   FiniteElementState &range;

   std::function<void(const mfem::FiniteElement &,
                      const mfem::FiniteElement &,
                      mfem::ElementTransformation &,
                      const mfem::Vector &,
                      mfem::Vector &)>
       operation;
};

}  // namespace mach

#endif
