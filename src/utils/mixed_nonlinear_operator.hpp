#ifndef MACH_L2_TRANSFER_OPERATOR
#define MACH_L2_TRANSFER_OPERATOR

#include "mfem.hpp"

#include "finite_element_state.hpp"
#include "mach_input.hpp"

namespace mach
{
/// Class that handles the potentially nonlinear transformation from a field in
/// an ND or RT function space to a representation of the transformed field in
/// an L2 space
/// Currently only supports element based transformations
class L2TransferOperator
{
public:
   /// compute the action of the operator
   void apply(const MachInputs &inputs, mfem::Vector &out_vec);
   /// \overload
   void apply(const mfem::Vector &state, mfem::Vector &out_vec)
   {
      MachInputs inputs{{"state", state}};
      apply(inputs, out_vec);
   }

   void vectorJacobianProduct(const std::string &wrt,
                              const MachInputs &inputs,
                              const mfem::Vector &out_bar,
                              mfem::Vector &wrt_bar);

   L2TransferOperator(FiniteElementState &domain,
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

class L2IdentityProjection : public L2TransferOperator
{
   L2IdentityProjection(FiniteElementState &domain, FiniteElementState &range);
};

class L2CurlProjection : public L2TransferOperator
{
   L2CurlProjection(FiniteElementState &domain, FiniteElementState &range);
};

class L2CurlMagnitudeProjection : public L2TransferOperator
{
   L2CurlMagnitudeProjection(FiniteElementState &domain,
                             FiniteElementState &range);
};

inline void calcOutput(L2TransferOperator &output,
                       const MachInputs &inputs,
                       mfem::Vector &out_vec)
{
   output.apply(inputs, out_vec);
}

inline void vectorJacobianProduct(L2TransferOperator &output,
                                  const std::string &wrt,
                                  const MachInputs &inputs,
                                  const mfem::Vector &out_bar,
                                  mfem::Vector &wrt_bar)
{
   output.vectorJacobianProduct(wrt, inputs, out_bar, wrt_bar);
}

}  // namespace mach

#endif
