#ifndef MACH_L2_TRANSFER_OPERATOR
#define MACH_L2_TRANSFER_OPERATOR

#include "mfem.hpp"

#include "finite_element_dual.hpp"
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
   void apply(const mfem::Vector &state_tv, mfem::Vector &out_vec)
   {
      MachInputs inputs{{"state", state_tv}};
      apply(inputs, out_vec);
   }

   void vectorJacobianProduct(const std::string &wrt,
                              const MachInputs &inputs,
                              const mfem::Vector &out_bar,
                              mfem::Vector &wrt_bar);

   L2TransferOperator(FiniteElementState &state,
                      FiniteElementState &output,
                      std::function<void(const mfem::FiniteElement &,
                                         const mfem::FiniteElement &,
                                         mfem::ElementTransformation &,
                                         const mfem::Vector &,
                                         mfem::Vector &)> operation,
                      std::function<void(const mfem::FiniteElement &,
                                         const mfem::FiniteElement &,
                                         mfem::ElementTransformation &,
                                         const mfem::Vector &,
                                         const mfem::Vector &,
                                         mfem::Vector &)> operation_state_bar)
    : state(state),
      output(output),
      output_adjoint(output.mesh(), output.space()),
      state_bar(state.mesh(), state.space()),
      operation(std::move(operation)),
      operation_state_bar(std::move(operation_state_bar))
   { }

private:
   FiniteElementState &state;
   FiniteElementState &output;
   FiniteElementState output_adjoint;
   FiniteElementDual state_bar;

   std::function<void(const mfem::FiniteElement &,
                      const mfem::FiniteElement &,
                      mfem::ElementTransformation &,
                      const mfem::Vector &,
                      mfem::Vector &)>
       operation;

   std::function<void(const mfem::FiniteElement &,
                      const mfem::FiniteElement &,
                      mfem::ElementTransformation &,
                      const mfem::Vector &,
                      const mfem::Vector &,
                      mfem::Vector &)>
       operation_state_bar;
};

/// Conveniece class that wraps the projection of an H1 state to its DG
/// representation
class ScalarL2IdentityProjection : public L2TransferOperator
{
public:
   ScalarL2IdentityProjection(FiniteElementState &state,
                              FiniteElementState &output);
};

/// Conveniece class that wraps the projection of an state to its DG
/// representation
class L2IdentityProjection : public L2TransferOperator
{
public:
   L2IdentityProjection(FiniteElementState &state, FiniteElementState &output);
};

/// Conveniece class that wraps the projection of the curl of the state to its
/// DG representation
class L2CurlProjection : public L2TransferOperator
{
public:
   L2CurlProjection(FiniteElementState &state, FiniteElementState &output);
};

/// Conveniece class that wraps the projection of the magnitude of the curl of
/// the state to its DG representation
class L2CurlMagnitudeProjection : public L2TransferOperator
{
public:
   L2CurlMagnitudeProjection(FiniteElementState &state,
                             FiniteElementState &output);
};

inline double calcOutput(L2TransferOperator &output, const MachInputs &inputs)
{
   return NAN;
}

inline void calcOutput(L2CurlMagnitudeProjection &output,
                       const MachInputs &inputs,
                       mfem::Vector &out_vec)
{
   output.apply(inputs, out_vec);
}

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
