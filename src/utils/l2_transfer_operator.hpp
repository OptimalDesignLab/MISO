#ifndef MACH_L2_TRANSFER_OPERATOR
#define MACH_L2_TRANSFER_OPERATOR

#include "mfem.hpp"

#include "finite_element_dual.hpp"
#include "finite_element_state.hpp"
#include "mach_input.hpp"

namespace mach
{
class L2TransferOperation
{
public:
   /// \param[in] state_fe - finite element that describes the mesh element
   /// \param[in] output_fe - finite element that describes the mesh element
   /// \param[in] trans - transformation between reference and physical space
   /// \param[in] el_state - input state defined on @a state_fe
   /// \param[out] el_output - output of operation defined on @a output_fe
   virtual void apply(const mfem::FiniteElement &state_fe,
                      const mfem::FiniteElement &output_fe,
                      mfem::ElementTransformation &trans,
                      const mfem::Vector &el_state,
                      mfem::Vector &el_output) const = 0;

   /// \param[in] state_fe - finite element that describes the mesh element
   /// \param[in] output_fe - finite element that describes the mesh element
   /// \param[in] trans - transformation between reference and physical space
   /// \param[in] el_output_adj - the element local output adjoint
   /// \param[in] el_state - input state defined on @a state_fe
   /// \param[out] el_state_bar - d(psi^T Op)/du for the element
   virtual void apply_state_bar(const mfem::FiniteElement &state_fe,
                                const mfem::FiniteElement &output_fe,
                                mfem::ElementTransformation &trans,
                                const mfem::Vector &el_output_adj,
                                const mfem::Vector &el_state,
                                mfem::Vector &el_state_bar) const
   { }

   /// \param[in] state_fe - finite element that describes the mesh element
   /// \param[in] output_fe - finite element that describes the mesh element
   /// \param[in] trans - transformation between reference and physical space
   /// \param[in] el_output_adj - the element local output adjoint
   /// \param[in] el_state - input state defined on @a state_fe
   /// \param[out] mesh_coords_bar - d(psi^T Op)/dX for the element
   virtual void apply_mesh_coords_bar(const mfem::FiniteElement &state_fe,
                                      const mfem::FiniteElement &output_fe,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &el_output_adj,
                                      const mfem::Vector &el_state,
                                      mfem::Vector &mesh_coords_bar) const
   { }

   virtual ~L2TransferOperation() = default;
};

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

   friend void setInputs(L2TransferOperator &output, const MachInputs &inputs);

   void vectorJacobianProduct(const mfem::Vector &out_bar,
                              const std::string &wrt,
                              mfem::Vector &wrt_bar);

   L2TransferOperator(FiniteElementState &state,
                      FiniteElementState &output,
                      std::unique_ptr<L2TransferOperation> operation)
    : state(state),
      output(output),
      output_adjoint(output.mesh(), output.space()),
      state_bar(state.mesh(), state.space()),
      mesh_coords_bar(
          state.mesh(),
          *dynamic_cast<mfem::ParGridFunction *>(state.mesh().GetNodes())
               ->ParFESpace()),
      operation(std::move(operation))
   { }

private:
   FiniteElementState &state;
   FiniteElementState &output;
   FiniteElementState output_adjoint;
   FiniteElementDual state_bar;
   FiniteElementDual mesh_coords_bar;

   std::unique_ptr<L2TransferOperation> operation;
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

inline void setInputs(L2TransferOperator &output, const MachInputs &inputs)
{
   mfem::Vector state_tv;
   setVectorFromInputs(inputs, "state", state_tv, false, true);
   output.state.distributeSharedDofs(state_tv);
}

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
                                  const mfem::Vector &out_bar,
                                  const std::string &wrt,
                                  mfem::Vector &wrt_bar)
{
   output.vectorJacobianProduct(out_bar, wrt, wrt_bar);
}

}  // namespace mach

#endif
