
#include "flow_solver.hpp"
#include "flow_residual.hpp"

using namespace std;
using namespace mfem;

/// Return the number of flow state variables
/// \param[in] solver_options - helps define/determine the number of states
/// \param[in] space_dim - the number of spatial dimensions, consistent mesh
/// \todo update for any RANS models, since this assumes no additional states
int getNumFlowStates(const nlohmann::json &solver_options,
                     int space_dim)
{
   return space_dim + 2;
}

namespace mach
{
FlowSolver::FlowSolver(MPI_Comm incomm,
                       const nlohmann::json &solver_options,
                       std::unique_ptr<mfem::Mesh> smesh)
 :
   PDESolver(incomm, solver_options, getNumFlowStates, std::move(smesh))
{ 
   res = make_unique<MachResidual>(FlowResidual(solver_options, fes(), 
                                                diff_stack));
}

} // namespace mach