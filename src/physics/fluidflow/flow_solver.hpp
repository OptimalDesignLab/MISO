#ifndef MACH_FLOW_SOLVER
#define MACH_FLOW_SOLVER

#include "mfem.hpp"

#include "pde_solver.hpp"

namespace mach
{
class FlowSolver : public PDESolver
{
public:
   /// Construct a `FlowSolver` based on the given options and mesh
   /// \param[in] incomm - the MPI communicator to associate with this solver
   /// \param[in] solver_options - options used to define the solver 
   /// \param[in] smesh - serial mesh; if not `nullptr`, moved to this solver
   FlowSolver(MPI_Comm incomm,
              const nlohmann::json &solver_options,
              std::unique_ptr<mfem::Mesh> smesh = nullptr);

private: 

};

}  // namespace mach

#endif