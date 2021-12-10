#ifndef MACH_FLOW_SOLVER
#define MACH_FLOW_SOLVER

#include "mfem.hpp"

#include "pde_solver.hpp"

namespace mach
{
/// Class for solving FlowResidual based problems
/// \note This assumes a constant mass matrix at present; that is, it does not
/// accommodate entropy variables as states with modal or DGD-type
/// discretizations.  However, this could be accommodated by making the mass
/// operator its own MachResidual, and then modifying the
/// `TimeDependentResidual` accordingly.
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
   /// Bilinear form for the mass-matrix operator
   mfem::ParBilinearForm mass;
   /// Mass matrix as HypreParMatrix
   std::unique_ptr<mfem::HypreParMatrix> mass_mat;
   /// Solver used for preconditioning Newton linear updates
   std::unique_ptr<mfem::Solver> prec;

   /// Construct a preconditioner based on the given options
   /// \param[in] options - options structure that determines preconditioner
   /// \returns unique pointer to the preconditioner object
   std::unique_ptr<mfem::Solver> constructPreconditioner(
       nlohmann::json &_options);
};

}  // namespace mach

#endif