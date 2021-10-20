#ifndef MFEM_EXTENSIONS
#define MFEM_EXTENSIONS

#include "nlohmann/json.hpp"
#include "mfem.hpp"

namespace mach
{
/// Backward Euler pseudo-transient continuation solver
class PseudoTransientSolver : public mfem::ODESolver
{
public:
   PseudoTransientSolver(std::ostream *out_stream) : out(out_stream) { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Relaxation version of implicit midpoint method
class RRKImplicitMidpointSolver : public mfem::ODESolver
{
public:
   RRKImplicitMidpointSolver(std::ostream *out_stream) : out(out_stream) { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Construct a preconditioner based on the given options
/// \param[in] options - options structure that determines preconditioner
/// \returns unique pointer to the preconditioner object
std::unique_ptr<mfem::Solver> constructPreconditioner(
   nlohmann::json &options,
   MPI_Comm comm);

/// Constuct a linear system solver based on the given options
/// \param[in] options - options structure that determines the solver
/// \param[in] prec - preconditioner object for iterative solvers
/// \returns unique pointer to the linear solver object
std::unique_ptr<mfem::Solver> constructLinearSolver(
    nlohmann::json &options,
    mfem::Solver &prec,
    MPI_Comm comm);

/// Constructs the nonlinear solver object
/// \param[in] options - options structure that determines the solver
/// \param[in] lin_solver - linear solver for the Newton steps
/// \returns unique pointer to the Newton solver object
std::unique_ptr<mfem::NewtonSolver> constructNonlinearSolver(
    nlohmann::json &options,
    mfem::Solver &lin_solver,
    MPI_Comm comm);

/// Construct an `ODESolver` object based on the given options
/// \param[in] options - options structure that determines ODE solver
/// \returns unique pointer to the `ODESolver` object
std::unique_ptr<mfem::ODESolver> constructODESolver(nlohmann::json &options);

}  // namespace mach

#endif
