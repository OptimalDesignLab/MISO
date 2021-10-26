#ifndef MFEM_EXTENSIONS
#define MFEM_EXTENSIONS

#include "mfem.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
/// steady ode solver
class SteadyODESolver : public mfem::ODESolver
{
public:
   SteadyODESolver(std::ostream *out_stream = nullptr) : out(out_stream)
   { }

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Backward Euler pseudo-transient continuation solver
class PseudoTransientSolver : public mfem::ODESolver
{
public:
   PseudoTransientSolver(std::ostream *out_stream = nullptr) : out(out_stream)
   { }

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
   RRKImplicitMidpointSolver(std::ostream *out_stream = nullptr)
    : out(out_stream)
   { }

   void Init(mfem::TimeDependentOperator &_f) override;

   void Step(mfem::Vector &x, double &t, double &dt) override;

protected:
   mfem::Vector k;
   std::ostream *out;
};

/// Constuct a linear system solver based on the given options
/// \param[in] comm - MPI communicator used by linear solver
/// \param[in] lin_options - options structure that determines the solver
/// \param[in] prec - non-owning pointer to preconditioner for linear solvers
/// \return unique pointer to the linear solver object
std::unique_ptr<mfem::Solver> constructLinearSolver(
    MPI_Comm comm,
    const nlohmann::json &lin_options,
    mfem::Solver *prec = nullptr);

/// Constructs the nonlinear solver object
/// \param[in] comm - MPI communicator used by non-linear solver
/// \param[in] nonlin_options - options structure that determines the solver
/// \param[in] lin_solver - linear solver for the Newton steps
/// \return unique pointer to the Newton solver object
std::unique_ptr<mfem::NewtonSolver> constructNonlinearSolver(
    MPI_Comm comm,
    const nlohmann::json &nonlin_options,
    mfem::Solver &lin_solver);

}  // namespace mach

#endif
