#ifndef MACH_EQUATION_SOLVER
#define MACH_EQUATION_SOLVER

#include <memory>
#include <optional>
#include <variant>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

namespace mach
{
/** \brief Wraps a system solver and handles the configuration of linear or
    nonlinear solvers.  This class solves a generic global system of (possibly)
    nonlinear algebraic equations. */
class EquationSolver : public mfem::Solver
{
public:
   // TODO: Eliminate this once a dependency injection approach is used for the
   // solvers
   EquationSolver() = default;

   /// Constructs a new solver wrapper
   /// \param[in] comm - the MPI communicator object
   /// \param[in] lin_options - the parameters for the linear solver
   /// \param[in] nonlin_options - the optional parameters for the optional
   /// nonlinear solver
   EquationSolver(
       MPI_Comm comm,
       const nlohmann::json &lin_options,
       std::unique_ptr<mfem::Solver> prec = nullptr,
       const std::optional<nlohmann::json> &nonlin_options = std::nullopt);

   /// Sets the linear solver preconditioner
   /// \param[in] prec - the linear solver preconditioner
   void SetPreconditioner(std::unique_ptr<mfem::Solver> &&prec)
   {
      prec_ = std::move(prec);
   }

   /// Updates the solver with the provided operator
   /// \param[in] op - the operator (system matrix) to use, "A" in Ax = b
   /// \note Implements mfem::Operator::SetOperator
   void SetOperator(const mfem::Operator &op) override;

   /// Solves the system
   /// \param[in] b - RHS of the system of equations
   /// \param[out] x - solution to the system of equations
   /// \note Implements mfem::Operator::Mult
   void Mult(const mfem::Vector &b, mfem::Vector &x) const override;

   /// Returns the underlying solver object
   /// \return A non-owning reference to the underlying nonlinear solver
   mfem::Solver &NonlinearSolver() { return *nonlin_solver_; }

   /// \overload
   const mfem::Solver &NonlinearSolver() const { return *nonlin_solver_; }

   /// Returns the underlying linear solver object
   /// \return A non-owning reference to the underlying linear solver
   mfem::Solver &LinearSolver() { return *lin_solver_; }

   /// \overload
   const mfem::Solver &LinearSolver() const { return *lin_solver_; }

private:
   // /// \brief Builds a preconditioner given a set of preconditioner
   // parameters
   // /// \param[in] comm The MPI communicator object
   // /// \param[in] prec_options The parameters for the preconditioner solver
   // std::unique_ptr<mfem::Solver> constructPreconditioner(
   //     MPI_Comm comm,
   //     const nlohmann::json &prec_options);

   /// \brief Builds an iterative solver given a set of linear solver parameters
   /// \param[in] comm - The MPI communicator object
   /// \param[in] lin_options - The parameters for the linear solver
   static std::unique_ptr<mfem::Solver> constructLinearSolver(
       MPI_Comm comm,
       const nlohmann::json &lin_options);

   /// \brief Builds an nonlinear solver given a set of nonlinear solver
   /// parameters
   /// \param[in] comm - The MPI communicator object
   /// \param[in] nonlin_options - The parameters for the nonlinear solver
   static std::unique_ptr<mfem::NewtonSolver> constructNonlinearSolver(
       MPI_Comm comm,
       const nlohmann::json &nonlin_options);

   /// \brief The linear solver preconditioner
   std::unique_ptr<mfem::Solver> prec_;

   /// \brief The linear solver object
   std::unique_ptr<mfem::Solver> lin_solver_;

   /// \brief The optional nonlinear Newton-Raphson solver object
   std::unique_ptr<mfem::NewtonSolver> nonlin_solver_;

   /// \brief Whether the solver (linear solver) has been configured with the
   /// nonlinear solver
   /// \note This is a workaround as some nonlinear solvers require SetOperator
   /// to be called before SetSolver */
   bool nonlin_solver_set_solver_called_ = false;
};

}  // namespace mach

#endif
