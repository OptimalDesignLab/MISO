#ifndef MACH_ABSTRACT_SOLVER
#define MACH_ABSTRACT_SOLVER

#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "data_logging.hpp"
#include "equation_solver.hpp"
#include "mach_input.hpp"
#include "mach_output.hpp"
#include "mach_residual.hpp"
#include "ode.hpp"

namespace mach
{
/// Serves as a base class for specific solvers
/// \todo Rename to AbstractSolver once we have old AbstractSolver inherit it
class AbstractSolver2
{
public:
   /// Solve for the state based on the residual `res` and `options`
   /// \param[inout] state - the solution to the governing equation
   /// \note On input, `state` should hold the initial condition
   void solveForState(mfem::Vector &state) { solveForState({}, state); }

   /// Solve for the state based on the residual `res` and `options`
   /// \param[in] inputs - scalars and fields that the `res` may depend on
   /// \param[inout] state - the solution to the governing equation
   /// \note On input, `state` should hold the initial condition
   void solveForState(const MachInputs &inputs, mfem::Vector &state);

   MachResidual &residual() { return *res; }
   const MachResidual &residual() const { return *res; }

   /// Compute the residual and store the it in @a residual
   /// \param[in] state - the state to evaluate the residual at
   /// \param[out] residual - the discrete residual vector
   void calcResidual(const mfem::Vector &state,
                     mfem::Vector &residual) const;

   /// Compute the residual based on inputs and store the it in @a residual
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \param[out] residual - the discrete residual vector
   void calcResidual(const MachInputs &inputs,
                     mfem::Vector &residual) const;

   /// Compute the residual norm based on inputs
   /// \param[in] state - the state to evaluate the residual at
   /// \return the norm of the discrete residual vector
   double calcResidualNorm(const mfem::Vector &state) const;
   
   /// Compute the residual norm based on inputs
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \return the norm of the discrete residual vector
   double calcResidualNorm(const MachInputs &inputs) const;

   int getStateSize() const { return getSize(*res); }
   virtual int getFieldSize(std::string name) const;

   // /// Creates the nonlinear form for the functional
   // /// \param[in] fun - specifies the desired functional
   // /// \param[in] options - options needed for calculating functional
   // /// \note if a nonlinear form for `fun` has already been created an
   // /// exception will be thrown
   // void createOutput(const std::string &fun, const nlohmann::json &options);

   // /// Evaluates and returns the output functional specifed by `fun`
   // /// \param[in] fun - specifies the desired functional
   // /// \param[in] inputs - collection of field or scalar inputs to set before
   // ///                     evaluating functional
   // /// \return scalar value of estimated functional value
   // double calcOutput(const std::string &fun, const MachInputs &inputs);

   AbstractSolver2(MPI_Comm incomm, const nlohmann::json &solver_options);

   virtual ~AbstractSolver2() = default;

protected:
   /// communicator used by MPI group for communication
   MPI_Comm comm;
   /// MPI process rank
   int rank;

   /// work vector for solvers
   mutable mfem::Vector work;

   /// solver options
   nlohmann::json options;

   /// residual defines the dynamics of an ODE (including steady ODEs)
   std::unique_ptr<MachResidual> res;

   /// \brief the ordinary differential equation that describes how to evolve
   /// the state variables
   std::unique_ptr<FirstOrderODE> ode;

   /// Optional data loggers that will save state vectors during timestepping
   std::vector<DataLoggerWithOpts> loggers;

   void addLogger(DataLogger logger, LoggingOptions options)
   {
      loggers.emplace_back(std::make_pair<DataLogger, LoggingOptions>(
          std::move(logger), std::move(options)));
   }

   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   virtual void initialHook(const mfem::Vector &state);

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::Vector &state);

   /// Find the step size based on the options
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt_old - the step size that was just taken
   /// \param[in] state - the current state
   /// \returns dt - the step size appropriate to the problem
   /// \note The base method simply returns the option in ["time-dis"]["dt"],
   /// truncated as necessary such that `t + dt = t_final`.
   virtual double calcStepSize(int iter,
                               double t,
                               double t_final,
                               double dt_old,
                               const mfem::Vector &state) const;

   /// Determines when to exit the time stepping loop
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (after the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt - the step size that was just taken
   /// \param[in] state - the current state
   /// \note The base method just checks if `t >= t_final`.
   virtual bool iterationExit(int iter,
                              double t,
                              double t_final,
                              double dt,
                              const mfem::Vector &state) const;

   /// For code that should be executed after the time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void terminalHook(int iter,
                             double t_final,
                             const mfem::Vector &state);

   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> linear_solver;
   /// newton solver for solving implicit problems
   std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;
};

}  // namespace mach

#endif
