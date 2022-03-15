#ifndef MACH_ABSTRACT_SOLVER
#define MACH_ABSTRACT_SOLVER

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "data_logging.hpp"
#include "mach_input.hpp"
#include "mach_output.hpp"
#include "mach_residual.hpp"
#include "ode.hpp"
#include "utils.hpp"

namespace mach
{
/// Serves as a base class for specific solvers
/// \todo Rename to AbstractSolver once we have old AbstractSolver inherit it
class AbstractSolver2
{
public:
   /// \brief Generic function that allows derived classes to set the state
   /// based on the type T
   /// \param[in] function - any object that a derived solver will know how to
   /// interpret and use to set the state
   /// \param[out] state - the true dof vector to set
   /// \param[in] name - name of the vector to set, defaults to "state"
   /// \tparam T - generic type T, the derived classes must know how to use it
   template <typename T>
   void setState(T function,
                 mfem::Vector &state,
                 const std::string &name = "state");

   /// \brief Generic function that allows derived classes to calculate the
   /// error in the state
   /// \param[in] ex_sol - any object that a derived solver will know how to
   /// interpret and use to calculate the error in the state
   /// \param[out] state - the true dof vector we're getting the error of
   /// \param[in] name - name of the state vector, defaults to "state"
   /// \tparam T - generic type T, the derived classes must know how to use it
   template <typename T>
   double calcStateError(T ex_sol,
                         const mfem::Vector &state,
                         const std::string &name = "state");

   /// Solve for the state based on the residual `res` and `options`
   /// \param[inout] state - the solution to the governing equation
   /// \note On input, `state` should hold the initial condition
   void solveForState(mfem::Vector &state) { solveForState({}, state); }

   /// Solve for the state based on the residual `res` and `options`
   /// \param[in] inputs - scalars and fields that the `res` may depend on
   /// \param[inout] state - the solution to the governing equation
   /// \note On input, `state` should hold the initial condition
   void solveForState(const MachInputs &inputs, mfem::Vector &state);

   /// Compute the residual and store the it in @a residual
   /// \param[in] state - the state to evaluate the residual at
   /// \param[out] residual - the discrete residual vector
   void calcResidual(const mfem::Vector &state, mfem::Vector &residual) const;

   /// Compute the residual based on inputs and store the it in @a residual
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \param[out] residual - the discrete residual vector
   void calcResidual(const MachInputs &inputs, mfem::Vector &residual) const;

   /// Compute the residual norm based on inputs
   /// \param[in] state - the state to evaluate the residual at
   /// \return the norm of the discrete residual vector
   double calcResidualNorm(const mfem::Vector &state) const;

   /// Compute the residual norm based on inputs
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \return the norm of the discrete residual vector
   double calcResidualNorm(const MachInputs &inputs) const;

   /// \return the size of the state vector
   int getStateSize() const;

   /// Gets the size of a field @a name known by the solver
   /// \param[in] name - the name of the field to look up the size of
   /// \return the discrete size of the field identified by @a name
   /// \note if the field @a name is unrecognized by the solver, 0 is returned
   virtual int getFieldSize(const std::string &name) const;

   /// Creates a MachOutput for the specified @a output based on @a options
   /// \param[in] output - specifies the desired output
   /// \note if an output for @a output has already been created an
   /// exception will be thrown
   void createOutput(const std::string &output);

   /// Creates a MachOutput for the specified @a output based on @a options
   /// \param[in] output - specifies the desired output
   /// \param[in] options - options needed for configuring the output
   /// \note if an output for @a output has already been created an
   /// exception will be thrown
   void createOutput(const std::string &output, const nlohmann::json &options);

   /// Sets options for the output specifed by @a output
   /// \param[in] output - specifies the desired output
   /// \param[in] options - options needed for configuring the output
   /// \note will only have an effect if a concrete output supports setting
   ///       options
   void setOutputOptions(const std::string &output,
                         const nlohmann::json &options);

   /// Evaluates and returns the output specifed by @a output
   /// \param[in] output - specifies the desired output
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating the output
   /// \return scalar value of estimated output value
   double calcOutput(const std::string &output, const MachInputs &inputs);

   /// Evaluates the vector-valued output specifed by @a output
   /// \param[in] output - specifies the desired output
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating the output
   /// \param[out] out_vec - estimated vector-valued output
   void calcOutput(const std::string &output,
                   const MachInputs &inputs,
                   mfem::Vector &out_vec);

   /// Evaluates and returns the partial derivative of output specifed by
   /// `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired output
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating the output partial
   /// \param[out] partial - the partial with respect to a scalar-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          double &partial);

   /// Evaluates and returns the partial derivative of the output specifed by
   /// `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired output
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating the output partial
   /// \param[out] partial - the partial with respect to a vector-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          mfem::Vector &partial);

   AbstractSolver2(MPI_Comm incomm, const nlohmann::json &solver_options);

   virtual ~AbstractSolver2() = default;

protected:
   /// communicator used by MPI group for communication
   MPI_Comm comm;

   /// MPI process rank
   int rank;

   /// print object
   std::ostream *out;

   /// storage for algorithmic differentiation (shared by all solvers)
   static adept::Stack diff_stack;

   /// work vector for solvers
   mutable mfem::Vector work;

   /// solver options
   nlohmann::json options;

   /// residual defines the just the spatial dynamics of a problem
   std::unique_ptr<MachResidual> spatial_res;
   /// residual defines the dynamics of an ODE (including steady ODEs)
   std::unique_ptr<MachResidual> space_time_res;

   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> linear_solver;
   /// newton solver for solving implicit problems
   std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

   /// \brief the ordinary differential equation that describes how to evolve
   /// the state variables
   std::unique_ptr<FirstOrderODE> ode;

   /// map of outputs the solver can compute
   std::map<std::string, MachOutput> outputs;

   /// Optional data loggers that will save state vectors during timestepping
   std::vector<DataLoggerWithOpts> loggers;

   void addLogger(DataLogger logger, LoggingOptions &&options)
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

   /// Add output @a out based on @a options
   virtual void addOutput(const std::string &out, const nlohmann::json &options)
   { }

   /// \brief Virtual method that allows derived solvers to deal with inputs
   /// from templated function setState
   /// \param[in] function - input function used to set the state
   /// \param[in] name - name of the vector to set
   /// \param[out] state - the true dof vector to set
   /// \note the derived classes must know what types @a function may hold and
   /// how to access/use them
   virtual void setState_(std::any function,
                          const std::string &name,
                          mfem::Vector &state);

   /// \brief Virtual method that allows derivated solvers to deal with inputs
   /// from templated function calcStateError
   /// \param[in] ex_sol - input function describing exact solution
   /// \param[in] name - name of the state vector
   /// \param[out] state - the true dof vector we're getting the error of
   /// \note the derived classes must know what types @a ex_sol may hold and
   /// how to access/use them
   virtual double calcStateError_(std::any ex_sol,
                                  const std::string &name,
                                  const mfem::Vector &state);
};

template <typename T>
void AbstractSolver2::setState(T function,
                               mfem::Vector &state,
                               const std::string &name)
{
   /// compile time conditional that checks if @a function is callable, and
   /// thus should be converted to a std::function
   if constexpr (is_callable_v<T>)
   {
      auto fun = make_function(function);
      auto any = std::make_any<decltype(fun)>(fun);
      setState_(any, name, state);
   }
   /// if @a function is not callable, we just pass it directly along
   else
   {
      auto any = [&]() constexpr
      {
         /// If T is either an mfem::Coefficient or an mfem::VectorCoefficient,
         /// we create the std::any to be a pointer to the base class, so that
         /// when casting the any to a concrete type we can just interact with
         /// the coefficient through the base class pointer
         if constexpr (std::is_base_of_v<mfem::Coefficient, T>)
         {
            return std::make_any<mfem::Coefficient *>(&function);
         }
         else if constexpr (std::is_base_of_v<mfem::VectorCoefficient, T>)
         {
            return std::make_any<mfem::VectorCoefficient *>(&function);
         }
         else
         {
            return std::make_any<decltype(function)>(function);
         }
      }
      ();
      setState_(any, name, state);
   }
}

template <typename T>
double AbstractSolver2::calcStateError(T ex_sol,
                                       const mfem::Vector &state,
                                       const std::string &name)
{
   /// compile time conditional that checks if @a ex_sol is callable, and
   /// thus should be converted to a std::function
   if constexpr (is_callable_v<T>)
   {
      auto fun = make_function(ex_sol);
      auto any = std::make_any<decltype(fun)>(fun);
      return calcStateError_(any, name, state);
   }
   /// if @a ex_sol is not callable, we just pass it directly along
   else
   {
      auto any = [&]() constexpr
      {
         /// If T is either an mfem::Coefficient or an mfem::VectorCoefficient,
         /// we create the std::any to be a pointer to the base class, so that
         /// when casting the any to a concrete type we can just interact with
         /// the coefficient through the base class pointer
         if constexpr (std::is_base_of_v<mfem::Coefficient, T>)
         {
            return std::make_any<mfem::Coefficient *>(&ex_sol);
         }
         else if constexpr (std::is_base_of_v<mfem::VectorCoefficient, T>)
         {
            return std::make_any<mfem::VectorCoefficient *>(&ex_sol);
         }
         else
         {
            return std::make_any<decltype(ex_sol)>(ex_sol);
         }
      }
      ();
      return calcStateError_(any, name, state);
   }
}

}  // namespace mach

#endif
