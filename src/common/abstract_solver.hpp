#ifndef MISO_ABSTRACT_SOLVER
#define MISO_ABSTRACT_SOLVER

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "data_logging.hpp"
#include "miso_input.hpp"
#include "miso_output.hpp"
#include "miso_residual.hpp"
#include "ode.hpp"
#include "utils.hpp"

namespace miso
{
/// Serves as a base class for specific solvers
/// \todo Rename to AbstractSolver once we have old AbstractSolver inherit it
class AbstractSolver2
{
public:
   /// \brief Set solver options, overwriting existing options
   void setOptions(const nlohmann::json &options)
   {
      AbstractSolver2::options.update(options, true);
   }

   /// \brief Retrieve the currently set solver options
   const nlohmann::json &getOptions() const { return options; }

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
   void solveForState(const MISOInputs &inputs, mfem::Vector &state);

   /// Solve for the adjoint based on the @a state and the @a state_bar
   /// \param[in] state - the converged solution that satisfies R(state) = 0
   /// \param[in] state_bar - the derivative of some function w.r.t. the
   /// @a state
   /// \param[out] adjoint - the solution to the equation
   /// \partial R / \partial @a state * adjoint^T = @a state_bar
   void solveForAdjoint(const mfem::Vector &state,
                        const mfem::Vector &state_bar,
                        mfem::Vector &adjoint)
   {
      solveForAdjoint({{"state", state}}, state_bar, adjoint);
   }

   /// Solve for the adjoint based on the @a state and the @a state_bar
   /// \param[in] inputs - scalars and fields that the residual may depend on
   /// that satisfies R(inputs) = 0
   /// \param[in] state_bar - the derivative of some function w.r.t. the
   /// @a state
   /// \param[out] adjoint - the solution to the equation
   /// \partial R / \partial @a state * adjoint^T = @a state_bar
   void solveForAdjoint(const MISOInputs &inputs,
                        const mfem::Vector &state_bar,
                        mfem::Vector &adjoint);

   /// Compute the residual and store the it in @a residual
   /// \param[in] state - the state to evaluate the residual at
   /// \param[out] residual - the discrete residual vector
   void calcResidual(const mfem::Vector &state, mfem::Vector &residual) const;

   /// Compute the residual based on inputs and store the it in @a residual
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \param[out] residual - the discrete residual vector
   void calcResidual(const MISOInputs &inputs, mfem::Vector &residual) const;

   /// Compute the residual norm based on inputs
   /// \param[in] state - the state to evaluate the residual at
   /// \return the norm of the discrete residual vector
   double calcResidualNorm(const mfem::Vector &state) const;

   /// Compute the residual norm based on inputs
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating residual
   /// \return the norm of the discrete residual vector
   double calcResidualNorm(const MISOInputs &inputs) const;

   /// \return the size of the state vector
   int getStateSize() const;

   /// Gets the size of a field @a name known by the solver
   /// \param[in] name - the name of the field to look up the size of
   /// \return the discrete size of the field identified by @a name
   /// \note if the field @a name is unrecognized by the solver, 0 is returned
   virtual int getFieldSize(const std::string &name) const;

   /// Creates a MISOOutput for the specified @a output based on @a options
   /// \param[in] output - specifies the desired output
   /// \note if an output for @a output has already been created an
   /// exception will be thrown
   void createOutput(const std::string &output);

   /// Creates a MISOOutput for the specified @a output based on @a options
   /// \param[in] output - specifies the desired output
   /// \param[in] options - options needed for configuring the output
   /// \note if an output for @a output has already been created an
   /// exception will be thrown
   void createOutput(const std::string &output, const nlohmann::json &options);

   /// Gets the size of the @a output known to the solver
   /// \param[in] output - the name of the output to look up the size of
   /// \return the discrete size of the @a output
   int getOutputSize(const std::string &output);

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
   double calcOutput(const std::string &output, const MISOInputs &inputs);

   /// Evaluates the vector-valued output specifed by @a output
   /// \param[in] output - specifies the desired output
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating the output
   /// \param[out] out_vec - estimated vector-valued output
   void calcOutput(const std::string &output,
                   const MISOInputs &inputs,
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
                          const MISOInputs &inputs,
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
                          const MISOInputs &inputs,
                          mfem::Vector &partial);

   /// Compute an output's sensitivity to @a wrt and contract it with @a wrt_dot
   /// \param[inout] output - the output whose sensitivity we want
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] out_dot - the assembled/contracted sensitivity is
   /// accumulated into out_dot
   void outputJacobianVectorProduct(const std::string &of,
                                    const MISOInputs &inputs,
                                    const mfem::Vector &wrt_dot,
                                    const std::string &wrt,
                                    mfem::Vector &out_dot);

   /// Compute an output's sensitivity to @a wrt and contract it with @a out_bar
   /// \param[inout] output - the output whose sensitivity we want
   /// \param[in] out_bar - the output-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] wrt_bar - the assembled/contracted sensitivity is
   /// accumulated into wrt_bar
   void outputVectorJacobianProduct(const std::string &of,
                                    const MISOInputs &inputs,
                                    const mfem::Vector &out_bar,
                                    const std::string &wrt,
                                    mfem::Vector &wrt_bar);

   /// Cache inputs for the residual and internally store Jacobians
   /// \param[in] inputs - the independent variables at which to evaluate `res`
   void linearize(const MISOInputs &inputs);

   /// Compute the residual's sensitivity to a scalar and contract it with
   /// wrt_dot
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   double jacobianVectorProduct(const mfem::Vector &wrt_dot,
                                const std::string &wrt);

   /// Compute the residual's sensitivity to a vector and contract it with
   /// wrt_dot
   /// \param[in] wrt_dot - the "wrt"-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] res_dot - the assembled/contracted sensitivity is
   /// accumulated into res_dot
   void jacobianVectorProduct(const mfem::Vector &wrt_dot,
                              const std::string &wrt,
                              mfem::Vector &res_dot);

   /// Compute the residual's sensitivity to a scalar and contract it with
   /// res_bar
   /// \param[in] res_bar - the residual-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \return the assembled/contracted sensitivity
   double vectorJacobianProduct(const mfem::Vector &res_bar,
                                const std::string &wrt);

   /// Compute the residual's sensitivity to a vector and contract it with
   /// res_bar
   /// \param[in] res_bar - the residual-sized vector to contract with the
   /// sensitivity
   /// \param[in] wrt - string denoting what variable to take the derivative
   /// with respect to
   /// \param[inout] wrt_bar - the assembled/contracted sensitivity is
   /// accumulated into wrt_bar
   void vectorJacobianProduct(const mfem::Vector &res_bar,
                              const std::string &wrt,
                              mfem::Vector &wrt_bar);

   AbstractSolver2(MPI_Comm incomm, const nlohmann::json &solver_options);

   virtual ~AbstractSolver2() = default;

protected:
   /// communicator used by MPI group for communication
   MPI_Comm comm;

   /// MPI process rank
   int rank;

   /// print object
   std::ostream *out;

   /// storage for algorithmic differentiation (shared by everything in miso)
   adept::Stack &diff_stack;

   /// work vector for solvers
   mutable mfem::Vector work;

   /// solver options
   nlohmann::json options;

   /// residual defines the just the spatial dynamics of a problem
   std::unique_ptr<MISOResidual> spatial_res;
   /// residual defines the dynamics of an ODE (including steady ODEs)
   std::unique_ptr<MISOResidual> space_time_res;

   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> linear_solver;
   /// newton solver for solving implicit problems
   std::unique_ptr<mfem::NewtonSolver> nonlinear_solver;

   /// linear system solver used for adjoint solve
   std::unique_ptr<mfem::Solver> adj_solver;

   /// \brief the ordinary differential equation that describes how to evolve
   /// the state variables
   std::unique_ptr<FirstOrderODE> ode;

   /// map of outputs the solver can compute
   std::map<std::string, MISOOutput> outputs;

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

}  // namespace miso

#endif
