#ifndef MACH_ABSTRACT_SOLVER
#define MACH_ABSTRACT_SOLVER

#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach_input.hpp"
#include "mach_output.hpp"
#include "mach_residual.hpp"

namespace mach
{

/// Serves as a base class for specific solvers
/// \todo Rename to AbstractSolver once we have old AbstractSolver inherit it
class AbstractSolver2 : public mfem::TimeDependentOperator
{
public:

   /// Base Class constructor
   /// \param[in] options - pre-loaded JSON options object
   /// \param[in] incomm - MPI communicator for parallelized problems
   AbstractSolver2(const nlohmann::json &options,
                   MPI_Comm incomm = MPI_COMM_WORLD);

   /// Execute set-up steps that are specific to the derived class.
   /// \note Use this to define the problem specific residual `res`.
   virtual void initDerived() = 0;

   /// explicitly prohibit copy/move construction
   AbstractSolver2(const AbstractSolver2 &) = delete;
   AbstractSolver2 &operator=(const AbstractSolver2 &) = delete;
   AbstractSolver2(AbstractSolver2 &&) = delete;
   AbstractSolver2 &operator=(AbstractSolver2 &&) = delete;
   
   /// class destructor
   virtual ~AbstractSolver2();

   /// return the options dictionary with read-only access
   inline const nlohmann::json &getOptions() const { return options; }

   /// Solve for the state based on the residual `res` and `options`
   /// \param[in] inputs - scalars and fields that the `res` may depend on 
   /// \param[out] state - the solution to the governing equation
   virtual void solveForState(const MachInputs &inputs, mfem::Vector &state);

/* 
   /// Creates the nonlinear form for the functional
   /// \param[in] fun - specifies the desired functional
   /// \note if a nonlinear form for `fun` has already been created an
   /// exception will be thrown
   void createOutput(const std::string &fun);

   /// Creates the nonlinear form for the functional
   /// \param[in] fun - specifies the desired functional
   /// \param[in] options - options needed for calculating functional
   /// \note if a nonlinear form for `fun` has already been created an
   /// exception will be thrown
   void createOutput(const std::string &fun, const nlohmann::json &options);

   /// Sets options for the output functional specifed by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \param[in] options - options needed for calculating functional
   /// \note will only have an effect if a subclass supports setting options
   ///       for the functional
   void setOutputOptions(const std::string &fun, const nlohmann::json &options);

   /// Evaluates and returns the output functional specifed by `fun`
   /// \param[in] fun - specifies the desired functional
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \return scalar value of estimated functional value
   double calcOutput(const std::string &fun, const MachInputs &inputs);

   /// Evaluates and returns the partial derivative of output functional
   /// specifed by `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired functional
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \param[out] partial - the partial with respect to a scalar-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          double &partial);

   /// Evaluates and returns the partial derivative of output functional
   /// specifed by `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired functional
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \param[out] partial - the partial with respect to a vector-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          double *partial);

   /// Evaluates and returns the partial derivative of output functional
   /// specifed by `of` with respect to the input specified by `wrt`
   /// \param[in] of - specifies the desired functional
   /// \param[in] wrt - specifies the input to differentiate with respect to
   /// \param[in] inputs - collection of field or scalar inputs to set before
   ///                     evaluating functional
   /// \param[out] partial - the partial with respect to a vector-valued input
   void calcOutputPartial(const std::string &of,
                          const std::string &wrt,
                          const MachInputs &inputs,
                          mfem::HypreParVector &partial);
 */

   /// Compute `k = dxdt = M^-1(R(x,t) + Kx + l)` for explicit methods
   /// \param[in] x - state at which to evaluate the dynamics/residual 
   /// \param[out] k - inverse mass matrix acting on the residual
   void Mult(const mfem::Vector &x, mfem::Vector &k) const override;

protected:
   /// communicator used by MPI group for communication
   MPI_Comm comm;
   /// process rank
   int rank;
   /// output-stream object (this is not owned, so don't delete)
   std::ostream *out;
   /// solver options
   nlohmann::json options;
   /// residual defines the dynamics of an ODE (including steady ODEs)
   std::unique_ptr<MachResidual> res;
   /// storage for algorithmic differentiation (shared by all solvers)
   static adept::Stack diff_stack;
   /// linear system preconditioner for solver in newton solver and adjoint
   std::unique_ptr<mfem::Solver> prec;
   /// linear system solver used in newton solver
   std::unique_ptr<mfem::Solver> linear_solver;
   /// newton solver for solving implicit problems 
   std::unique_ptr<mfem::NewtonSolver> newton_solver;
   /// time-marching method (might be nullptr)
   std::unique_ptr<mfem::ODESolver> ode_solver;
   /// map of outputs
   std::map<std::string, MachOutput> outputs;

   /// Returns the size of the state vector (locally)
   /// \todo Should we have a local and Global version of this?
   int getNumState() { return getSize(*res); }

   /// Add functional `fun` based on options
   virtual void addOutput(const std::string &fun, const nlohmann::json &options)
   { }

   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   virtual void initialHook(const mfem::Vector &state) { }

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::Vector &state) { }

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
                               const mfem::ParGridFunction &state) const;

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
                             const mfem::Vector &state) { }

private:
   /// Used to do the bulk of the initialization shared between constructors
   /// \param[in] options - pre-loaded JSON options object
   /// \param[in] comm - MPI communicator to use for parallel operations
   void initBase(const nlohmann::json &file_options, MPI_Comm comm);
}

/// \todo Change this to `SolverPtr` eventually 
using SolverPtr2 = std::unique_ptr<AbstractSolver2>;

/// Creates a new `DerivedSolver` and initializes it 
/// \param[in] json_options - json object that stores options
/// \param[in] comm - MPI communicator for parallel operations
/// \tparam DerivedSolver - a derived class of `AbstractSolver`
template <class DerivedSolver>
SolverPtr2 createSolver(const nlohmann::json &json_options,
                       MPI_Comm comm = MPI_COMM_WORLD)
{
   SolverPtr solver(new DerivedSolver(json_options, comm));
   solver->initDerived();
   return solver;
}

/// Creates a new `DerivedSolver` and initializes it
/// \param[in] json_options - json object that stores options
/// \param[in] smesh - if provided, defines the mesh for the problem
/// \param[in] comm - MPI communicator for parallel operations
/// \tparam DerivedSolver - a derived class of `AbstractSolver`
template <class DerivedSolver>
SolverPtr2 createSolver(const nlohmann::json &json_options,
                       std::unique_ptr<mfem::Mesh> smesh,
                       MPI_Comm comm = MPI_COMM_WORLD)
{
   // auto solver = std::make_unique<DerivedSolver>(opt_file_name, move(smesh));
   SolverPtr solver(new DerivedSolver(json_options, move(smesh), comm));
   solver->initDerived();
   return solver;
}

/// Creates a new `DerivedSolver` and initializes it
/// \param[in] opt_file_name - file where options are stored
/// \param[in] smesh - if provided, defines the mesh for the problem
/// \param[in] comm - MPI communicator for parallel operations
/// \tparam DerivedSolver - a derived class of `AbstractSolver`
template <class DerivedSolver>
SolverPtr2 createSolver(const std::string &opt_file_name,
                       std::unique_ptr<mfem::Mesh> smesh,
                       MPI_Comm comm = MPI_COMM_WORLD)
{
   nlohmann::json json_options;
   std::ifstream options_file(opt_file_name);
   options_file >> json_options;
   return createSolver<DerivedSolver>(json_options, move(smesh), comm);
}

} // namespace mach
#endif 