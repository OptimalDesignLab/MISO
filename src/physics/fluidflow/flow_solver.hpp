#ifndef MACH_FLOW_SOLVER
#define MACH_FLOW_SOLVER

#include "mfem.hpp"

#include "flow_residual.hpp"
#include "pde_solver.hpp"

namespace mach
{
/// Class for solving FlowResidual based problems
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
/// \note This assumes a constant mass matrix at present; that is, it does not
/// accommodate entropy variables as states with modal or DGD-type
/// discretizations.  However, this could be accommodated by making the mass
/// operator its own MachResidual, and then modifying the
/// `TimeDependentResidual` accordingly.
template <int dim, bool entvar = false>
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

   /// Returns the L2 error between the discrete and exact conservative vars.
   /// \param[in] u_exact - function that defines the exact **state**
   /// \param[in] entry - if >= 0, the L2 error of state `entry` is returned
   /// \returns L2 error
   /// \note The solution given by `u_exact` is for the state, conservative or
   /// entropy variables.  **Do not give the exact solution for the conservative
   /// variables if using entropy variables**.   The conversion to conservative
   /// variables is done by this function.
   double calcConservativeVarsL2Error(void (*u_exact)(const mfem::Vector &,
                                                      mfem::Vector &),
                                      int entry);

private:
   using FlowResType = FlowResidual<dim,entvar>;
   /// Initial residual norm for PTC and convergence checks
   double res_norm0 = -1.0;
   /// Bilinear form for the mass-matrix operator
   mfem::ParBilinearForm mass;
   /// Mass matrix as HypreParMatrix
   std::unique_ptr<mfem::HypreParMatrix> mass_mat;
   /// Solver used for preconditioning Newton linear updates
   std::unique_ptr<mfem::Solver> prec;
   /// used to record the total entropy
   std::ofstream entropy_log;

   /// Construct a preconditioner based on the given options
   /// \param[in] options - options structure that determines preconditioner
   /// \returns unique pointer to the preconditioner object
   std::unique_ptr<mfem::Solver> constructPreconditioner(
       nlohmann::json &_options);

   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   virtual void derivedPDEInitialHook(const mfem::Vector &state) override;

   /// Code that should be executed each time step, before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void derivedPDEIterationHook(int iter,
                                        double t,
                                        double dt,
                                        const mfem::Vector &state) override;

   /// Code that should be executed after time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void derivedPDETerminalHook(int iter,
                                       double t_final,
                                       const mfem::Vector &state) override;

   /// Find the step size based on the options; e.g. for constant CFL or PTC
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt_old - the step size that was just taken
   /// \param[in] state - the current state
   /// \returns dt - the step size appropriate to the problem
   /// \note If "const-cfl" option is invoked, this uses the average spectral
   /// radius to estimate the largest wave speed, and uses the minimum distance
   /// between nodes for the length in the CFL number.
   /// \note If the "steady" option is true, the time step will increase based
   /// on the baseline value of "dt" and the residual norm.
   virtual double calcStepSize(int iter, double t, double t_final,
                               double dt_old,
                               const mfem::Vector &state) const override;

   /// Determines when to exit the time stepping loop
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (after the step)
   /// \param[in] t_final - the final time
   /// \param[in] dt - the step size that was just taken
   /// \param[in] state - the current state
   /// \note If a steady problem is being solved, the "steady-abstol" and 
   /// "steady-reltol" options from "time-dis" to determine convergence.
   virtual bool iterationExit(int iter,
                              double t,
                              double t_final,
                              double dt,
                              const mfem::Vector &state) const override;

   /// Add output @a fun based on @a options
   void addOutput(const std::string &fun,
                  const nlohmann::json &options) override;

};

}  // namespace mach

#endif