#ifndef MACH_FLOW_CONTROL_SOLVER
#define MACH_FLOW_CONTROL_SOLVER

#include "mfem.hpp"

#include "flow_control_residual.hpp"
#include "pde_solver.hpp"

namespace mach
{
/// Class for solving FlowControlResidual based problems
/// \tparam dim - number of spatial dimensions (1, 2, or 3)
/// \tparam entvar - if true, the entropy variables are used in the integrators
template <int dim, bool entvar = false>
class FlowControlSolver : public AbstractSolver2
{
public:
   /// Construct a `FlowSolver` based on the given options and mesh
   /// \param[in] incomm - the MPI communicator to associate with this solver
   /// \param[in] solver_options - options used to define the solver
   /// \param[in] smesh - serial mesh; if not `nullptr`, moved to this solver
   FlowControlSolver(MPI_Comm incomm,
                     const nlohmann::json &solver_options,
                     std::unique_ptr<mfem::Mesh> smesh = nullptr);

private:
   using ResType = FlowControlResidual<dim, entvar>;
   /// object defining the mfem computational mesh
   std::unique_ptr<mfem::ParMesh> mesh_ = nullptr;
   /// used to record the total entropy
   std::ofstream entropy_log;
   /// Map of all grid-function-based state vectors used by the solver
   std::map<std::string, FiniteElementState> fields;

   /// Reference to flow solver state vector
   FiniteElementState &flowState() { return fields.at("flow_state"); }
   const FiniteElementState &flowState() const
   {
      return fields.at("flow_state");
   }

   /// Reference to the flow-state vector's finite element space
   mfem::ParFiniteElementSpace &fes() { return flowState().space(); }
   const mfem::ParFiniteElementSpace &fes() const
   {
      return flowState().space();
   }

   /// Constructs the mesh member based on c preprocesor defs
   /// \param[in] smesh - if provided, defines the mesh for the problem
   std::unique_ptr<mfem::ParMesh> constructMesh(
       MPI_Comm comm,
       const nlohmann::json &mesh_options,
       std::unique_ptr<mfem::Mesh> smesh);

   /// Add output @a fun based on @a options
   void addOutput(const std::string &fun,
                  const nlohmann::json &options) override;

   /// For code that should be executed before the time stepping begins
   /// \param[in] state - the current state
   virtual void initialHook(const mfem::Vector &state) override final;

   /// For code that should be executed before `ode_solver->Step`
   /// \param[in] iter - the current iteration
   /// \param[in] t - the current time (before the step)
   /// \param[in] dt - the step size that will be taken
   /// \param[in] state - the current state
   virtual void iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::Vector &state) override final;

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
   virtual double calcStepSize(int iter,
                               double t,
                               double t_final,
                               double dt_old,
                               const mfem::Vector &state) const override final;

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
                              const mfem::Vector &state) const override final;

   /// For code that should be executed after the time stepping ends
   /// \param[in] iter - the terminal iteration
   /// \param[in] t_final - the final time
   /// \param[in] state - the current state
   virtual void terminalHook(int iter,
                             double t_final,
                             const mfem::Vector &state) override final;
};

} // namespace mach

#endif