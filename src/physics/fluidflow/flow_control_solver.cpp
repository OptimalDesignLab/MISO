#include "diag_mass_integ.hpp"
#include "flow_control_solver.hpp"
#include "flow_control_residual.hpp"
#include "mfem_extensions.hpp"

using namespace std;
using namespace mfem;

/// Return the number of flow state variables
/// \param[in] solver_options - helps define/determine the number of states
/// \param[in] space_dim - the number of spatial dimensions, consistent mesh
/// \todo update for any RANS models, since this assumes no additional states
int getNumFlowStates(const nlohmann::json &solver_options, int space_dim)
{
   return space_dim + 2;
}

namespace mach
{
template <int dim, bool entvar>
FlowControlSolver<dim, entvar>::FlowControlSolver(
    MPI_Comm incomm,
    const nlohmann::json &solver_options,
    std::unique_ptr<mfem::Mesh> smesh)
 : AbstractSolver2(incomm, solver_options),
   mesh_(constructMesh(comm, options["mesh"], std::move(smesh)))
{
   int num_states = dim + 2;
   fields.emplace(
       "flow_state", createState(*mesh_, options["space-dis"], num_states, "state"));
   // We may need the following method from PDESolver eventually
   //setUpExternalFields();

   // Check for consistency between the template parameters, mesh, and options
   if (mesh_->SpaceDimension() != dim)
   {
      throw MachException(
          "FlowControlSolver<dim,entvar> constructor:\n"
          "\tMesh space dimension does not match template"
          "parameter dim");
   }
   bool ent_state = options["flow-param"].value("entropy-state", false);
   if (ent_state != entvar)
   {
      throw MachException(
          "FlowControlSolver<dim,entvar> constructor:\n"
          "\tentropy-state option is inconsistent with entvar"
          "template parameter");
   }
   if ((entvar) && (!options["time-dis"]["steady"]))
   {
      throw MachException(
          "FlowControlSolver<dim,entvar> constructor:\n"
          "\tnot set up for using entropy-variables as states for unsteady "
          "problem (need nonlinear mass-integrator).");
   }

   // Construct spatial residual and the space-time residual
   spatial_res =
       std::make_unique<mach::MachResidual>(FlowControlResidual<dim, entvar>(
           options, fes(), diff_stack, *out));
   auto mass_matrix = getMassMatrix(*spatial_res, options);
   space_time_res = std::make_unique<mach::MachResidual>(
       mach::TimeDependentResidual(*spatial_res, mass_matrix));

   // construct the preconditioner, linear solver, and nonlinear solver
   auto prec_solver_opts = options["lin-prec"];
   auto prec = getPreconditioner(*spatial_res, prec_solver_opts);
   auto lin_solver_opts = options["lin-solver"];
   linear_solver = constructLinearSolver(comm, lin_solver_opts, prec);
   auto nonlin_solver_opts = options["nonlin-solver"];
   nonlinear_solver =
       constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
   nonlinear_solver->SetOperator(*space_time_res);

   // construct the ODE solver (also used for pseudo-transient continuation)
   auto ode_opts = options["time-dis"];
   ode = make_unique<FirstOrderODE>(
       *space_time_res, ode_opts, *nonlinear_solver, out);

   if (options["paraview"].at("each-timestep"))
   {
      ParaViewLogger paraview(options["paraview"]["directory"], mesh_.get());
      paraview.registerField("state", fields.at("state").gridFunc());
      addLogger(std::move(paraview), {.each_timestep = true});
   }
}

template <int dim, bool entvar>
std::unique_ptr<mfem::ParMesh> FlowControlSolver<dim, entvar>::constructMesh(
    MPI_Comm comm,
    const nlohmann::json &mesh_options,
    std::unique_ptr<mfem::Mesh> smesh)
{
   auto mesh_file = mesh_options["file"].get<std::string>();
   std::string mesh_ext;
   std::size_t i = mesh_file.rfind('.', mesh_file.length());
   if (i != std::string::npos)
   {
      mesh_ext = (mesh_file.substr(i + 1, mesh_file.length() - i));
   }
   else
   {
      throw MachException(
          "AbstractSolver::constructMesh(smesh)\n"
          "\tMesh file has no extension!\n");
   }

   std::unique_ptr<mfem::ParMesh> mesh;
   // if serial mesh passed in, use that
   if (smesh != nullptr)
   {
      mesh = std::make_unique<mfem::ParMesh>(comm, *smesh);
   }
   // native MFEM mesh
   else if (mesh_ext == "mesh")
   {
      // read in the serial mesh
      smesh = std::make_unique<mfem::Mesh>(mesh_file.c_str(), 1, 1);
      mesh = std::make_unique<mfem::ParMesh>(comm, *smesh);
   }
   // // PUMI mesh
   // else if (mesh_ext == "smb")
   // {
   //    mesh = constructPumiMesh(comm, mesh_options);
   // }
   mesh->EnsureNodes();

   mesh->RemoveInternalBoundaries();
   // std::cout << "bdr_attr: ";
   // mesh->bdr_attributes.Print(std::cout);
   return mesh;
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::setState_(std::any function,
                                               const std::string &name,
                                               mfem::Vector &state)
{
   // We need a function to separate state into its control and flow states
   // This function could be owned by the Solver or the Residual.

   AbstractSolver2::setState_(function, name, state);

   useAny(
       function,
       [&](std::function<double(const mfem::Vector &)> &fun)
       { fields.at(name).project(fun, state); },
       [&](mfem::Coefficient *coeff)
       { fields.at(name).project(*coeff, state); },
       [&](std::function<void(const mfem::Vector &, mfem::Vector &)> &vec_fun)
       { fields.at(name).project(vec_fun, state); },
       [&](mfem::VectorCoefficient *vec_coeff)
       { fields.at(name).project(*vec_coeff, state); });
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::initialHook(const mfem::Vector &state)
{
   AbstractSolver2::initialHook(state);
   getState().distributeSharedDofs(state);
   if (options["time-dis"]["steady"].template get<bool>())
   {
      throw MachException("FlowControlSolver not set up to handle steady "
                          "simulations!\n");
   }
   if (options["time-dis"]["entropy-log"])
   {
      double t0 = options["time-dis"]["t-initial"];  // Should be passed in!!!
      auto inputs = MachInputs({{"time", t0}, {"state", state}});
      double entropy = calcEntropy(*spatial_res, inputs);
      if (rank == 0)
      {
         *out << "before time stepping, entropy is " << entropy << endl;
         remove("entropy-log.txt");
         entropy_log.open("entropy-log.txt", fstream::app);
         entropy_log << setprecision(16);
      }
   }
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::iterationHook(int iter,
                              double t,
                              double dt,
                              const mfem::Vector &state)
{
   AbstractSolver2::iterationHook(iter, t, dt, state);
   if (options["time-dis"]["entropy-log"])
   {
      auto inputs = MachInputs({{"time", t}, {"state", state}});
      double entropy = calcEntropy(*spatial_res, inputs);
      if (rank == 0)
      {
         entropy_log << t << ' ' << entropy << endl;
      }
   }
}

template <int dim, bool entvar>
double FlowControlSolver<dim, entvar>::calcStepSize(int iter,
                                                    double t,
                                                    double t_final,
                                                    double dt_old,
                                                    const Vector &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      throw MachException(
          "FlowControlSolver not set up to handle steady "
          "simulations!\n");
   }
   if (!options["time-dis"]["const-cfl"].get<bool>())
   {
      return AbstractSolver2::calcStepSize(iter, t, t_final, dt_old, state);
   }
   // Otherwise, use a constant CFL condition
   auto cfl = options["time-dis"]["cfl"].get<double>();
   // here we call the FlowResidual method for the min time step, which needs
   // the current state; this is provided by the state field of PDESolver,
   // which we access with getState()
   return getConcrete<ResType>(*spatial_res)
       .minCFLTimeStep(cfl, getState().gridFunc());
}

template <int dim, bool entvar>
bool FlowControlSolver<dim, entvar>::iterationExit(int iter,
                                                   double t,
                                                   double t_final,
                                                   double dt,
                                                   const Vector &state) const
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      throw MachException(
          "FlowControlSolver not set up to handle steady "
          "simulations!\n");
   }
   else
   {
      return AbstractSolver2::iterationExit(iter, t, t_final, dt, state);
   }
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::terminalHook(int iter,
                             double t_final,
                             const mfem::Vector &state)
{
   AbstractSolver2::terminalHook(iter, t_final, state);
   if (options["time-dis"]["entropy-log"])
   {
      auto inputs = MachInputs({{"time", t_final}, {"state", state}});
      double entropy = calcEntropy(*spatial_res, inputs);
      if (rank == 0)
      {
         entropy_log << t_final << ' ' << entropy << endl;
         entropy_log.close();
      }
   }
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::addOutput(const std::string &fun,
                                               const nlohmann::json &options)
{
   FlowControlResidual<dim, entvar> &flow_control_res =
       getConcrete<FlowControlResidual<dim, entvar>>(*spatial_res);
   outputs.emplace(fun, flow_control_res.constructOutput(fun, options));
}

// explicit instantiation
template class FlowControlSolver<1, true>;
template class FlowControlSolver<1, false>;
template class FlowControlSolver<2, true>;
template class FlowControlSolver<2, false>;
template class FlowControlSolver<3, true>;
template class FlowControlSolver<3, false>;

}  // namespace mach