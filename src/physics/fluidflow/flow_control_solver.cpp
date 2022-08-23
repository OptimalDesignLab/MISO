#include <any>

#include "mfem.hpp"

#ifdef MFEM_USE_PUMI
#include "apf.h"
#include "apfMDS.h"
#include "apfMesh.h"
#include "apfNumbering.h"
#include "PCU.h"
#include "crv.h"
#include "gmi_mesh.h"
#include "gmi_null.h"
#ifdef MFEM_USE_SIMMETRIX
#include "SimUtil.h"
#include "gmi_sim.h"
#endif  // MFEM_USE_SIMMETRIX
#ifdef MFEM_USE_EGADS
#include "gmi_egads.h"
#endif  // MFEM_USE_EGADS
#endif  // MFEM_USE_PUMI

#include "diag_mass_integ.hpp"
#include "finite_element_state.hpp"
#include "flow_control_residual.hpp"
#include "mfem_extensions.hpp"
#include "sbp_fe.hpp"
#include "utils.hpp"

#include "flow_control_solver.hpp"

using namespace std;
using namespace mfem;

// namespace
// {
// template <typename T>
// T createFiniteElementVector(mfem::ParMesh &mesh,
//                             const nlohmann::json &space_options,
//                             const int num_states,
//                             const std::string &name)
// {
//    const int dim = mesh.Dimension();
//    const auto order = space_options["degree"].get<int>();
//    const auto basis_type = space_options["basis-type"].get<std::string>();
//    const bool galerkin_diff = space_options.value("GD", false);
//    // Define the SBP elements and finite-element space; eventually, we will
//    want
//    // to have a case or if statement here for both CSBP and DSBP, and (?)
//    // standard FEM. and here it is for first two
//    std::unique_ptr<mfem::FiniteElementCollection> fec;
//    if (basis_type == "csbp")
//    {
//       fec = std::make_unique<mfem::SBPCollection>(order, dim);
//    }
//    else if (basis_type == "dsbp" || galerkin_diff)
//    {
//       fec = std::make_unique<mfem::DSBPCollection>(order, dim);
//    }
//    else if (basis_type == "nedelec" || basis_type == "nd")
//    {
//       fec = std::make_unique<mfem::ND_FECollection>(order, dim);
//    }
//    else if (basis_type == "H1")
//    {
//       fec = std::make_unique<mfem::H1_FECollection>(order, dim);
//    }

//    T vec(mesh,
//          {.order = order,
//           .num_states = num_states,
//           .coll = std::move(fec),
//           .ordering = mfem::Ordering::byVDIM,
//           .name = name});
//    return vec;
// }

// mach::FiniteElementState createState(mfem::ParMesh &mesh,
//                                      const nlohmann::json &space_options,
//                                      const int num_states,
//                                      const std::string &name)
// {
//    return createFiniteElementVector<mach::FiniteElementState>(
//        mesh, space_options, num_states, name);
// }

// mach::FiniteElementDual createDual(mfem::ParMesh &mesh,
//                                    const nlohmann::json &space_options,
//                                    const int num_states,
//                                    const std::string &name)
// {
//    return createFiniteElementVector<mach::FiniteElementDual>(
//        mesh, space_options, num_states, name);
// }

// }  // namespace

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
   fields.emplace("flow_state",
                  FiniteElementState(
                      *mesh_, options["space-dis"], num_states, "flow_state"));
   // We may need the following method from PDESolver eventually
   // setUpExternalFields();

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
           options, fes(), fields, diff_stack, *out));
   auto *mass_matrix = getMassMatrix(*spatial_res, options);
   space_time_res = std::make_unique<mach::MachResidual>(
       mach::TimeDependentResidual(*spatial_res, mass_matrix));

   // get the preconditioner, and construct the linear solver and nonlinear
   // solver
   auto *prec = getPreconditioner(*spatial_res);
   const auto &lin_solver_opts = options["lin-solver"];
   linear_solver = constructLinearSolver(comm, lin_solver_opts, prec);
   const auto &nonlin_solver_opts = options["nonlin-solver"];
   nonlinear_solver =
       constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
   nonlinear_solver->SetOperator(*space_time_res);

   // construct the ODE solver (also used for pseudo-transient continuation)
   const auto &ode_opts = options["time-dis"];
   ode = make_unique<FirstOrderODE>(
       *space_time_res, ode_opts, *nonlinear_solver, out);

   if (options["paraview"].at("each-timestep"))
   {
      ParaViewLogger paraview(options["paraview"]["directory"], mesh_.get());
      paraview.registerField("state", fields.at("flow_state").gridFunc());
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
   AbstractSolver2::setState_(function, name, state);

   useAny(function,
          [&](std::pair<std::function<void(Vector &)>,
                        std::function<void(const Vector &, Vector &)>> &pair)
          {
             auto &control_func = pair.first;
             auto &flow_func = pair.second;
             Vector control_state;
             Vector flow_state;
             extractStates(state, control_state, flow_state);
             if (rank == 0)
             {
                control_func(control_state);
             }
             fields.at("flow_state").project(flow_func, flow_state);
          });
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::initialHook(const mfem::Vector &state)
{
   // The base class method initialHook assumes state is one discipline,
   // so we call it here with flow_state
   Vector control_state;
   Vector flow_state;
   extractStates(state, control_state, flow_state);
   AbstractSolver2::initialHook(flow_state);
   // getState().distributeSharedDofs(state);
   if (options["time-dis"]["steady"])
   {
      throw MachException(
          "FlowControlSolver not set up to handle steady "
          "simulations!\n");
   }
   // Prefer to use loggers eventually
   if (options["outputs"].value("each-timestep", false))
   {
      double t0 = options["time-dis"]["t-initial"];  // Should be passed in!!!
      Vector control_state;
      Vector flow_state;
      extractStates(state, control_state, flow_state);
      *out << "Outputs before time marching:" << endl;
      for (auto &pair : outputs)
      {
         auto inputs = MachInputs({});
         if (pair.first == "entropy")  // need compound state
         {
            inputs = MachInputs({{"time", t0}, {"state", state}});
         }
         else
         {
            inputs = MachInputs({{"time", t0}, {"state", flow_state}});
         }
         double fun = mach::calcOutput(pair.second, inputs);
         *out << "\t" << pair.first << " = " << fun << endl;
         if (rank == 0)
         {
            /// When running batches, this indicates the sample number
            int sample = options.value("sample", 0);
            string output_filename =
                pair.first + "-" + std::to_string(sample) + ".txt";
            remove(output_filename.c_str());
            output_log.emplace(pair.first,
                               ofstream(output_filename, fstream::app));
            // output_log[pair.first].open(output_filenam, fstream::app);
            output_log[pair.first] << setprecision(16);
         }
      }
   }
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::iterationHook(int iter,
                                                   double t,
                                                   double dt,
                                                   const mfem::Vector &state)
{
   // The base class method iterationHook assumes state is one discipline,
   // so we call it here with flow_state
   Vector control_state;
   Vector flow_state;
   extractStates(state, control_state, flow_state);
   AbstractSolver2::iterationHook(iter, t, dt, flow_state);

   if (options["outputs"].value("each-timestep", false))
   {
      extractStates(state, control_state, flow_state);
      for (auto &pair : outputs)
      {
         auto inputs = MachInputs({});
         if (pair.first == "entropy")  // need compound state
         {
            inputs = MachInputs({{"time", t}, {"state", state}});
         }
         else
         {
            inputs = MachInputs({{"time", t}, {"state", flow_state}});
         }
         double fun = mach::calcOutput(pair.second, inputs);
         if (rank == 0)
         {
            output_log[pair.first] << t << ' ' << fun << endl;
         }
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
   if (!options["time-dis"]["const-cfl"].template get<bool>())
   {
      return AbstractSolver2::calcStepSize(iter, t, t_final, dt_old, state);
   }
   // Otherwise, use a constant CFL condition
   auto cfl = options["time-dis"]["cfl"].template get<double>();
   // here we call the FlowResidual method for the min time step, which needs
   // the current flow state as a grid function
   Vector control_state;
   Vector flow_state;
   extractStates(state, control_state, flow_state);
   FiniteElementState &flow_field = fields.at("flow_state");
   flow_field.distributeSharedDofs(flow_state);
   return getConcrete<ResType>(*spatial_res)
       .minCFLTimeStep(cfl, flow_field.gridFunc());
}

template <int dim, bool entvar>
bool FlowControlSolver<dim, entvar>::iterationExit(int iter,
                                                   double t,
                                                   double t_final,
                                                   double dt,
                                                   const Vector &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
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
   // The base class method terminalHook assumes state is one discipline,
   // so we call it here with flow_state
   Vector control_state;
   Vector flow_state;
   extractStates(state, control_state, flow_state);
   AbstractSolver2::terminalHook(iter, t_final, flow_state);

   if (options["outputs"].value("each-timestep", false))
   {
      extractStates(state, control_state, flow_state);
      for (auto &pair : outputs)
      {
         auto inputs = MachInputs({});
         if (pair.first == "entropy")  // need compound state
         {
            inputs = MachInputs({{"time", t_final}, {"state", state}});
         }
         else
         {
            inputs = MachInputs({{"time", t_final}, {"state", flow_state}});
         }
         double fun = mach::calcOutput(pair.second, inputs);
         if (rank == 0)
         {
            output_log[pair.first] << t_final << ' ' << fun << endl;
            output_log[pair.first].close();
         }
      }
   }
}

template <int dim, bool entvar>
void FlowControlSolver<dim, entvar>::addOutput(const std::string &fun,
                                               const nlohmann::json &options)
{
   auto &flow_control_res = getConcrete<ResType>(*spatial_res);
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