#include <memory>
#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "mach_linearform.hpp"
#include "mach_load.hpp"
#include "mach_nonlinearform.hpp"
#include "mach_residual.hpp"
#include "mfem_extensions.hpp"
#include "pde_solver.hpp"
#include "utils.hpp"

/// Class for ODE that follows the MachResidual API
class ThermalResidual final
{
public:
   ThermalResidual(mfem::ParFiniteElementSpace &fes)
    : fes_(fes),
      fields(std::make_unique<std::unordered_map<std::string, mfem::ParGridFunction>>()),
      res(fes_, *fields),
      kappa(std::make_unique<mfem::ConstantCoefficient>(-1.0)),
      load(std::make_unique<mach::MachLinearForm>(fes_, *fields)),
      force(std::make_unique<mfem::FunctionCoefficient>([](const mfem::Vector &p, double t)
      {
         auto x = p(0);
         return 2 - exp(-t) * (3*sin(2*x) + 8*sin(3*x));
      }))
   {
      res.addDomainIntegrator(new mfem::DiffusionIntegrator(*kappa));

      // mfem::Array<int> ess_bdr(fes_.GetParMesh()->bdr_attributes.Max());
      // ess_bdr = 1;
      // nlohmann::json ess_bdr_opts;
      // ess_bdr_opts["ess-bdr"] = {1, 2, 3, 4};
      // setOptions(res, options);

      load->addDomainIntegrator(new mfem::DomainLFIntegrator(*force));
   }

   friend int getSize(const ThermalResidual &residual)
   {
      return getSize(residual.res);
   }

   friend void setInputs(ThermalResidual &residual, const mach::MachInputs &inputs)
   {
      setInputs(residual.res, inputs);
      setInputs(*residual.load, inputs);
      auto input = inputs.find("time");
      if (input != inputs.end())
      {
         residual.force->SetTime(input->second.getValue());
      }
   }

   friend void setOptions(ThermalResidual &residual, const nlohmann::json &options)
   {
      setOptions(residual.res, options);
      setOptions(*residual.load, options);
   }

   friend void evaluate(ThermalResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      evaluate(residual.res, inputs, res_vec);
      setInputs(*residual.load, inputs);
      addLoad(*residual.load, res_vec);
   }

   friend mfem::Operator &getJacobian(ThermalResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      return getJacobian(residual.res, inputs, std::move(wrt));
   }

private:
   mfem::ParFiniteElementSpace &fes_;
   std::unique_ptr<std::unordered_map<std::string, mfem::ParGridFunction>> fields;
   mach::MachNonlinearForm res;
   std::unique_ptr<mfem::ConstantCoefficient> kappa;
   std::unique_ptr<mach::MachLinearForm> load;
   std::unique_ptr<mfem::FunctionCoefficient> force;
};

/// Solver that uses `ThermalResidual` to define its dynamics
class ThermalSolver : public mach::PDESolver
{
public:
   ThermalSolver(MPI_Comm comm,
                 const nlohmann::json &solver_options,
                 std::unique_ptr<mfem::Mesh> smesh)
      : PDESolver(comm, solver_options, num_states, std::move(smesh))
   {
      if (options["time-dis"]["steady"].get<bool>())
      {
         res = std::make_unique<mach::MachResidual>(ThermalResidual(fes()));
      }
      else
      {
         mass.emplace(&fes());
         mass->AddDomainIntegrator(new mfem::MassIntegrator);
         mass->Assemble(0); // keep sparsity pattern of M the same as spatial Jacobian
         mass->Finalize(0);
         mfem::Array<int> ess_bdr(fes().GetParMesh()->bdr_attributes.Max());
         auto tmp = options["ess-bdr"].get<std::vector<int>>();
         for (auto &bdr : tmp)
         {
            ess_bdr[bdr - 1] = 1;
         }
         mfem::Array<int> ess_tdof_list;
         fes().GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
         mass_mat.emplace();
         mass->FormSystemMatrix(ess_tdof_list, *mass_mat);

         res = std::make_unique<mach::MachResidual>(
            mach::TimeDependentResidual(ThermalResidual(fes()), &(*mass_mat)));
      }
      setOptions(*res, options);

      auto prec_opts = options["lin-prec"];
      prec = constructPreconditioner(comm, prec_opts);
      auto lin_solver_opts = options["lin-solver"];
      linear_solver = mach::constructLinearSolver(comm, lin_solver_opts);
      auto nonlin_solver_opts = options["nonlin-solver"];
      nonlinear_solver = mach::constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
      nonlinear_solver->SetOperator(*res);

      auto ode_opts = options["time-dis"];
      ode = std::make_unique<mach::FirstOrderODE>(*res, ode_opts, *nonlinear_solver);
   }
private:
   static constexpr int num_states = 1;

   /// Optional mass matrix depending on if the problem is steady or not
   std::optional<mfem::ParBilinearForm> mass = std::nullopt;
   std::optional<mfem::HypreParMatrix> mass_mat = std::nullopt;

   std::unique_ptr<mfem::Solver> constructPreconditioner(
       MPI_Comm comm,
       const nlohmann::json &prec_options)
   {
      auto amg = std::make_unique<mfem::HypreBoomerAMG>();
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
      return amg;
   }
};

TEST_CASE("Testing PDESolver unsteady heat equation MMS")
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;
   using namespace mach;

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "print-options": true,
      "time-dis": {
         "steady": false,
         "type": "MIDPOINT",
         "t-final": 5.0,
         "dt": 0.05
      },
      "space-dis": {
         "basis-type": "H1",
         "degree": 1
      },
      "lin-solver": {
         "type": "pcg",
         "reltol": 1e-12,
         "abstol": 0.0,
         "printlevel": -1,
         "maxiter": 500
      },
      "nonlin-solver": {
         "maxiter": 1,
         "printlevel": -1
      },
      "ess-bdr": [1, 3]
   })"_json;

   constexpr int nxy = 4;
   auto mesh = std::make_unique<mfem::Mesh>(
      Mesh::MakeCartesian2D(nxy, nxy, Element::TRIANGLE, true, M_PI, M_PI));

   // Create solver and solve for the state 
   ThermalSolver solver(MPI_COMM_WORLD, options, std::move(mesh));
   auto &state = solver.getState();

   FunctionCoefficient exact_sol([](const mfem::Vector &p, double t)
   {
      auto x = p(0);
      return pow(x, 2) + exp(-t) * (sin(x) + sin(2*x) + sin(3*x));
   });
   exact_sol.SetTime(0.0);
   state.project(exact_sol);
   MachInputs inputs;
   solver.solveForState(inputs, state.trueVec());
   state.distributeSharedDofs();

   // Check that solution is reasonable accurate
   auto tfinal = options["time-dis"]["t-final"].get<double>();
   exact_sol.SetTime(tfinal);
   auto error = state.gridFunc().ComputeLpError(2, exact_sol);

   if (verbose)
   {
      // std::cout << "discrete solution = " << u(0) << ": " << u(1) << std::endl;
      // std::cout << "exact solution    = " << u_exact(0) << ": " << u_exact(1)
      //           << std::endl;
      std::cout << "terminal solution error = " << error << std::endl;
   }
   // REQUIRE( error == Approx(1.86013e-05).margin(1e-8) );
}