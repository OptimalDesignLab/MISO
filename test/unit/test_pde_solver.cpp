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
   ThermalResidual(mfem::ParFiniteElementSpace &fes,
                   const nlohmann::json &options)
    : fes_(fes),
      fields(std::make_unique<std::unordered_map<std::string, mfem::ParGridFunction>>()),
      res(fes_, *fields),
      // kappa(std::make_unique<mfem::ConstantCoefficient>(1.0))
      load(std::make_unique<mach::MachLinearForm>(fes_, *fields)),
      force(std::make_unique<mfem::FunctionCoefficient>([](const mfem::Vector &p, double t)
      {
         auto x = p(0);
         return 2 - exp(-t) * (3*sin(2*x) + 8*sin(3*x));
      })),
      prec(constructPreconditioner(fes_.GetComm(), options["lin-prec"]))
   {
      res.addDomainIntegrator(new mfem::DiffusionIntegrator);
      load->addDomainIntegrator(new mfem::DomainLFIntegrator(*force));
      force->SetTime(1e16);
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

   friend mfem::Solver *getPreconditioner(ThermalResidual &residual)
   {
      return residual.prec.get();
   }

private:
   mfem::ParFiniteElementSpace &fes_;
   std::unique_ptr<std::unordered_map<std::string, mfem::ParGridFunction>> fields;
   mach::MachNonlinearForm res;
   // std::unique_ptr<mfem::ConstantCoefficient> kappa;
   std::unique_ptr<mach::MachLinearForm> load;
   std::unique_ptr<mfem::FunctionCoefficient> force;

   /// preconditioner for inverting residual's state Jacobian
   std::unique_ptr<mfem::Solver> prec;
   std::unique_ptr<mfem::Solver> constructPreconditioner(
       MPI_Comm comm,
       const nlohmann::json &prec_options)
   {
      auto amg = std::make_unique<mfem::HypreBoomerAMG>();
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
      return amg;
   }
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
         res = std::make_unique<mach::MachResidual>(ThermalResidual(fes(), options));
      }
      else
      {
         mass.emplace(&fes());
         mass->AddDomainIntegrator(new mfem::MassIntegrator);
         mass->Assemble(0); // keep sparsity pattern of M the same as spatial Jacobian
         mass->Finalize(0);
         mfem::Array<int> ess_bdr(fes().GetParMesh()->bdr_attributes.Max());
         ess_bdr = 0;
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
            mach::TimeDependentResidual(ThermalResidual(fes(), options), &(*mass_mat)));
      }
      setOptions(*res, options);

      auto *prec = getPreconditioner(*res);
      auto lin_solver_opts = options["lin-solver"];
      linear_solver = mach::constructLinearSolver(comm, lin_solver_opts, prec);
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
         "t-final": 1.0,
         "dt": 0.05
      },
      "space-dis": {
         "basis-type": "H1",
         "degree": 2
      },
      "lin-solver": {
         "type": "pcg",
         "reltol": 1e-12,
         "abstol": 0.0,
         "printlevel": 1,
         "maxiter": 500
      },
      "nonlin-solver": {
         "maxiter": 1,
         "printlevel": 3
      },
      "ess-bdr": [2, 4]
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
      std::cout << "terminal solution error = " << error << std::endl;
   }
   REQUIRE(error == Approx(0.0608685).margin(1e-8));
}

TEST_CASE("Testing PDESolver steady heat equation MMS")
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
         "steady": true,
         "type": "steady",
         "t-final": 1e16,
         "dt": 1e16
      },
      "space-dis": {
         "basis-type": "H1",
         "degree": 1
      },
      "lin-solver": {
         "type": "pcg",
         "reltol": 1e-12,
         "abstol": 0.0,
         "printlevel": 1,
         "maxiter": 500
      },
      "nonlin-solver": {
         "maxiter": 1,
         "printlevel": 2
      },
      "ess-bdr": [2, 4]
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
      return pow(x, 2);
   });
   // state.project(exact_sol);
   ConstantCoefficient one(1.0);
   state.gridFunc().ProjectCoefficient(one);
   mfem::Array<int> bdr_attr(4);
   bdr_attr = 1; bdr_attr[1] = 1; bdr_attr[3] = 1;
   state.gridFunc().ProjectBdrCoefficient(exact_sol, bdr_attr);
   state.initializeTrueVec();
   MachInputs inputs;
   solver.solveForState(inputs, state.trueVec());
   state.distributeSharedDofs();

   // Check that solution is reasonable accurate
   auto tfinal = options["time-dis"]["t-final"].get<double>();
   exact_sol.SetTime(tfinal);
   auto error = state.gridFunc().ComputeLpError(2, exact_sol);

   if (verbose)
   {
      std::cout << "terminal solution error = " << error << std::endl;
   }
   REQUIRE(error == Approx(0.353809).margin(1e-8));
}