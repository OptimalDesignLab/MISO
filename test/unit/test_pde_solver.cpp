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
   : res(fes, fields),
     kappa(-1.0)
   {
      res.addDomainIntegrator(new mfem::DiffusionIntegrator(kappa));

      mfem::Array<int> ess_bdr(fes.GetParMesh()->bdr_attributes.Max());
      ess_bdr = 1;
      nlohmann::json ess_bdr_opts;
      ess_bdr_opts["ess_bdr"] = {1, 2, 3, 4};
      setOptions(res, ess_bdr_opts);
   }

   friend int getSize(const ThermalResidual &residual)
   {
      return getSize(residual.res);
   }

   friend void evaluate(ThermalResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      evaluate(residual.res, inputs, res_vec);
   }

   friend mfem::Operator &getJacobian(ThermalResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      return getJacobian(residual.res, inputs, std::move(wrt));
   }

private:
   std::unordered_map<std::string, mfem::ParGridFunction> fields;
   mach::MachNonlinearForm res;
   mfem::ConstantCoefficient kappa;
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
      auto fes = state.space();

      
      // res = std::make_unique<mach::MachResidual>(ThermalResidual(fes));
      res = std::make_unique<mach::MachResidual>(mach::TimeDependentResidual(ThermalResidual(fes)));

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

   mfem::ParBilinearForm mass;

   std::unique_ptr<mfem::Solver> constructPreconditioner(
       MPI_Comm comm,
       const nlohmann::json &prec_options)
   {
      auto amg = std::make_unique<mfem::HypreBoomerAMG>();
      amg->SetPrintLevel(prec_options["printlevel"].get<int>());
      return amg;
   }
};

TEST_CASE("Testing AbstractSolver as TimeDependentOperator with RK4",
          "[abstract-solver]")
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
         "type": "MIDPOINT",
         "t-final": 5.0,
         "dt": 0.05
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
      }
   })"_json;

   constexpr int nxy = 4;
   auto mesh = std::make_unique<mfem::Mesh>(
      Mesh::MakeCartesian2D(nxy, nxy, Element::TRIANGLE));

   // Create solver and solve for the state 
   ThermalSolver solver(MPI_COMM_WORLD, options, std::move(mesh));
   Vector u(2);
   u(0) = 1.0;
   u(1) = 0.5;
   MachInputs inputs;
   solver.solveForState(inputs, u);

   // Check that solution is reasonable accurate
   auto exact_sol = [](double t, Vector &u)
   {
      const double e = std::exp(1.0);
      const double sepe = sqrt(e) + e;
      u.SetSize(2);
      u(0) = log(e + pow(e,1.5)) - log(sqrt(e) + exp(sepe*t));
      u(1) = log((sepe*exp(sepe*t))/(sqrt(e) + exp(sepe*t)));
   };
   Vector u_exact;
   exact_sol(options["time-dis"]["t-final"].get<double>(), u_exact);
   double error = sqrt( pow(u(0) - u_exact(0),2) + pow(u(1) - u_exact(1),2));

   if (verbose)
   {
      std::cout << "discrete solution = " << u(0) << ": " << u(1) << std::endl;
      std::cout << "exact solution    = " << u_exact(0) << ": " << u_exact(1)
                << std::endl;
      std::cout << "terminal solution error = " << error << std::endl;
   }
   REQUIRE( error == Approx(1.86013e-05).margin(1e-8) );
}