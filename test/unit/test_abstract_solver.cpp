#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "mach_residual.hpp"
#include "matrix_operators.hpp"
#include "mfem_extensions.hpp"
#include "utils.hpp"

/// Class for ODE that follows the MachResidual API
class ExpODEResidual final
{
public:
   ExpODEResidual() : work(2), Jac(*this) {}

   friend int getSize(const ExpODEResidual &residual) { return 2; }

   friend void evaluate(ExpODEResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      mfem::Vector x(inputs.at("state").getField(), 2);
      res_vec.SetSize(2);
      res_vec(0) = exp(x(1));
      res_vec(1) = -exp(x(0));
   }
   friend mfem::Operator &getJacobian(ExpODEResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      mfem::Vector x(inputs.at("state").getField(), 2);
      //residual.Jac(0,0) = 0.0;
      //residual.Jac(0,1) = exp(x(1));
      //residual.Jac(1,0) = -exp(x(0));
      //residual.Jac(1,1) = 0.0;
      //return residual.Jac;
      residual.Jac.setState(x);
      return residual.Jac;
   }
   friend double calcEntropy(ExpODEResidual &residual,
                             const mach::MachInputs &inputs)
   {
      mfem::Vector x(inputs.at("state").getField(), 2);
      return exp(x(0)) + exp(x(1));
   }
   friend double calcEntropyChange(ExpODEResidual &residual,
                                   const mach::MachInputs &inputs)
   {
      mfem::Vector x(inputs.at("state").getField(), 2);
      mfem::Vector x_dot(inputs.at("state_dot").getField(), 2);
      double dt = inputs.at("dt").getValue();
      mfem::Vector &y = residual.work;
      add(x, dt, x_dot, y);
      // should be zero 
      return exp(y(0))*exp(y(1)) - exp(y(1))*exp(y(0)); 
   }
private:
   //mfem::DenseMatrix Jac;
   mach::JacobianFree<ExpODEResidual> Jac;
   mfem::Vector work; 
};

/// Solver that uses `ExpODEResidual` to define its dynamics
class ExponentialODESolver : public mach::AbstractSolver2
{
public:
   ExponentialODESolver(MPI_Comm comm, const nlohmann::json &solver_options)
      : AbstractSolver2(comm, solver_options)
   {
      // res = std::make_unique<mach::MachResidual>(ExpODEResidual());
      res = std::make_unique<mach::MachResidual>(mach::TimeDependentResidual(ExpODEResidual()));

      auto lin_solver_opts = options["lin-solver"];
      linear_solver = mach::constructLinearSolver(comm, lin_solver_opts);
      auto nonlin_solver_opts = options["nonlin-solver"];
      nonlinear_solver = mach::constructNonlinearSolver(comm, nonlin_solver_opts, *linear_solver);
      nonlinear_solver->SetOperator(*res);

      auto ode_opts = options["time-dis"];
      ode = std::make_unique<mach::FirstOrderODE>(*res, ode_opts, *nonlinear_solver);
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
         "type": "RK4",
         "t-final": 5.0,
         "dt": 0.05
      },
      "lin-solver": {
         "type": "pcg",
         "reltol": 1e-14,
         "abstol": 0.0,
         "printlevel": -1,
         "maxiter": 500
      },
      "nonlin-solver": {
         "maxiter": 1,
         "printlevel": -1
      }
   })"_json;

   // Create solver and solve for the state 
   ExponentialODESolver solver(MPI_COMM_WORLD, options);
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

TEST_CASE("Testing AbstractSolver as TimeDependentOperator with RRK",
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
         "type": "RRK",
         "t-final": 5.0,
         "dt": 0.05
      },
      "lin-solver": {
         "type": "gmres",
         "reltol": 1e-14,
         "abstol": 0.0,
         "printlevel": -1,
         "maxiter": 500
      },
      "nonlin-solver": {
         "maxiter": 10,
         "printlevel": -1
      }
   })"_json;

   // Create solver and solve for the state 
   ExponentialODESolver solver(MPI_COMM_WORLD, options);
   Vector u0(2), u(2);
   u0(0) = 1.0;
   u0(1) = 0.5;
   u = u0;
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
   double entropy0 = exp(u0(0)) + exp(u0(1));
   double entropy = exp(u(0)) + exp(u(1));

   if (verbose)
   {
      std::cout << "discrete solution = " << u(0) << ": " << u(1) << std::endl;
      std::cout << "exact solution    = " << u_exact(0) << ": " << u_exact(1)
                << std::endl;
      std::cout << "terminal solution error = " << error << std::endl;
      std::cout << "entropy error = " << entropy - entropy0 << std::endl;
   }
   REQUIRE( error == Approx(0.003).margin(1e-4) );

   REQUIRE( entropy == Approx(entropy0).margin(1e-12) );
}