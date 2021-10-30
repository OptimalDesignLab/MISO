#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "mach_residual.hpp"
#include "matrix_operators.hpp"
#include "mfem_extensions.hpp"
#include "utils.hpp"

using std::cout;
using std::endl;

/// Class for ODE that follows the MachResidual API
class ExpODEResidual final
{
public:
   ExpODEResidual() : work(2), Jac(2) {}

   friend int getSize(const ExpODEResidual &residual) { return 2; }

   friend void setInputs(ExpODEResidual &residual,
                         const mach::MachInputs &inputs)
   {
      mach::setValueFromInputs(inputs, "dt", residual.dt);
      mach::setVectorFromInputs(inputs, "state", residual.state);
   }
   friend void evaluate(ExpODEResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      mfem::Vector dxdt;
      mach::setVectorFromInputs(inputs, "state", dxdt);
      res_vec.SetSize(2);
      res_vec(0) = dxdt(0);
      res_vec(1) = dxdt(1);
      if (fabs(residual.dt) < 1e-15)
      {
         // Explicit time marching; x = residual.state
         auto &x = residual.state;
         res_vec(0) +=  exp(x(1));
         res_vec(1) += -exp(x(0));
      }
      else 
      {
         // Implicit time marching; x = residual.state + dt*dxdt
         auto &x = residual.work;
         add(residual.state, residual.dt, dxdt, x);
         res_vec(0) +=  exp(x(1));
         res_vec(1) += -exp(x(0));
      }
   }
   friend mfem::Operator &getJacobian(ExpODEResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      using std::cout;
      using std::endl;
      if (fabs(residual.dt) < 1e-15)
      {
         // Explicit time marching; Jacobian w.r.t. dxdt is identity 
         residual.Jac = 0.0;
         residual.Jac(0,0) = 1.0;
         residual.Jac(1,1) = 1.0;
         return residual.Jac;
      }
      // Implicit time marching
      mfem::Vector dxdt;
      mach::setVectorFromInputs(inputs, "state", dxdt);
      auto &x = residual.work;
      add(residual.state, residual.dt, dxdt, x);
      residual.Jac(0,0) = 1.0;
      residual.Jac(0,1) = residual.dt*exp(x(1));
      residual.Jac(1,0) = residual.dt*exp(x(0));
      residual.Jac(1,1) = 1.0;
      return residual.Jac;
   }
   friend double calcEntropy(ExpODEResidual &residual,
                             const mach::MachInputs &inputs)
   {
      mfem::Vector x;
      mach::setVectorFromInputs(inputs, "state", x, false, true);
      return exp(x(0)) + exp(x(1));
   }
   friend double calcEntropyChange(ExpODEResidual &residual,
                                   const mach::MachInputs &inputs)
   {
      mfem::Vector x, dxdt;
      mach::setVectorFromInputs(inputs, "state", x, false, true);
      mach::setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
      double dt;
      mach::setValueFromInputs(inputs, "dt", dt, true);
      auto &y = residual.work;
      add(x, dt, dxdt, y);
      // should be zero 
      return exp(y(0))*exp(y(1)) - exp(y(1))*exp(y(0)); 
   }
private:
   double dt = NAN;
   mfem::DenseMatrix Jac;
   mfem::Vector work;
   mfem::Vector state; 
};

/// Solver that uses `ExpODEResidual` to define its dynamics
class ExponentialODESolver : public mach::AbstractSolver2
{
public:
   ExponentialODESolver(MPI_Comm comm, const nlohmann::json &solver_options)
      : AbstractSolver2(comm, solver_options)
   {
      // res = std::make_unique<mach::MachResidual>(ExpODEResidual());
      res = std::make_unique<mach::MachResidual>(ExpODEResidual());

      auto lin_solver_opts = options["lin-solver"];
      linear_solver = mach::constructLinearSolver(comm, lin_solver_opts);
      auto nonlin_solver_opts = options["nonlin-solver"];
      nonlinear_solver = mach::constructNonlinearSolver(
         comm, nonlin_solver_opts, *linear_solver);
      nonlinear_solver->SetOperator(*res);

      auto ode_opts = options["time-dis"];
      ode = std::make_unique<mach::FirstOrderODE>(*res, ode_opts, 
                                                  *nonlinear_solver);
   }
};

TEST_CASE("Testing AbstractSolver using RK4", "[abstract-solver]")
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
         "reltol": 1e-12,
         "abstol": 1e-14,
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

TEST_CASE("Testing AbstractSolver using RRK", "[abstract-solver]")
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