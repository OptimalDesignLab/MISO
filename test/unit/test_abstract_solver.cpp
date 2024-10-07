#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "abstract_solver.hpp"
#include "miso_input.hpp"
#include "miso_residual.hpp"
#include "matrix_operators.hpp"
#include "mfem_extensions.hpp"
#include "utils.hpp"

using std::cout;
using std::endl;

/// Class for ODE that follows the MISOResidual API
class ExpODEResidual final
{
public:
   ExpODEResidual() : work(2), Jac(2) {}

   friend int getSize(const ExpODEResidual &residual) { return 2; }

   friend void setInputs(ExpODEResidual &residual,
                         const miso::MISOInputs &inputs)
   {
      miso::setValueFromInputs(inputs, "dt", residual.dt);
      miso::setVectorFromInputs(inputs, "state", residual.state);
   }
   friend void evaluate(ExpODEResidual &residual,
                        const miso::MISOInputs &inputs,
                        mfem::Vector &res_vec)
   {
      mfem::Vector dxdt;
      miso::setVectorFromInputs(inputs, "state", dxdt);
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
                                      const miso::MISOInputs &inputs,
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
      miso::setVectorFromInputs(inputs, "state", dxdt);
      auto &x = residual.work;
      add(residual.state, residual.dt, dxdt, x);
      residual.Jac(0,0) = 1.0;
      residual.Jac(0,1) = residual.dt*exp(x(1));
      residual.Jac(1,0) = residual.dt*exp(x(0));
      residual.Jac(1,1) = 1.0;
      return residual.Jac;
   }
   friend double calcEntropy(ExpODEResidual &residual,
                             const miso::MISOInputs &inputs)
   {
      mfem::Vector x;
      miso::setVectorFromInputs(inputs, "state", x, false, true);
      return exp(x(0)) + exp(x(1));
   }
   friend double calcEntropyChange(ExpODEResidual &residual,
                                   const miso::MISOInputs &inputs)
   {
      mfem::Vector x, dxdt;
      miso::setVectorFromInputs(inputs, "state", x, false, true);
      miso::setVectorFromInputs(inputs, "state_dot", dxdt, false, true);
      double dt;
      miso::setValueFromInputs(inputs, "dt", dt, true);
      //auto &y = residual.work;
      //add(x, dt, dxdt, y);
      // should be zero 
      return exp(x(0))*dxdt(0) + exp(x(1))*dxdt(1);
      //return exp(y(0))*exp(y(1)) - exp(y(1))*exp(y(0)); 
   }
private:
   double dt = NAN;
   mfem::DenseMatrix Jac;
   mfem::Vector work;
   mfem::Vector state; 
};

/// Solver that uses `ExpODEResidual` to define its dynamics
class ExponentialODESolver : public miso::AbstractSolver2
{
public:
   ExponentialODESolver(MPI_Comm comm, const nlohmann::json &solver_options)
      : AbstractSolver2(comm, solver_options)
   {
      space_time_res = std::make_unique<miso::MISOResidual>(ExpODEResidual());

      auto lin_solver_opts = options["lin-solver"];
      linear_solver = miso::constructLinearSolver(comm, lin_solver_opts);
      auto nonlin_solver_opts = options["nonlin-solver"];
      nonlinear_solver = miso::constructNonlinearSolver(
         comm, nonlin_solver_opts, *linear_solver);
      nonlinear_solver->SetOperator(*space_time_res);

      auto ode_opts = options["time-dis"];
      ode = std::make_unique<miso::FirstOrderODE>(*space_time_res, ode_opts, 
                                                  *nonlinear_solver);
   }

};

TEST_CASE("Testing AbstractSolver using RK4", "[abstract-solver]")
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? miso::getOutStream(0) : miso::getOutStream(1);
   using namespace mfem;
   using namespace miso;

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
   Vector u(solver.getStateSize());
   u(0) = 1.0;
   u(1) = 0.5;
   MISOInputs inputs;
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
   std::ostream *out = verbose ? miso::getOutStream(0) : miso::getOutStream(1);
   using namespace mfem;
   using namespace miso;

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

   // Check that solution is reasonable accurate
   auto exact_sol = [](double t, Vector &u)
   {
      const double e = std::exp(1.0);
      const double sepe = sqrt(e) + e;
      u.SetSize(2);
      u(0) = log(e + pow(e,1.5)) - log(sqrt(e) + exp(sepe*t));
      u(1) = log((sepe*exp(sepe*t))/(sqrt(e) + exp(sepe*t)));
   };

   // Create solver and solve for the state 
   ExponentialODESolver solver(MPI_COMM_WORLD, options);
   Vector u0(solver.getStateSize()), u(solver.getStateSize());
   solver.setState([&](mfem::Vector &u) {exact_sol(0.0, u); }, u0);
   u = u0;
   MISOInputs inputs;
   solver.solveForState(inputs, u);

   auto t_final = options["time-dis"]["t-final"].get<double>();
   Vector u_exact;
   exact_sol(options["time-dis"]["t-final"].get<double>(), u_exact);
   // auto error = solver.calcStateError([&](mfem::Vector &u) {exact_sol(t_final, u); }, u);
   auto error = solver.calcStateError(u_exact, u);
   double entropy0 = exp(u0(0)) + exp(u0(1));
   double entropy = exp(u(0)) + exp(u(1));

// -19.8579: 1.47408
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

TEST_CASE("Testing AbstractSolver using RRK6", "[abstract-solver]")
{
   const bool verbose = true; // set to true for some output 
   std::ostream *out = verbose ? miso::getOutStream(0) : miso::getOutStream(1);
   using namespace mfem;
   using namespace miso;

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "print-options": true,
      "time-dis": {
         "type": "RRK6",
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

   // Check that solution is reasonable accurate
   auto exact_sol = [](double t, Vector &u)
   {
      const double e = std::exp(1.0);
      const double sepe = sqrt(e) + e;
      u.SetSize(2);
      u(0) = log(e + pow(e,1.5)) - log(sqrt(e) + exp(sepe*t));
      u(1) = log((sepe*exp(sepe*t))/(sqrt(e) + exp(sepe*t)));
   };

   // Create solver and solve for the state 
   ExponentialODESolver solver(MPI_COMM_WORLD, options);
   Vector u0(solver.getStateSize()), u(solver.getStateSize());
   solver.setState([&](mfem::Vector &u) {exact_sol(0.0, u); }, u0);
   u = u0;
   MISOInputs inputs;
   solver.solveForState(inputs, u);

   auto t_final = options["time-dis"]["t-final"].get<double>();
   Vector u_exact;
   exact_sol(options["time-dis"]["t-final"].get<double>(), u_exact);
   // auto error = solver.calcStateError([&](mfem::Vector &u) {exact_sol(t_final, u); }, u);
   auto error = solver.calcStateError(u_exact, u);
   double entropy0 = exp(u0(0)) + exp(u0(1));
   double entropy = exp(u(0)) + exp(u(1));

// -19.8579: 1.47408
   if (verbose)
   {
      std::cout << "discrete solution = " << u(0) << ": " << u(1) << std::endl;
      std::cout << "exact solution    = " << u_exact(0) << ": " << u_exact(1)
                << std::endl;
      std::cout << "terminal solution error = " << error << std::endl;
      std::cout << "entropy error = " << entropy - entropy0 << std::endl;
   }
   REQUIRE( error < 7.0e-7 );

   REQUIRE( entropy == Approx(entropy0).margin(1e-12) );
}