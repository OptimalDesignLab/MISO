#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "abstract_solver.hpp"
#include "mach_input.hpp"
#include "mach_residual.hpp"
#include "mfem_extensions.hpp"
#include "utils.hpp"

/// Class for ODE that follows the MachResidual API
class ExpODEResidual final
{
public:
   ExpODEResidual() : Jac(2) {}
   friend int getSize(const ExpODEResidual &residual) { return 2; }
   friend void setInputs(ExpODEResidual &residual,
                         const mach::MachInputs &inputs)
   {
      auto it = inputs.find("state");
      if (it != inputs.end())
      {
         residual.state.SetDataAndSize(it->second.getField(),
                                       getSize(residual));
      }
      it = inputs.find("state_dot");
      if (it != inputs.end())
      {
         residual.state_dot.SetDataAndSize(it->second.getField(),
                                           getSize(residual));
      }
      it = inputs.find("dt");
      if (it != inputs.end())
      {
         residual.dt = it->second.getValue();
      }
      it = inputs.find("time");
      if (it != inputs.end())
      {
         residual.time = it->second.getValue();
      }
   }
   friend void setOptions(ExpODEResidual &residual,
                          const nlohmann::json &options) {}
   friend void evaluate(ExpODEResidual &residual,
                        const mach::MachInputs &inputs,
                        mfem::Vector &res_vec)
   {
      mfem::Vector &x = residual.state;
      // mfem::Vector &x_dot = residual.state_dot;
      mfem::Vector x_dot(inputs.at("state").getField(), getSize(residual));
      double dt = residual.dt;
      res_vec.SetSize(2);
      res_vec(0) = x_dot(0) + exp(x(1) + dt * x_dot(1));
      res_vec(1) = x_dot(1) - exp(x(0) + dt * x_dot(0));
   }
   friend mfem::Operator &getJacobian(ExpODEResidual &residual,
                                      const mach::MachInputs &inputs,
                                      std::string wrt)
   {
      mfem::Vector &x = residual.state;
      // mfem::Vector &x_dot = residual.state_dot;
      mfem::Vector x_dot(inputs.at("state").getField(), getSize(residual));
      double dt = residual.dt;
      residual.Jac(0,0) = 1.0;
      residual.Jac(0,1) = dt*exp(x(1) + dt * x_dot(1));
      residual.Jac(1,0) = -dt*exp(x(0) + dt * x_dot(0));
      residual.Jac(1,1) = 1.0;
      return residual.Jac;
   }
private:
   double dt;
   double time;
   mfem::Vector state;
   mfem::Vector state_dot;
   mfem::DenseMatrix Jac;
};

/// Solver that uses `ExpODEResidual` to define its dynamics
class ExponentialODESolver : public mach::AbstractSolver2
{
public:
   ExponentialODESolver(const nlohmann::json &solver_options, MPI_Comm comm)
      : AbstractSolver2(solver_options, comm)
   {
      res = std::make_unique<mach::MachResidual>(ExpODEResidual());

      auto lin_solver_opts = options["lin-solver"];
      auto nonlin_solver_opts = options["nonlin-solver"];
      solver = std::make_unique<mach::EquationSolver>(comm,
                                                      lin_solver_opts,
                                                      nullptr,
                                                      nonlin_solver_opts);
      solver->SetOperator(*res);

      auto ode_opts = options["time-dis"];
      ode = std::make_unique<mach::FirstOrderODE>(*res, ode_opts, *solver);
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
         "type": "gmres",
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
   ExponentialODESolver solver(options, MPI_COMM_WORLD);
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