#include <random>

#include "catch.hpp"

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
                         const mach::MachInputs &inputs) {}
   friend void setOptions(ExpODEResidual &residual,
                          const nlohmann::json &options) {}
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
      double dt = inputs.at("dt").getValue();
      residual.Jac(0,0) = 1.0;
      residual.Jac(0,1) = dt*exp(x(1));
      residual.Jac(1,0) = -dt*exp(x(0));
      residual.Jac(1,1) = 1.0;
      return residual.Jac;
   }
private:
   mfem::DenseMatrix Jac;
};

/// Solver that uses `ExpODEResidual` to define its dynamics
class ExponentialODESolver : public mach::AbstractSolver2
{
public:
   void initDerived() override
   {
      height = width = 2;
      res.reset(new mach::MachResidual(ExpODEResidual()));
   }
protected:
   ExponentialODESolver(const nlohmann::json &json_options, MPI_Comm comm) :
      AbstractSolver2(json_options, comm) {}
   friend mach::SolverPtr2 mach::createSolver<ExponentialODESolver>(
       const nlohmann::json &json_options, MPI_Comm comm);
};

TEST_CASE("Testing AbstractSolver as TimeDependentOperator with RK4",
          "[abstract-solver]")
{
   const bool verbose = false; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;
   using namespace mach;

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "print-options": false,
      "time-dis": {
         "type": "RK4",
         "t-final": 5.0,
         "dt": 0.05
      }
   })"_json;

   // Create solver and solve for the state 
   auto solver = createSolver<ExponentialODESolver>(options, MPI_COMM_WORLD);
   Vector u(2);
   u(0) = 1.0;
   u(1) = 0.5;
   MachInputs inputs;
   solver->solveForState(inputs, u);

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
