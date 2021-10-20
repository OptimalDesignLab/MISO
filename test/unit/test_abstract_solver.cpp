#include <random>

#include "catch.hpp"

#include "mfem_extensions.hpp"
#include "abstract_solver.hpp"


TEST_CASE("Testing AbstractSolver as TimeDependentOperator",
          "[abstract-solver]")
{
   const bool verbose = false; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;

   class ExpODEResidual final
   {
      friend void evaluate(ExpODEResidual &residual,
                           const MachInputs &inputs,
                           mfem::Vector &res_vec)
      {
         Vector x(inputs.at("state").getField(), 2);
         res_vec.SetSize(2);
         res_vec(0) =  exp(x(1));
         res_vec(1) = -exp(x(0));
      }
   };

   /// Solver that uses `ExpODEResidual` to define its dynamics
   class ExponentialODESolver : public AbstractSolver2
   {
   public:
      void initDerived() override
      {
         res.reset(new ExpODEResidual);
      }   
      friend SolverPtr createSolver<ExponentialODESolver>(
         const nlohmann::json &json_options);
   };

   // Provide the options explicitly for regression tests
   auto options = R"(
   {
      "print-options": false
      "time-dis": {
         "type": "RK4",
         "t-final": 5.0,
         "dt": 0.05
      }
   })"_json;

   // Create solver and solve for the state 
   auto solver = createSolver<ExponentialODESolver>(options);
   Vector u(2);
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
   double entropy0 = dynamic_cast<ExponentialODE&>(*ode).Entropy(u0);
   double entropy = dynamic_cast<ExponentialODE&>(*ode).Entropy(u);

   if (verbose)
   {
      std::cout << "discrete solution = " << u(0) << ": " << u(1) << std::endl;
      std::cout << "exact solution    = " << u_exact(0) << ": " << u_exact(1)
                << std::endl;
      std::cout << "terminal solution error = " << error << std::endl;
   }
   //REQUIRE( error == Approx(0.003).margin(1e-4) );
}
