#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

/// Set the initial value of the control state
/// \param[out] u0 - the control state at time t0
void cinit(Vector & u0);

/// Set the initial value of the flow state
/// \param[in] x - spatial location
/// \param[out] u0 - value of the state at `x`
void uinit(const Vector &x, Vector& u0);

int main(int argc, char *argv[])
{
   // Get the options
   const char *options_file = "forced_box_options.json";
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);

   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);

      // Build mesh over the square domain
      const int num = 20;
      auto smesh = std::make_unique<Mesh>(Mesh::MakeCartesian2D(
         num, num, Element::TRIANGLE, true /* gen. edges */, 1.0, 1.0, true));

      // Create solver and set initial guess to constant
      FlowControlSolver<2> solver(MPI_COMM_WORLD, options, std::move(smesh));
      mfem::Vector state_tv(solver.getStateSize());
      //solver.setState(uinit, state_tv);
      //solver.setState({.control_func = cinit, .flow_func = uinit}, state_tv);

      solver.setState(
          std::make_pair(std::function(cinit), std::function(uinit)), state_tv);

      // Set all the necessary inputs
      const double Kp = 0.4, Ti = 0.8, Td = 0.5, beta = 2.5, eta = 0.8;
      const double target_entropy = 0.0;
      bool closed_loop = true;
      Vector P(4);
      P(0) = 36.7614241; 
      P(1) = -88.2050467;
      P(2) = -88.2050467;
      P(3) = 213.6134806;
      mfem::Vector x_actuator({0.0, 0.5});
      MachInputs inputs({{"state", state_tv},
                         {"time", 0.0},
                         {"x-actuator", x_actuator},
                         {"Kp", Kp},
                         {"Ti", Ti},
                         {"Td", Td},
                         {"beta", beta},
                         {"eta", eta},
                         {"target-entropy", target_entropy},
                         {"boundary-entropy", 0.0},
                         {"closed-loop", float(closed_loop)},
                         {"P-matrix", P}});

      // get the initial entropy 
      solver.createOutput("entropy", options["outputs"].at("entropy"));
      double entropy0 = solver.calcOutput("entropy", inputs);
      cout << "initial entropy = " << entropy0 << endl;

      // Solve for the state
      solver.solveForState(inputs, state_tv);

      // get the final entropy 
      double entropy = solver.calcOutput("entropy", inputs);
      cout << "final entropy = " << entropy << endl;
   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
   MPI_Finalize();
}

void cinit(Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.05;
}

void uinit(const Vector &x, Vector &u0)
{
   u0.SetSize(4);
   u0(0) = 1.0;
   u0(1) = 0.0;
   u0(2) = 0.0;
   double press = pow(u0(0), euler::gamma);
   u0(3) = press/euler::gami;
}

