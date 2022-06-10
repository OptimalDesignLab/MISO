#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "mach.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

/// Set the initial value of the control state
/// \param[out] u0 - the control state at time t0
void cInit(Vector & u0);

/// Set the y position of the control given the x position
/// \param[in] x - the x position of the control 
/// \note This assumes a NACA 0012 airfoil
/// \note Assumes, but does not check, that x is between 0 and 1
double yControl(double x);

int main(int argc, char *argv[])
{
   // Get the options
   const char *options_file = "airfoil_flow_control_options.json";
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;

   // Open the parameter file
   const char *params_file = "control_parameters.json";
   nlohmann::json params;
   ifstream params_source(params_file);
   params_source >> params;

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
   int sample;
   args.AddOption(&sample, "-s", "--sample", "Parameter sample to use");
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

      // Create solver and set initial condition 
      FlowControlSolver<2> solver(MPI_COMM_WORLD, options);
      //FlowSolver<2> solver(MPI_COMM_WORLD, options);
      mfem::Vector state_tv(solver.getStateSize());
      Vector qfar(4);
      solver.getFreeStreamState(qfar);
      auto uInit = [&](const Vector &x, Vector &u0) { u0 = qfar; };
      solver.setState(
          std::make_pair(std::function(cInit), std::function(uInit)), state_tv);
      //solver.setState(uInit, state_tv);

      // Set the inputs using the desired sample
      nlohmann::json param = params["samples"][sample];
      *out << "Control parameters used:" << std::endl;
      *out << std::setw(3) << param << std::endl;
      double Kp = param["Kp"];
      double Ti = param["Ti"];
      double Td = param["Td"];
      double beta = param["beta"];
      double eta = param["eta"];
      double xac = param["x_c"];
      double yac = yControl(xac);
      mfem::Vector x_actuator({xac, yac});
      double target_entropy = param["target-ent"];
      bool closed_loop = param["closed-loop"];
      std::vector<double> pvec = param["P-matrix"].get<std::vector<double>>();
      Vector P(pvec.data(), 4);   

      // // Set all the necessary inputs
      // const double Kp = 0.4, Ti = 0.8, Td = 0.5, beta = 2.5, eta = 0.8;
      // // target entropy is from boundary entropy evaluated at initial field 
      // const double target_entropy = 0.1135579764419388;
      // bool closed_loop = true;
      // Vector P(4);
      // P(0) = 36.7614241; 
      // P(1) = -88.2050467;
      // P(2) = -88.2050467;
      // P(3) = 213.6134806;
      // double xac = 0.25;
      // double yac = yControl(xac);
      // mfem::Vector x_actuator({xac, yac});

      solver.setOptions({"sample", sample});

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

      // create the outputs to track
      solver.createOutput("entropy", options["outputs"].at("entropy"));
      solver.createOutput("boundary-entropy",
                          options["outputs"].at("boundary-entropy"));
      solver.createOutput("drag", options["outputs"].at("drag"));
      solver.createOutput("lift", options["outputs"].at("lift"));

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

void cInit(Vector &u)
{
   u(0) = 0.0;
   u(1) = 0.0; // 0.01
}

double yControl(double x)
{
   const double t = 0.12;  // thickness for NACA0012
   double y = 5 * t *
              (0.2969 * sqrt(x) -
               (0.1260 - (0.3516 + (0.2843 - 0.1015 * x) * x) * x) * x);
   return y;
}
