/// Solve the steady isentropic vortex problem on a quarter annulus

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <fstream>
#include <iostream>

#include "miso.hpp"

using namespace std;
using namespace mfem;
using namespace miso;

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

int main(int argc, char *argv[])
{
   // Get the options
   const char *options_file = "steady_vortex_options.json";
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
   int mesh_degree = 2;
   int nx = 1;
   int ny = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&mesh_degree, "-d", "--degree",
                  "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {
      // Get options from file and build mesh
      string opt_file_name(options_file);
      auto mesh = buildQuarterAnnulusMesh(mesh_degree, nx, nx);
      *out << "Number of elements " << mesh->GetNE() << endl;
      ofstream sol_ofs("steady_vortex_mesh.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, 2);

      // Create solver and set initial condition
      FlowSolver<2, entvar> solver(MPI_COMM_WORLD, options, std::move(mesh));
      mfem::Vector state_tv(solver.getStateSize());
      solver.setState(uexact, state_tv);

      // get the initial density error
      double l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
      double res_error = solver.calcResidualNorm(state_tv);
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;

      // Create the output(s), inputs, and solve for state
      solver.createOutput("entropy", options["outputs"].at("entropy"));
      solver.createOutput("drag", options["outputs"].at("drag"));
      MisoInputs inputs({{"state", state_tv}});
      solver.solveForState(inputs, state_tv);

      // Evaluate the density, entropy and drag errors
      res_error = solver.calcResidualNorm(state_tv); 
      l2_error = solver.calcConservativeVarsL2Error(uexact, 0);
      double entropy = solver.calcOutput("entropy", inputs);
      double drag = solver.calcOutput("drag", inputs);
      out->precision(15);
      *out << "\nfinal residual norm = " << res_error;
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      *out << "\nDrag error = " << fabs(drag - (-1/ miso::euler::gamma)) 
           << endl;
      *out << "\nTotal entropy = " << entropy;
      *out << "\nEntropy error = "
           << fabs(entropy - calcEntropyTotalExact()) << endl;
   }
   catch (MISOException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
   MPI_Finalize();
}

// Returns the exact total entropy value over the quarter annulus
// Note: the number 8.74655... that appears below is the integral of r*rho over the radii
// from 1 to 3.  It was approixmated using a degree 51 Gaussian quadrature.
double calcEntropyTotalExact()
{
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double si = log(prsi/pow(rhoi, euler::gamma));
   return -si*8.746553803443305*M_PI*0.5/0.4;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector& q)
{
   q.SetSize(4);
   Vector u(4);
   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double rinv = ri/sqrt(x(0)*x(0) + x(1)*x(1));
   double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/euler::gami);
   double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                    (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1)/x(0));
   }
   else
   {
      theta = M_PI/2.0;
   }
   double press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   double a = sqrt(euler::gamma*press/rho);

   u(0) = rho;
   u(1) = -rho*a*Ma*sin(theta);
   u(2) = rho*a*Ma*cos(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;

   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2, false>(u.GetData(), q.GetData());
   }
}
