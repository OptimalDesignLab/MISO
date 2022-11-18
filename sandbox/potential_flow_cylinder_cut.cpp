// Solve for the steady flow around a NACA0012

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "euler_dg_cut.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0, 1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector &p);

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector &u);
/// Generate quarter annulus mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
int main(int argc, char *argv[])
{
   const char *options_file = "potential_flow_cylinder_cut_options.json";
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   int N = 5;
   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
   args.AddOption(&N, "-n", "--#elements", "number of elements");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   double M = 0.2;
   double circ = 0.1;
   double rad = 0.5;
   try
   {
      // construct the solver, set the initial condition, and solve
      unique_ptr<Mesh> smesh(new Mesh(buildMesh(N)));
      *out << "Number of elements " << smesh->GetNE() << '\n';
      ofstream sol_ofs("cart_mesh_dg_cut_init.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 0);
      string opt_file_name(options_file);
      auto solver =
          createSolver<CutEulerDGSolver<2, entvar>>(opt_file_name, move(smesh));
      //   Vector qfar(4);
      //   static_cast<CutEulerDGSolver<2, entvar> *>(solver.get())
      //       ->getFreeStreamState(qfar);
      // qfar.Print();
      out->precision(15);
      // Vector wfar(4);
      // TODO: I do not like that we have to perform this conversion outside the
      // solver...
      // calcEntropyVars<double, 2>(qfar.GetData(), wfar.GetData());
      // solver->setInitialCondition(qfar);
      solver->setInitialCondition(uexact);
      solver->printSolution("cylinder-steady-dg-cut-potential-init", 0);
      auto drag_opts = R"({ "boundaries": [0, 0, 0, 0]})"_json;
      solver->createOutput("drag", drag_opts);
      solver->createOutput("lift", drag_opts);
      double drag;
      *out << "\nInitial Drag error = " << abs(solver->calcOutput("drag"))
           << endl;
      *out << "\nexact cl value = " << (circ / M) << endl;
      *out << "\nInitial cl value = " << abs(solver->calcOutput("lift"))
           << endl;
      // get the initial density error
      double l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      *out << "Initial \n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      solver->solveForState();
      solver->printSolution("cylinder-steady-dg-cut-potential-final", -1);
      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
                << endl;
      l2_error = (static_cast<CutEulerDGSolver<2, entvar> &>(*solver)
                      .calcConservativeVarsL2Error(uexact, 0));
      *out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\nDrag error = " << abs(solver->calcOutput("drag")) << endl;
      *out << "\ncl value = " << abs(solver->calcOutput("lift")) << endl;
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

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector &p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
}
#if 0
/// use this for flow over an ellipse
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   double theta;
   double Ma = 0.2;
   double rho = 1.0;
   double p = 1.0 /*/ euler::gamma*/;
   /// ellipse parameters
   double xc = 5.0;
   double yc = 5.0;
   double a = 2.5;
   double b = sqrt(a * (a - 1.0));
   double s =
       ((x(0) - xc) * (x(0) - xc)) - ((x(1) - yc) * (x(1) - yc)) - 4.0 * b * b;
   double t = 2.0 * (x(0) - xc) * (x(1) - yc);
   theta = atan2(t, s);
   double signx = 1.0;
   if (x(0) - xc < 0)
   {
      signx = -1.0;
   }
   double r = sqrt(t * t + s * s);
   double xi = 0.5 * (x(0) - xc + (signx * sqrt(r) * cos(theta / 2.0)));
   double eta = 0.5 * (x(1) - yc + (signx * sqrt(r) * sin(theta / 2.0)));
   double term_a = xi * xi - eta * eta - a * a;
   double term_b = xi * xi - eta * eta - b * b;
   double term_c = 4.0 * xi * xi * eta * eta;
   double term_d = (term_b * term_b) + term_c;
   u(0) = rho;
   u(1) = rho * Ma * ((term_a * term_b) + term_c) / term_d;
   u(2) = -rho * Ma * 2.0 * xi * eta * (term_b - term_a) / term_d;
   u(3) = p / euler::gami + 0.5 * Ma * Ma;
   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}
#endif
#if 1
void uexact(const Vector &x, Vector &q)
{
   q.SetSize(4);
   Vector u(4);
   double theta;
   double Ma = 0.2;
   double rho = 1.0;
   double p = 1.0 / euler::gamma;
   /// circle parameters
   double xc = 5.00;
   double yc = 5.00;
   double rad = 0.5;
   double circ = 0.1;
   theta = atan2(x(1) - yc, x(0) - xc);
   double r = sqrt(((x(0) - xc) * (x(0) - xc)) + ((x(1) - yc) * (x(1) - yc)));
   double rinv = rad / r;
   double rtilde = 1.0 / rinv;
   double Vr = Ma * (1.0 - rinv * rinv) * cos(theta);
   double Vth = -Ma * (1.0 + rinv * rinv) * sin(theta) - circ / (M_PI * rtilde);
   double ux = (Vr * cos(theta)) - (Vth * sin(theta));
   double uy = (Vr * sin(theta)) + (Vth * cos(theta));
   // directly derived u , v from complex potential, w
   //    u(1) = rho*Ma*(1.0 - rinv * rinv*cos(2.0*theta));
   //    u(2) = -rho * Ma*rinv * rinv * sin(2.0*theta);
   // u(3) = p / euler::gami + 0.5 * Ma * Ma;
   double p_bern =
       1.0 / euler::gamma + 0.5 * Ma * Ma - 0.5 * rho * (ux * ux + uy * uy);
   double p_euler =
       euler::gami * (u(3) - 0.5 * rho * (u(1) * u(1) + u(2) * u(2)));

   u(0) = rho;
   u(1) = rho * ux;
   u(2) = rho * uy;
   //u(3) = p_bern / euler::gami + 0.5 * Ma * Ma;
   u(3) = p_bern / euler::gami + 0.5 * (ux * ux + uy * uy);
   //    cout << "p_bern: " << p_bern << endl;
   //    cout << "p_euler: " << p_euler << endl;
   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }

}
#endif
Mesh buildMesh(int N)
{
   Mesh mesh = Mesh::MakeCartesian2D(
       N, N, Element::QUADRILATERAL, true, 10.0, 10.0, true);
   return mesh;
}
