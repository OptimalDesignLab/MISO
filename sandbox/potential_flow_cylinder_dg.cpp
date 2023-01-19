// Solve for the steady flow around a NACA0012

// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include <random>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "euler_dg.hpp"

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
/// Generate  mesh
/// \param[in] N - number of elements in x-y direction
Mesh buildMesh(int N);
/// Generate circle mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] refine - the level of refinement
std::unique_ptr<Mesh> buildCircleMesh(int degree, int refine);
int main(int argc, char *argv[])
{
   const char *options_file = "potential_flow_cylinder_dg_options.json";
   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   int N = 5;
   int degree = 2.0;
   int refine = -1.0;
   // Parse command-line options
   OptionsParser args(argc, argv);
   args.AddOption(&options_file, "-o", "--options", "Options file to use.");
   args.AddOption(&degree, "-d", "--#deg", "degree of mesh elements");
   args.AddOption(&refine, "-r", "--#ref", "level of refinement");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   double M = 0.2;
   double circ = 0.0;
   double rad = 0.5;

   try
   {
      // construct the solver, set the initial condition, and solve
      // unique_ptr<Mesh> smesh(new Mesh(buildMesh(N)));
      unique_ptr<Mesh> smesh = buildCircleMesh(degree, refine);
      // const char *mesh_file = "cylinder_p2.msh";
      // auto smesh  = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));
      // for (int l = 0; l < refine; l++)
      // {
      //    smesh->UniformRefinement();
      // }
      *out << "Number of elements " << smesh->GetNE() << '\n';
      ofstream sol_ofs("circle_mesh_dg_init.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs);
      string opt_file_name(options_file);
      auto solver =
          createSolver<EulerDGSolver<2, entvar>>(opt_file_name, move(smesh));
      out->precision(15);
      solver->setInitialCondition(uexact);
      // solver->setInitialCondition(qfar);
      solver->printSolution("cylinder-steady-dg-potential-init");
      auto drag_opts = R"({ "boundaries": [1, 0]})"_json;
      auto lift_opts = R"({ "boundaries": [1, 0]})"_json;
      solver->createOutput("drag", drag_opts);
      solver->createOutput("lift", drag_opts);
      *out << "\nInitial cl value (slip-wall) = "
           << (solver->calcOutput("lift")) << endl;
      // solver->createOutput("lift", lift_opts);
      // *out << "\nInitial cl value (far-field) = "
      //      << abs(solver->calcOutput("lift")) << endl;
      double drag;
      *out << "\nInitial Drag error = " << abs(solver->calcOutput("drag"))
           << endl;
      *out << "\nexact cl value = " << (circ / M) << endl;

      // get the initial density error
      double l2_error = (static_cast<EulerDGSolver<2, entvar> &>(*solver)
                             .calcConservativeVarsL2Error(uexact, 1));
      double res_error = solver->calcResidualNorm();
      *out << "Initial \n|| rho_h - rho ||_{L^2} = " << l2_error;
      *out << "\ninitial residual norm = " << res_error << endl;
      solver->solveForState();
      solver->printSolution("cylinder-steady-dg-potential-final", -1);
      mfem::out << "\nfinal residual norm = " << solver->calcResidualNorm()
                << endl;
      l2_error = (static_cast<EulerDGSolver<2, entvar> &>(*solver)
                      .calcConservativeVarsL2Error(uexact, 1));
      *out << "============================================================"
           << endl;
      *out << "|| rho_h - rho ||_{L^2} = " << l2_error << endl;
      *out << "============================================================"
           << endl;
      *out << "\nDrag error = " << abs(solver->calcOutput("drag")) << endl;
      *out << " **** "
           << "cl value = " << (solver->calcOutput("lift")) << " **** " << endl;
      *out << "\ncl error = " << abs(solver->calcOutput("lift") - circ / M)
           << endl;
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
   double xc = 5.0;
   double yc = 5.0;
   double rad = 0.5;
   double circ = 0.0;
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
   // rho = 1.0 - 0.5*euler::gami*Ma*Ma*((ux * ux + uy * uy) - 1.0);
   // rho = pow(rho, 1.0/euler::gami);
   double p_bern =
       1.0 / euler::gamma + 0.5 * Ma * Ma - 0.5 * rho * (ux * ux + uy * uy);
   // rho = euler::gamma * p_bern;
   u(0) = rho;
   u(1) = rho * ux;
   u(2) = rho * uy;
   // u(3) = p_bern / euler::gami + 0.5 * Ma * Ma;
   u(3) = p_bern / euler::gami + 0.5 * rho * (ux * ux + uy * uy);
   // double p_euler = euler::gami * (u(3) - 0.5 * (ux * ux + uy * uy));
   //  cout << "p_bern: " << p_bern << endl;
   //  cout << "p_euler: " << p_euler << endl;
   //  cout << "p diff: " << (p_bern - p_euler) << endl;
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
       N, N, Element::QUADRILATERAL, true, 20.0, 20.0, true);
   return mesh;
}

unique_ptr<Mesh> buildCircleMesh(int degree, int ref_levels)
{
   const char *mesh_file = "periodic_rectangle.mesh";
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(mesh_file, 1, 1));
   ofstream solp_ofs("pre_perturbed_mesh.vtk");
   solp_ofs.precision(14);
   mesh_ptr->PrintVTK(solp_ofs, 0);
   solp_ofs.close();
#if 0
   double pert = 0.2;
   double nel = mesh_ptr->GetNE();
   double a0 = 0.5;
   // Randomly perturb interior nodes
   std::default_random_engine gen(std::random_device{}());
   std::uniform_real_distribution<double> uni_rand(-pert, pert);
   static constexpr double eps = std::numeric_limits<double>::epsilon();
   for (int i = 0; i < mesh_ptr->GetNV(); ++i)
   {
      double *vertex = mesh_ptr->GetVertex(i);

      // make sure vertex is interior
      // if (vertex[0] > eps && vertex[0] < 10.0 - eps)
      // {
         cout << "inside perturbation loop " << endl;
         cout << "old vertex: " << endl;
         cout << vertex[0] << " , " << vertex[1] << endl;
         // perturb coordinates
         vertex[0] += uni_rand(gen)  / sqrt(nel);
         vertex[1] += uni_rand(gen)  / sqrt(nel);
         double *vertex_new = mesh_ptr->GetVertex(i);
         cout << "new vertex: " << endl;
         cout << vertex_new[0] << " , " << vertex_new[1] << endl;
     //}
   }
   ofstream sol_ofs("perturbed_mesh_final.mesh");
   sol_ofs.precision(14);
   mesh_ptr->Print(sol_ofs);
#endif
   for (int l = 0; l < ref_levels; l++)
   {
      mesh_ptr->UniformRefinement();
   }
   cout << "Number of elements " << mesh_ptr->GetNE() << '\n';
   ofstream mesh_ofs("rectangle_mesh_circle.vtk");
   mesh_ofs.precision(14);
   mesh_ptr->PrintVTK(mesh_ofs, 0);
   mesh_ofs.close();
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes =
       new FiniteElementSpace(mesh_ptr.get(), fec, 2, Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy)
   {
      double xc = 5.0;
      double yc = 5.0;
      double r_far = 11.0;
      double a0 = 0.5;
      double b0 = a0;
      double delta = 1.0;  // We will have to experiment with this
      double r = 1.0 + tanh(delta * (rt(0) / r_far - 1.0)) / tanh(delta);
      double theta = rt(1);
      double b = b0 + (a0 - b0) * r;
      xy(0) = a0 * (r * r_far + 1.0) * cos(theta) + xc;
      xy(1) = b * (r * r_far + 1.0) * sin(theta) + yc;
      /// perturb mesh
#if 1
      double r_out = 4.8;
      // double a0 = 0.5;
      // double xc = 5.0;
      // double yc = 5.0;
      // Randomly perturb interior nodes
      std::default_random_engine gen(std::random_device{}());
      double pert = 0.5;
      std::uniform_real_distribution<double> uni_rand(-pert, pert);
      static constexpr double eps = std::numeric_limits<double>::epsilon();

      double rdist =
          sqrt((xy(0) - xc) * (xy(0) - xc) + (xy(1) - yc) * (xy(1) - yc));
      // make sure vertex is interior
      if (rdist < r_out - eps && rdist > a0 + 0.2)
      {
         cout << "inside perturbation loop " << endl;
         // perturb coordinates
         xy(0) += uni_rand(gen) * a0 * 0.1 ;
         xy(1) += uni_rand(gen) * a0 * 0.1 ;
      }
#endif
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);
   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}