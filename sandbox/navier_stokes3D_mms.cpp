/// Solve the Navier-Stokes MMS verification

#include "mfem.hpp"
#include "navier_stokes.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

/// Generate smoothly perturbed mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_x - number of nodes in the x direction
/// \param[in] num_y - number of nodes in the y direction
unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y, int num_z);

/// \brief Defines the exact solution for the manufactured solution
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

int main(int argc, char *argv[])
{
   const char *options_file = "navier_stokes3D_mms_options.json";

   // Initialize MPI
   int num_procs, rank;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   ostream *out = getOutStream(rank);
   *out << std::setprecision(15); 

   // Parse command-line options
   OptionsParser args(argc, argv);

   int degree = 2.0;
   int nx = 1;
   int ny = 1;
   int nz = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nx", "--num-x", "number of x-direction segments");
   args.AddOption(&ny, "-ny", "--num-y", "number of y-direction segments");
   args.AddOption(&nz, "-nz", "--num-z", "number of z-direction segments");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(*out);
      return 1;
   }

   try
   {
      // construct the mesh
      string opt_file_name(options_file);
      auto smesh = buildCurvilinearMesh(degree, nx, ny, nz);
      *out << "Number of elements " << smesh->GetNE() <<'\n';
      ofstream sol_ofs("navier_stokes_mms_mesh.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 3);

      // construct the solver and set the initial condition
      auto solver = createSolver<NavierStokesSolver<3>>(opt_file_name,
                                                        move(smesh));
      solver->setInitialCondition(uexact);
      solver->printSolution("init", degree+1);
      solver->printResidual("init-res", degree+1);

      *out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(uexact, 0) << '\n' << endl;
      *out << "\ninitial residual norm = " << solver->calcResidualNorm() << endl;

      solver->solveForState();
      solver->printSolution("final", degree+1);
      double drag = solver->calcOutput("drag");

      *out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver->calcL2Error(uexact, 0) << '\n' << endl;
      *out << "\ndrag \"error\" = " << drag - 1.6 << endl;
      *out << "\nfinal residual norm = " << solver->calcResidualNorm() << endl;

      // TEMP
      //static_cast<NavierStokesSolver<2>*>(solver.get())->setSolutionError(uexact);
      //solver->printSolution("error", degree+1);

      // Solve for and print out the adjoint
      solver->solveForAdjoint("drag");
      solver->printAdjoint("adjoint", degree+1);

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

unique_ptr<Mesh> buildCurvilinearMesh(int degree, int num_x, int num_y, int num_z)
{
   Mesh mesh = Mesh::MakeCartesian3D(num_x, num_y, num_z, 
                                     Element::TETRAHEDRON, 1.0, 1.0, 1.0, Ordering::byVDIM);
   
   // Loop through the boundary elements and compute the normals at the centers of those elements
   // for (int it = 0; it < mesh.GetNBE(); ++it)
   // {  
   //    mesh.SetBdrAttribute(it, 1);
   //    Vector normal(3);
   //    ElementTransformation *Trans = mesh.GetBdrElementTransformation(it);
   //    Trans->SetIntPoint(&Geometries.GetCenter(Trans->GetGeometryType()));
   //    CalcOrtho(Trans->Jacobian(), normal);
   //    std::cout << normal(0) << " " << normal(1) << " " << normal(2) << "\n";

   //    if ((abs(normal(0)) < 1e-14) && (abs(normal(1)) < 1e-14) && (abs(normal(2)) > 1e-14))
   //    { mesh.SetBdrAttribute(it, 2); }

   //    if ((abs(normal(0)) < 1e-14) && (abs(normal(1)) > 1e-14) && (abs(normal(2)) < 1e-14))
   //    { mesh.SetBdrAttribute(it, 3); }
   // }

   for (int i = 0; i < mesh.GetNBE(); ++i)
   {
      std::cout << mesh.GetBdrAttribute(i) << "\n";
   }

   return make_unique<Mesh>(mesh);
}

void uexact(const Vector &x, Vector& q)
{
   const double r_0 = 1.0;
   const double r_xyz = 1.0;
   const double u_0 = 0.0;
   const double v_0 = 0.0;
   const double w_0 = 0.0;
   const double T_0 = 1.0;
            
   q(0) = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x(0))*sin(2*r_xyz*M_PI*x(1))*sin(2*r_xyz*M_PI*x(2));
   q(1) = u_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   q(2) = v_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   q(3) = w_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.)); 
   double T = T_0;
   double p = q(0) * T;
   q(4) = p/euler::gami + 0.5 * q(0) * (q(1)*q(1) + q(2)*q(2) + q(3)*q(3)); 
}
