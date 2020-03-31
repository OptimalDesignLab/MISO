/// Solve the steady isentropic vortex problem on a quarter annulus
#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"

#include <fstream>
#include <iostream>
#include <random>

// #ifdef MFEM_USE_SIMMETRIX
// #include <SimUtil.h>
// #include <gmi_sim.h>
// #endif
// #include <apfMDS.h>
// #include <gmi_null.h>
// #include <PCU.h>
// #include <apfConvert.h>
// #include <gmi_mesh.h>
// #include <crv.h>

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const mfem::Vector &x, mfem::Vector &u);
void uexact_single(const mfem::Vector &x, mfem::Vector &u);
void upoly(const mfem::Vector &x, mfem::Vector &u);

/// Generate quarter annulus mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
   // the specfic option file for this problem
   // const char *options_file = "galerkin_difference.json";
   // nlohmann::json options;
   // ifstream option_source(options_file);
   // option_source >> options;
   // option_source.close();

#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#else
   int myid = 0;
#endif

   int dim; // space dimension of the mesh
   //Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2;
   int p = 1;
   int nx = 1;
   int ny = 1;
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-thetat", "number of angular segments");
   args.AddOption(&p, "-p", "--order", "projection operator degree");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
      std::cout << "Number of elements " << smesh->GetNE() << '\n';
      ofstream sol_ofs("gd_test.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs, 0);
      sol_ofs.close();
      mfem::Mesh *mesh = new Mesh(*smesh);
      dim = mesh->Dimension();
      // for(int degree = 0; degree < 4; degree++)
      // {
      //    GalerkinDifference gd(degree, pumi_mesh, fec.get(), 1, Ordering::byVDIM);
      //    gd.BuildGDProlongation();
      //    mfem::GridFunction x(&gd);
      //    mfem::GridFunction x_exact(&gd);

      // }

      cout << "Construct the GD fespace.\n";
      DSBPCollection fec(1, dim);
      GalerkinDifference gd(mesh, &fec, 1, Ordering::byVDIM, p);
      //GalerkinDifference gd(degree, pumi_mesh, 1, Ordering::byVDIM);
      cout << "Now build the prolongation matrix.\n";
      //GalerkinDifference gd(options_file, pumi_mesh);

      // Test the prolongation matrix with gridfunction vdim = 4
      mfem::GridFunction x(&gd);
      mfem::GridFunction x_exact(&gd);

      mfem::VectorFunctionCoefficient u0(1, upoly);
      x_exact.ProjectCoefficient(u0);
      // cout << "Check the exact solution:\n";
      // x_exact.Print(cout ,4);

      mfem::CentGridFunction x_cent(&gd);
      x_cent.ProjectCoefficient(u0);
      //cout << "Size of x_cent is " << x_cent.Size() << '\n';
      // cout << "\n\n\n\nCheck the the center values:\n";
      // x_cent.Print(cout, 4);

      gd.GetProlongationMatrix()->Mult(x_cent, x);
      // cout << "\n\n\n\nCheck the results:\n";
      // x.Print(cout,4);
      x -= x_exact;
      cout << "Check the projection error: " << x.Norml2() << '\n';

      std::cout << "Check P transpose: ";
      mfem::GridFunction y(&gd);
      mfem::CentGridFunction y_cent(&gd);

      mfem::VectorFunctionCoefficient u1(1, upoly);
      y.ProjectCoefficient(u1);
      y_cent.ProjectCoefficient(u1);

      mfem::GridFunction y_project(&gd);
      gd.GetProlongationMatrix()->Mult(y_cent, y_project);
      
      const FiniteElement *fe;
      ElementTransformation *T;
      double h;
      int num_pt;

      // sum up the L2 error over all states
      for (int i = 0; i < gd.GetNE(); i++)
      {
         fe = gd.GetFE(i);
         const IntegrationRule *ir = &(fe->GetNodes());
         T = gd.GetElementTransformation(i);
         num_pt = ir->GetNPoints();
         for (int j = 0; j < num_pt; j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            h = ip.weight * T->Weight();
            y_project(i*num_pt+j) = h * y_project(i*num_pt+j);
         }
      }
      double lhs = x_exact * y_project;

      mfem::CentGridFunction x_project(&gd);
      for (int i = 0; i < gd.GetNE(); i++)
      {
         fe = gd.GetFE(i);
         const IntegrationRule *ir = &(fe->GetNodes());
         T = gd.GetElementTransformation(i);
         num_pt = ir->GetNPoints();
         for (int j = 0; j < num_pt; j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            h = ip.weight * T->Weight();
            x_exact(i*num_pt+j) = h * x_exact(i*num_pt+j);
         }
      }
      gd.GetProlongationMatrix()->MultTranspose(x_exact, x_project);
      double rhs = y_cent * x_project;
      std::cout << "x^t * P * yc - yc^t * P^t * x^t = " << lhs - rhs << '\n';
      std::cout << "lhs = " << lhs << ", rhs = " << rhs << '\n';
      // Test the prolongation matrix with gridfunction vdim = 1
      // mfem::GridFunction x(&gd);
      // mfem::GridFunction x_exact(&gd);
      // cout << "Size of x and x_exact is " << x.Size() << '\n';

      // mfem::VectorFunctionCoefficient u0(1, uexact_single);
      // x_exact.ProjectCoefficient(u0);
      // cout << "Check the exact solution:\n";
      // x_exact.Print(cout ,7);

      // mfem::CentGridFunction x_cent(&gd);
      // cout << "Size of x_cent is " << x_cent.Size() << '\n';
      // x_cent.ProjectCoefficient(u0);
      // cout << "\n\n\n\nCheck the the center values:\n";
      // x_cent.Print(cout, 7);

      // gd.GetProlongationMatrix()->Mult(x_cent, x);
      // cout << "\n\n\n\nCheck the results:\n";
      // x.Print(cout,7);
      // x -= x_exact;
      // cout << "Check the error: " << x.Norml2() << '\n';
      delete mesh;
   }

   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
#ifdef MFEM_USE_MPI
   MPI_Finalize();
#endif
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const mfem::Vector &x, mfem::Vector &u)
{
   // steady vortext exact solution with shifted coordinate
   u.SetSize(1);
   double ri = 1.0;
   double Mai = 0.5; //0.95
   double rhoi = 2.0;
   double prsi = 1.0 / euler::gamma;
   double rinv = ri / sqrt(x(0) * x(0) + x(1) * x(1));
   double rho = rhoi * pow(1.0 + 0.5 * euler::gami * Mai * Mai * (1.0 - rinv * rinv),
                           1.0 / euler::gami);
   double Ma = sqrt((2.0 / euler::gami) * ((pow(rhoi / rho, euler::gami)) *
                                               (1.0 + 0.5 * euler::gami * Mai * Mai) -
                                           1.0));
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan( x(1)/x(0) );
   }
   else
   {
      theta = M_PI / 2.0;
   }
   double press = prsi * pow((1.0 + 0.5 * euler::gami * Mai * Mai) /
                                 (1.0 + 0.5 * euler::gami * Ma * Ma),
                             euler::gamma / euler::gami);
   double a = sqrt(euler::gamma * press / rho);

   // u(0) = rho;
   // u(1) = rho*a*Ma*sin(theta);
   // u(2) = -rho*a*Ma*cos(theta);
   // u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;

   u(0) = rho + rho * a * Ma * sin(theta) - rho * a * Ma * cos(theta) + press / euler::gami + 0.5 * rho * a * a * Ma * Ma;
}

void upoly(const mfem::Vector &x, mfem::Vector &u)
{
   u(0) = 0;
   for (int i = 2; i >= 0; i--)
   {
      u(0) += pow(x(0) + x(1), i);
   }
}

void uexact_single(const mfem::Vector &x, mfem::Vector &u)
{
   u(0) = x(0);
   //u(0) = x(0) * x (0) - 7.0 * x(0) + 3.0;
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI * 0.5, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes

   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector &rt, Vector &xy) {
      xy(0) = (rt(0) + 1.0) * cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0) * sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}