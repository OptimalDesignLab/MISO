/// Solve the steady isentropic vortex problem on a quarter annulus
#include "mfem.hpp"
#include "euler.hpp"
#include "rbfspace.hpp"
#include "centgridfunc.hpp"

#include <fstream>
#include <iostream>
#include <random>
const bool entvar = true;

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief multiply exact solutions for testing
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void u_const(const mfem::Vector &x, mfem::Vector &u);
void u_poly(const mfem::Vector &x, mfem::Vector &u);
void u_exact(const mfem::Vector &x, mfem::Vector &u);

/// Generate quarter annulus mesh
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);

// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
   //Parse command-line options
   OptionsParser args(argc, argv);
   int p = 1;
   int nx = 4;
   int ny = 4;
   int o = 1;
   double sp = 1.0;
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-thetat", "number of angular segments");
   args.AddOption(&p, "-p", "--problem", "projection operator degree");
   args.AddOption(&o, "-o", "--order", "level of refinement");
   args.AddOption(&sp, "-sp", "--span", "level of refinement");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   try
   {
   //================== Multiply mesh files for testing ====================
      int dim;
      int degree = o+1; // annulus mesh degree, should be p+1
      unique_ptr<Mesh> mesh = buildQuarterAnnulusMesh(degree, nx, ny);
      ofstream sol_ofs("rbf_test.vtk");
      sol_ofs.precision(14);
      mesh->PrintVTK(sol_ofs, 0);
      sol_ofs.close();
      std::cout << "Number of elements " << mesh->GetNE() << '\n';
      dim = mesh->Dimension();
      int num_state = dim + 2;

   //================== Construct the gd and normal finite element spaces =========
      cout << "Construct the RBF fespace.\n";
      int nel = mesh->GetNE();
      DSBPCollection fec(o, dim);
      RBFSpace rbf(mesh.get(), &fec, nel, num_state, sp, Ordering::byVDIM);
      FiniteElementSpace fes(mesh.get(), &fec, num_state, Ordering::byVDIM);

   //================== Construct the gridfunction and apply the exact solution =======
      mfem::CentGridFunction x_cent(&rbf);
      mfem::GridFunction x(&fes);
      mfem::GridFunction x_exact(&fes);

      if (p == 1)
      {
         mfem::VectorFunctionCoefficient u0_fun(num_state, u_const);
         x_cent.ProjectCoefficient(u0_fun);
         x_exact.ProjectCoefficient(u0_fun);
         x = 0.0;
      }
      else if (p == 2)
      {
         mfem::VectorFunctionCoefficient u0_fun(num_state, u_poly);
         x_cent.ProjectCoefficient(u0_fun);
         x_exact.ProjectCoefficient(u0_fun);
         x = 0.0;
      }
      else
      {
         mfem::VectorFunctionCoefficient u0_fun(num_state, u_exact);
         x_cent.ProjectCoefficient(u0_fun);
         x_exact.ProjectCoefficient(u0_fun);
         x = 0.0;
      }

   //============== Prolong the solution to SBP nodes =================================
   //============== and check the simple l2 norm error ================================
      rbf.GetProlongationMatrix()->Mult(x_cent, x);
      x -= x_exact;
      cout << "Check the projection l2 error: " << x.Norml2() << '\n';
   }

   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
   return 0;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void u_exact(const mfem::Vector &x, mfem::Vector &q)
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
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}

void u_const(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(x.Size()+2);
   for (int i = 0; i < x.Size()+2; i++)
   {
      u(i) = (double)i;
   }
}

void u_poly(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(4);
   u = 0.0;
   for (int o = 0; o <= 2; o++)
   {
      u(0) += pow(x(0), o);
      u(1) += pow(x(1), o);
      u(2) += pow(x(0)+x(1), o);
      u(3) += pow(x(0)-x(1), o);
   }
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