/// Solve the steady isentropic vortex problem on a quarter annulus
#include <fstream>
#include <iostream>

#include "mfem.hpp"
#include "surface.hpp"

int main(int argc, char *argv[])
{
   using namespace std;
   using namespace mfem;
   using namespace mach;

   // Initialize MPI
   MPI_Init(&argc, &argv);

   // import the airfoil mesh
   const char *mesh_file = "surface_distance.mesh";
   Mesh mesh(mesh_file, 1, 1);

   // Construct the surface object
   Array<int> bdr_attr_marker(4);
   bdr_attr_marker[0] = 0;
   bdr_attr_marker[1] = 0;
   bdr_attr_marker[2] = 1;
   bdr_attr_marker[3] = 1;
   Surface<2> surf(mesh, bdr_attr_marker);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto dist_fun = [&surf](const Vector &x)
   {
      return surf.calcDistance(x);
   };

   // Needed just to construct the FunctionCoefficient
   auto dist_fun_rev = [](const Vector &x, double dist_bar, Vector &x_bar)
   {
   };

   // 
   int order = 1;
   int dim = 2;
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   GridFunction u(&fes);
   FunctionCoefficient dist(dist_fun, dist_fun_rev);
   u.ProjectCoefficient(dist);

   ofstream sol_ofs("surface_distance.vtk");
   sol_ofs.precision(14);
   mesh.PrintVTK(sol_ofs, 1);
   u.SaveVTK(sol_ofs, "distance", 1);
   sol_ofs.close();
}
