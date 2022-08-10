#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "rbfgridfunc.hpp"
#include "optimization.hpp"
#include "bfgsnewton.hpp"
#include "naca12.data"
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace mfem;
using namespace mach;



std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(0.0,1.0);

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);
mfem::Vector buildBasisCenter(mfem::Mesh *mesh, int numBasis);
mfem::Vector buildBasisCenter2(int nx, int ny);
mfem::Vector buildBasisCenter3();

template<typename T>
void writeBasisCentervtp(const mfem::Vector &q, T& stream);

int main(int argc, char *argv[])
{
   const char *options_file = "wedgeshock_optimization_options.json";
   int myid = 0;
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 1;
   int nx = 10;
   int ny = 10;
   int bx = 10;
   int by = 10;
   args.AddOption(&nx, "-nx", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-ny", "--num-theta", "number of angular segments");
   args.AddOption(&bx, "-bx", "--num-rad", "number of radial segments");
   args.AddOption(&by, "-by", "--num-theta", "number of angular segments");
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
      // mesh for basis
      unique_ptr<Mesh> bmesh(new Mesh(nx,ny,Element::TRIANGLE,true,2.0,1.0,true));
      ofstream savevtk("wedgeshock_basis.vtk");
      bmesh->PrintVTK(savevtk, 0);
      savevtk.close();
      int dim = bmesh->Dimension();
      int num_state = dim + 2;

      // initialize the basis center (design variables)
      int numBasis = bmesh->GetNE();
      Vector center = buildBasisCenter(bmesh.get(),numBasis);
      ofstream centerwrite("center_initial.vtp");
      writeBasisCentervtp(center, centerwrite);
      centerwrite.close();

      unique_ptr<Mesh> smesh(new Mesh(nx,ny,Element::TRIANGLE,true,2.0,1.0,true));

      // initialize the optimization object
      string optfile(options_file);
      DGDOptimizer dgdopt(center,optfile,move(smesh));
      dgdopt.InitializeSolver();
      dgdopt.SetInitialCondition(uexact);
      dgdopt.printSolution(center,"wedgeshock-initial");
      //dgdopt.checkJacobian(center);

      //BFGSNewtonSolver bfgsSolver(1.0,1e6,1e-4,0.7,40);
      BFGSNewtonSolver bfgsSolver(optfile);
      bfgsSolver.SetOperator(dgdopt);
      Vector opti_value(center.Size());
      bfgsSolver.Mult(center,opti_value);

      dgdopt.printSolution(opti_value,"wedgeshock-final");

      ofstream optwrite("center_optimal.vtp");
      writeBasisCentervtp(opti_value, optwrite);
      optwrite.close();
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

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));
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
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}

// Exact solution; 
void uexact(const Vector &x, Vector& u)
{
   u.SetSize(4);
   double Mai = 2.4; //Ma1
   double rhoi = 1.0; //rho1
   double prsi = 1.0/euler::gamma;
   //assuming theta = 25 degrees, Ma1 = 2.4
   double theta = 25*2*M_PI/360;
   double beta =  52.17187440*2*M_PI/360; 
   //taken from Figure 9.9, Anderson for theta = 25 degrees, Ma1 = 2.4
   
   //compute mach number downstream of shock
   double Ma1n = Mai*sin(beta);
   double Ma2n = sqrt((1+(.5*euler::gami)*Ma1n*Ma1n) /
                     (euler::gamma*Ma1n*Ma1n - .5*euler::gami));
   double Ma = Ma2n/sin(beta-theta);
   
   //compute other quantities using continuity, momentum, and energy equations
   double rho = rhoi*(euler::gamma+1)*Ma1n*Ma1n / 
                  (2+euler::gami*Ma1n*Ma1n);
   double press = prsi*(1 + (2*euler::gamma/(euler::gamma+1))*(Ma1n*Ma1n - 1)); 
   double a = sqrt(euler::gamma*press/rho);

   
   double thresh = x(1)/tan(beta); //assuming wedge tip is origin
   // if behind shock, set back to upstream state
   if(x(0) <= thresh+.5)
   {
      theta = 0;
      Ma = Mai;
      rho = rhoi;
      press = prsi;
      a = sqrt(euler::gamma*press/rho);
   }

   u(0) = rho;
   u(1) = rho*a*Ma*cos(theta);
   u(2) = rho*a*Ma*sin(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

// build the basis center
mfem::Vector buildBasisCenter(mfem::Mesh *mesh, int numBasis)
{
   Vector center(2*numBasis);
   Vector loc(2);
   for (int k = 0; k < numBasis; k++)
   {
      mesh->GetElementCenter(k,loc);
      center(k*2) = loc(0);
      center(k*2+1) = loc(1);
   }
   return center;
}

mfem::Vector buildBasisCenter2(int nx, int ny)
{
   int numBasis = nx * ny;
   double dx = 60./(nx-1);
   double dy = 60./(ny-1);
   std::vector<double> cent;

   double x,y;
   int row, col;
   for (int i = 0; i < numBasis; i++)
   {
      row = i/ny;
      col = i%ny;

      x = -30. + row * dx;
      y = -30. + col * dy;
      if (sqrt(pow(x,2)+pow(y,2)) < 30.0)
      {
         cent.push_back(x);
         cent.push_back(y);
      }
   }
   mfem::Vector center(cent.size());

   for (int i = 0; i < cent.size()/2; i++)
   {
      center(2*i) = cent[2*i];
      center(2*i+1) = cent[2*i+1];
   }
   return center;
}

mfem::Vector buildBasisCenter3()
{
   mfem::Vector center(132*2);
   for (int i = 0; i < 132*2; i++)
   {
      center(i) = naca12[i];
   }
   return center;
}

template <typename T>
void writeBasisCentervtp(const mfem::Vector &center, T &stream)
{
   int nb = center.Size()/2;
   stream << "<?xml version=\"1.0\"?>\n";
   stream << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
   stream << "<PolyData>\n";
   stream << "<Piece NumberOfPoints=\"" << nb << "\" NumberOfVerts=\"" << nb << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
   stream << "<Points>\n";
   stream << "  <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";
   for (int i = 0; i < nb; i++)
   {
      stream << center(i*2) << ' ' << center(i*2+1) << ' ' << 0.0 << ' ';
   }
   stream << "</DataArray>\n";
   stream << "</Points>\n";
   stream << "<Verts>\n";
   stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
   for (size_t i = 0; i < nb; ++i)
      stream << i << ' ';
   stream << "</DataArray>\n";
   stream << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
   for (size_t i = 1; i <= nb; ++i)
      stream << i << ' ';
   stream << "</DataArray>\n";
   stream << "</Verts>\n";
   stream << "<PointData Scalars=\"w\">\n";
   stream << "  <DataArray type=\"Float32\" Name=\"w\" NumberOfComponents=\"1\" format=\"ascii\">";
   for (int i = 0; i < nb; i++)
      stream << 1.0 << ' ';
   stream << "</DataArray>\n";
   stream << "</PointData>\n";
   stream << "</Piece>\n";
   stream << "</PolyData>\n";
   stream << "</VTKFile>\n";
}
