#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "rbfgridfunc.hpp"
#include "optimization.hpp"
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

template<typename T>
void writeBasisCentervtp(const mfem::Vector &q, T& stream);

int main(int argc, char *argv[])
{
   const char *options_file = "optimizationtest_options.json";
   int myid = 0;
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 1;
   int nx = 1;
   int ny = 1;
   int numRad = 10;
   int numTheta = 10;
   int extra = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.AddOption(&extra,"-e","--extra","number of anglular points");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }

   // generate the mesh
   unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree + 1, nx, ny);
   ofstream savevtk("optimizationtest.vtk");
   smesh->PrintVTK(savevtk, 0);
   savevtk.close();
   std::cout << "Number of elements " << smesh->GetNE() << '\n';
   int dim = smesh->Dimension();
   int num_state = dim + 2;


   // initialize the basis center (design variables)
   int numBasis = smesh->GetNE();
   Vector center = buildBasisCenter(smesh.get(),numBasis);

   // initialize the optimization object
   string optfile(options_file);
   DGDOptimizer dgdopt(center,optfile,move(smesh));
   dgdopt.InitializeSolver();
   dgdopt.SetInitialCondition(uexact);

   double l2norm = dgdopt.GetEnergy(center);
   cout << "initial l2 norm is " << l2norm << '\n';
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

// the exact solution
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
   q = u;
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
   ofstream centerwrite("center.vtp");
   writeBasisCentervtp(center, centerwrite);
   centerwrite.close();
   return center;
}


template <typename T>
void writeBasisCentervtp(const mfem::Vector &center, T &stream)
{
   stream << "<?xml version=\"1.0\"?>\n";
   stream << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
   stream << "<PolyData>\n";
   stream << "<Piece NumberOfPoints=\"" << center.Size() << "\" NumberOfVerts=\"" << center.Size() << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
   stream << "<Points>\n";
   stream << "  <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";
   int numBasis = center.Size()/2;
   for (int i = 0; i < numBasis; i++)
   {
      stream << center(2*i) << ' ' << center(2*i+1) << ' ' << 0.0 << ' ';
   }
   stream << "</DataArray>\n";
   stream << "</Points>\n";
   stream << "<Verts>\n";
   stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
   for (size_t i = 0; i < center.Size(); ++i)
      stream << i << ' ';
   stream << "</DataArray>\n";
   stream << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
   for (size_t i = 1; i <= center.Size(); ++i)
      stream << i << ' ';
   stream << "</DataArray>\n";
   stream << "</Verts>\n";
   stream << "<PointData Scalars=\"w\">\n";
   stream << "  <DataArray type=\"Float32\" Name=\"w\" NumberOfComponents=\"1\" format=\"ascii\">";
   for (int i = 0; i < center.Size(); i++)
      stream << 1.0 << ' ';
   stream << "</DataArray>\n";
   stream << "</PointData>\n";
   stream << "</Piece>\n";
   stream << "</PolyData>\n";
   stream << "</VTKFile>\n";
}