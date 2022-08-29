#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "rbfgridfunc.hpp"
#include "optimization.hpp"
#include "bfgsnewton.hpp"
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
std::unique_ptr<Mesh> buildQuarterAnnulusMeshPert(int degree, int num_rad,
                                                  int num_ang, double pert);

mfem::Vector buildBasisCenter(mfem::Mesh *mesh, int numBasis);
mfem::Vector buildBasisCenter2(int numt, int numr);

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
   args.AddOption(&numRad, "-br", "--basisrad", "number of radial segments");
   args.AddOption(&numTheta, "-bt", "--basistheta", "number of angular segments");
   args.AddOption(&extra,"-e","--extra","number of anglular points");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   try
   {
      // generate the mesh
      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree + 1, nx, ny);
      ofstream savevtk("optimizationtest.vtp");
      smesh->PrintVTK(savevtk, 0);
      savevtk.close();
      std::cout << "Number of elements " << smesh->GetNE() << '\n';
      int dim = smesh->Dimension();
      int num_state = dim + 2;


      // mesh for basis
      // unique_ptr<Mesh> bmesh = buildQuarterAnnulusMeshPert(degree, numRad, numTheta,0.25);
      // int numBasis = bmesh->GetNE();
      // Vector center(dim*numBasis);
      // Vector loc(dim);
      // for (int k = 0; k < numBasis; k++)
      // {  
      //    bmesh->GetElementCenter(k,loc);
      //    center(k*2) = loc(0);
      //    center(k*2+1) = loc(1);
      // }

      Vector center = buildBasisCenter2(numRad,numTheta);
      int numBasis = center.Size()/dim;

      ofstream centerwrite("center_initial.vtk");
      writeBasisCentervtp(center, centerwrite);
      centerwrite.close();

      // initialize the optimization object
      string optfile(options_file);
      DGDOptimizer dgdopt(center,optfile,move(smesh));
      dgdopt.InitializeSolver();
      dgdopt.SetInitialCondition(uexact);

      // double l2norm = dgdopt.GetEnergy(center);
      // cout << "initial objective value is " << l2norm << '\n';
      // dgdopt.checkJacobian(center);

      //BFGSNewtonSolver bfgsSolver(0.1,1e3,1e-4,0.9,20);
      BFGSNewtonSolver bfgsSolver(optfile);
      bfgsSolver.SetOperator(dgdopt);
      Vector opti_value(center.Size());
      bfgsSolver.Mult(center,opti_value);

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
   return center;
}

mfem::Vector buildBasisCenter2(int nx, int ny)
{
   int numBasis = nx * ny;
   double dx = 3./(nx-1);
   double dy = 3./(ny-1);
   std::vector<double> cent;

   double x,y;
   int row, col;
   double dist;
   for (int i = 0; i < numBasis; i++)
   {
      row = i/ny;
      col = i%ny;

      x = row * dx;
      y = col * dy;
      dist = sqrt(pow(x,2)+ pow(y,2));

      if (1.1 < dist && dist < 2.9)
      {
         cent.push_back(x);
         cent.push_back(y);
      }
   }
   mfem::Vector center(cent.size());

   for (int i = 0; i < cent.size(); i++)
   {
      center(i) = cent[i];
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

unique_ptr<Mesh> buildQuarterAnnulusMeshPert(int degree, int num_rad, int num_ang, double pert)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));

   // Randomly perturb interior nodes
   std::default_random_engine gen(std::random_device{}());
   std::uniform_real_distribution<double> uni_rand(-pert, pert);
   static constexpr double eps = std::numeric_limits<double>::epsilon();
   for (int i = 0; i < mesh_ptr->GetNV(); ++i)
   {
      double *vertex = mesh_ptr->GetVertex(i);
      // make sure vertex is interior
      if (vertex[0] > eps && vertex[0] < 2.0 - eps && 
          vertex[1] > eps && vertex[1] < M_PI * 0.5 - eps)
      {
         // perturb coordinates 
         vertex[0] += uni_rand(gen)*2.0/num_rad;
         vertex[1] += uni_rand(gen)*M_PI * 0.5/num_ang;
      }
   }

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