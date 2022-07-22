/// Solve the steady isentropic vortex problem on a quarter annulus
// set this const expression to true in order to use entropy variables for state
constexpr bool entvar = false;

#include<random>
#include "adept.h"
#include "mfem.hpp"
#include "euler.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(0.0,1.0);

/// \brief Defines the random function for the jabocian check
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void pert(const Vector &x, Vector& p);

/// \brief Returns the value of the integrated math entropy over the domain
double calcEntropyTotalExact();

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);
mfem::Vector buildBasisCenters(int, int);
mfem::Vector buildBasisCenters2(int, int);
/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);
unique_ptr<Mesh> buildQuarterAnnulusMeshPert(int degree, int num_rad, int num_ang, double pert);
template<typename T>
void writeBasisCentervtp(const mfem::Vector &q, T& stream);

int main(int argc, char *argv[])
{
   const char *options_file = "steady_vortex_options.json";
   int myid = 0;
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 2;
   int nx = 10;
   int ny = 10;
   int numRad = 10;
   int numTheta = 10;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.AddOption(&numRad,"-br","--numrad","number of radius points");
   args.AddOption(&numTheta,"-bt","--numtheta","number of anglular points");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
  
   try
   {
      string opt_file_name(options_file);

      unique_ptr<Mesh> bmesh = buildQuarterAnnulusMeshPert(degree, numRad, numTheta,0.25);
      int numBasis = bmesh->GetNE();
      Vector center(2*numBasis);
      Vector loc(2);
      for (int k = 0; k < numBasis; k++)
      {  
         bmesh->GetElementCenter(k,loc);
         center(k*2) = loc(0);
         center(k*2+1) = loc(1);
      }

      //mfem::Vector center = buildBasisCenters2(numRad,numTheta);
      std::cout << "Number of basis is " << center.Size()/2 << std::endl;
      ofstream centerwrite("center.vtp");
      writeBasisCentervtp(center, centerwrite);
      centerwrite.close();

      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree, nx, ny);
      std::cout << "Number of elements " << smesh->GetNE() <<'\n';
      unique_ptr<AbstractSolver> solver(
         new EulerSolver<2, entvar>(opt_file_name, move(smesh)));
      solver->initDerived(center);

      solver->setInitialCondition(uexact);

      // get the initial density error
      double l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
      double res_error = solver->calcResidualNorm();
      if (0==myid)
      {
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l2_error;
         mfem::out << "\ninitial residual norm = " << res_error << endl;
      }
      // solver->checkJacobian(pert);
      solver->solveForState();
      solver->printSolution("rbf_final",0);
      solver->printError("rbf_final_error", 0, uexact);
      // get the final density error
      l2_error = (static_cast<EulerSolver<2, entvar>&>(*solver)
                            .calcConservativeVarsL2Error(uexact, 0));
      res_error = solver->calcResidualNorm();
      double drag = abs(solver->calcOutput("drag") - (-1 / mach::euler::gamma));
      double entropy = solver->calcOutput("entropy");
      std::cout << "Number of basis is " << center.Size()/2 << std::endl;

      if (0==myid)
      {
         mfem::out << "\nfinal residual norm = " << res_error;
         mfem::out << "\n|| rho_h - rho ||_{L^2} = " << l2_error << endl;
         mfem::out << "\nDrag error = " << drag << endl;
         mfem::out << "\nTotal entropy = " << entropy;
         mfem::out << "\nEntropy error = "
                   << fabs(entropy - calcEntropyTotalExact()) << endl;
      }

   }
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
}

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(4);
   for (int i = 0; i < 4; i++)
   {
      p(i) = normal_rand(gen);
   }
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
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
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

mfem::Vector buildBasisCenters(int numRad, int numTheta)
{
   double r,theta;
   int numBasis = numRad * numTheta;
   Vector basisCenter(2*numBasis);
   for (int i = 0; i < numBasis; i++)
   {
      r = 1.0 + 2.0 * normal_rand(gen);
      theta = M_PI/2.0 * normal_rand(gen);
      basisCenter(i*2) = r *cos(theta);
      basisCenter(i*2+1) = r * sin(theta);
   }
   return basisCenter;
}

mfem::Vector buildBasisCenters2(int nx, int ny)
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

      if (1.0 < dist && dist < 3.0)
      {
         cent.push_back(x);
         cent.push_back(y);
      }
   }
   cout << "cent size is " << cent.size() << '\n';
   mfem::Vector center(cent.size());

   for (int i = 0; i < cent.size()/2; i++)
   {
      center(2*i) = cent[2*i];
      center(2*i+1) = cent[2*i+1];
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