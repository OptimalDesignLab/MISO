constexpr bool entvar = true;
#include "mfem.hpp"
#include "euler.hpp"
#include "RBFSpace.hpp"
#include "rbfgridfunc.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;


/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);
void upoly(const Vector &x, Vector& u);
void utest(const Vector &x, Vector &u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);
Array<Vector *> buildBasisCenters(int, int);

int main(int argc, char *argv[])
{
   const char *options_file = "rbf_test.json";
   int myid = 0;
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 1;
   int extra_basis = 1;
   int nx = 1;
   int ny = 1;
   int numRad = 10;
   int numTheta = 10;
   double sp = 1.0;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.AddOption(&numRad,"-br","--numrad","number of radius points");
   args.AddOption(&numTheta,"-bt","--numtheta","number of anglular points");
   args.AddOption(&extra_basis,"-e", "--extra", "extra number of basis");
   args.AddOption(&sp,"-s","-shape","shapeParameter in RBF kernel");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   try
   {
      // generate the mesh
      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree+1, nx, ny);
      std::cout << "Number of elements " << smesh->GetNE() <<'\n';
      int dim = smesh->Dimension();
      int num_state = dim +2;

      // initialize the basis centers
      int numBasis = smesh->GetNE();
      Array<Vector *> center(numBasis);
      for (int k = 0; k < numBasis; k++)
      {  
         center[k] = new Vector(dim);
         smesh->GetElementCenter(k,*center[k]);
      }

      // Array<Vector *> center = buildBasisCenters(numRad,numTheta);
      // int numBasis = numRad * numTheta;
      // for (int i = 0; i < numBasis; i++)
      // {
      //    cout << "basis " << i << ": ";
      //    center[i]->Print();
      // }

      // initialize the fe collection and rbf space
      DSBPCollection fec(degree,smesh->Dimension());
      RBFSpace rbfspace(smesh.get(),&fec,center,1.0,num_state,extra_basis,Ordering::byVDIM,degree);
      FiniteElementSpace fes(smesh.get(),&fec,num_state,Ordering::byVDIM);

      //================== Construct the gridfunction and apply the exact solution =======
      mfem::CentGridFunction x_cent(&rbfspace);
      mfem::GridFunction x(&fes);
      mfem::GridFunction x_exact(&fes);

      mfem::VectorFunctionCoefficient u0_fun(num_state, upoly);
      x_cent.ProjectCoefficient(u0_fun);
      x_exact.ProjectCoefficient(u0_fun);
      x = 0.0;
      ofstream x_centprint("x_cent.txt");
      x_cent.Print(x_centprint,4);
      x_centprint.close();

      //============== Prolong the solution to SBP nodes =================================
      //============== and check the simple l2 norm error ================================
      ofstream sol_ofs("rbf_test.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs,0);
      x_exact.SaveVTK(sol_ofs,"exact",0);


      rbfspace.GetProlongationMatrix()->Mult(x_cent, x);
      ofstream x_prolongprint("x_prolong.txt");
      x.Print(x_prolongprint,4);
      x_prolongprint.close();
      x_cent.Print(x_centprint,4);
      x_centprint.close();

      //============== Prolong the solution to SBP nodes =================================
      //============== and check the simple l2 norm error ======
      x.SaveVTK(sol_ofs,"prolong",0);
      x -= x_exact;
      x.SaveVTK(sol_ofs,"error",0);
      sol_ofs.close();
      cout << "Check the projection l2 error: " << x.Norml2() << '\n';
      for (int k = 0; k < numBasis; k++)
      {
         delete center[k];
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

void upoly(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(4);
   u = 0.0;
   for (int p = 0; p < 4; p++)
   {
      u(p) = pow(x(0), p);
   }
}

void utest(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(4);
   u(0) = 1.0;
   u(1) = 2.0;
   u(2) = 3.0;
   u(3) = 4.0;
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

mfem::Array<mfem::Vector *> buildBasisCenters(int numRad, int numTheta)
{
   int i,j;
   double rad_inv = 2.0/(numRad -1);
   double theta_inv = 0.5*M_PI/(numTheta - 1);

   Vector radPoints(numRad), thetaPoints(numTheta);
   for (i = 0; i < numRad; i++)
   {
      radPoints(i) = 1.0 + i * rad_inv;
   }
   for (i = 0; i < numTheta; i++)
   {
      thetaPoints(i) = theta_inv * i;
   }

   Array<Vector *> basisCenter(numRad*numTheta);

   int b_id;
   Vector b_coord(2);
   for (i = 0; i < numRad; i++)
   {
      for (j = 0; j < numTheta; j++)
      {
         b_coord(0) = radPoints(i) * cos(thetaPoints(j));
         b_coord(1) = radPoints(i) * sin(thetaPoints(j));
         b_id = i * numTheta + j;
         basisCenter[b_id] = new Vector(2);
         (*basisCenter[b_id]) = b_coord;
      }
   }

   return basisCenter;
}