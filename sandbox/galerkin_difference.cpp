/// Solve the steady isentropic vortex problem on a quarter annulus
#include<random>
//#include "adept.h"

#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "centgridfunc.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using namespace mach;
using namespace apf;

const int degree = 2;

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const mfem::Vector &x, mfem::Vector &u);
void uexact_single(const mfem::Vector &x, mfem::Vector &u);

// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
   // the specfic option file for this problem
   const char *options_file = "galerkin_difference.json";
   nlohmann::json options;
   ifstream option_source(options_file);
   option_source >> options;
   option_source.close();

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
   int nEle; // number of element in the pumi mesh
   // Parse command-line options
   OptionsParser args(argc, argv);
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
      PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
      Sim_readLicenseFile(0);
      gmi_sim_start();
      gmi_register_sim();
#endif
      gmi_register_mesh();

      apf::Mesh2 *pumi_mesh;
      pumi_mesh = apf::loadMdsMesh(options["model-file"].get<string>().c_str(),
                           options["pumi-mesh"]["file"].get<string>().c_str());
      pumi_mesh->verify();
      

      dim = pumi_mesh->getDimension();
      nEle = pumi_mesh->count(dim);

      // for(int degree = 0; degree < 4; degree++)
      // {
      //    GalerkinDifference gd(degree, pumi_mesh, fec.get(), 1, Ordering::byVDIM);
      //    gd.BuildGDProlongation();
      //    mfem::GridFunction x(&gd);
      //    mfem::GridFunction x_exact(&gd);

      // }
      
      
      cout << "Construct the GD fespace.\n";
      GalerkinDifference gd(degree, pumi_mesh, 1, Ordering::byVDIM);
      cout << "Now build the prolongation matrix.\n";
      //GalerkinDifference gd(options_file, pumi_mesh);
      gd.BuildGDProlongation();


      // Test the prolongation matrix with gridfunction vdim = 4
      mfem::GridFunction x(&gd);
      mfem::GridFunction x_exact(&gd);
      cout << "Size of x and x_exact is " << x.Size() << '\n';

      mfem::VectorFunctionCoefficient u0(1, uexact);
      x_exact.ProjectCoefficient(u0);
      cout << "Check the exact solution:\n";
      x_exact.Print(cout ,4);


      mfem::CentGridFunction x_cent(&gd);
      cout << "Size of x_cent is " << x_cent.Size() << '\n';
      x_cent.ProjectCoefficient(u0);
      cout << "\n\n\n\nCheck the the center values:\n";
      x_cent.Print(cout, 4);


      gd.GetProlongationMatrix()->Mult(x_cent, x);
      cout << "\n\n\n\nCheck the results:\n";
      x.Print(cout,4);
      x -= x_exact;
      cout << "Check the error: " << x.Norml2() << '\n';

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
      PCU_Comm_Free();
      
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
   u(0) = 0;
   for(int i = degree; i >= 0; i--)
   {
      u(0) += pow(x(0)+x(1), i);
   }
   // different degree 2d polynomial to test the acccuracy
   // u(0) = 1.0;  // constant
   // u(1) = x(0); // linear function
   // u(2) = x(0) * x(0) - x(0) * x(1) + x(1) * x(1); // quadrature function
   // u(3) = x(0) * x(0) * x(1) - 2.0 * x(0) * x(1) + 3.0 * x(1) * x(1) * x(1); // cubic function
   
   // steady vortext exact solution with shifted coordinate
   // u.SetSize(4);
   // double ri = 1.0;
   // double Mai = 0.5; //0.95 
   // double rhoi = 2.0;
   // double prsi = 1.0/euler::gamma;
   // double rinv = ri/sqrt((x(0)+2.0)*(x(0)+2.0) + (x(1)+1.0)*(x(1)+1.0));
   // double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
   //                       1.0/euler::gami);
   // double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
   //                  (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   // double theta;
   // if ((x(0)+2.0) > 1e-15)
   // {
   //    theta = atan((x(1)+1.0)/(x(0)+2.0));
   // }
   // else
   // {
   //    theta = M_PI/2.0;
   // }
   // double press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
   //               (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   // double a = sqrt(euler::gamma*press/rho);

   // u(0) = rho;
   // u(1) = rho*a*Ma*sin(theta);
   // u(2) = -rho*a*Ma*cos(theta);
   // u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

void uexact_single(const mfem::Vector &x, mfem::Vector &u)
{
   u(0) = x(0);
   //u(0) = x(0) * x (0) - 7.0 * x(0) + 3.0;
}
