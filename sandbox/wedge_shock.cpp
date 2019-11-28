/// Solve the wedge shock problem on a 2D mesh, exploiting symmetry

#include "mfem.hpp"
#include "euler.hpp"
#include "euler_fluxes.hpp"
#include <fstream>
#include <iostream>
#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief Defines the exact solution for the wedge shock problem
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);
void uinit(const Vector&x, Vector& u);

int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
#endif
   // Parse command-line options
//    const char *mesh_file = "../../cad/wedge1.smb";
// #ifdef MFEM_USE_SIMMETRIX
//    const char *model_file = "../../cad/wedge1.x_t";
// #else
//    const char *model_file = "../../cad/wedge1.dmg";
// #endif
   OptionsParser args(argc, argv);
   const char *options_file = "wedge_shock_options.json";
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
      // construct the solver, set the initial condition, and solve
      string opt_file_name(options_file);
      const int dim = 2;
   //    // reorder boundary faces, just in case
   //    apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
   //    apf::MeshEntity* ent ;
   //    int ent_cnt = 0;
   //    while ((ent = pumi_mesh->iterate(itr)))
   //    {
   //       apf::ModelEntity *me = pumi_mesh->toModel(ent);
   //       if (pumi_mesh->getModelType(me) == (dim-1))
   //       {
   //          //Get tag from model by  reverse classification
   //          int tag = pumi_mesh->getModelTag(me);
   //          (mesh->GetBdrElement(ent_cnt))->SetAttribute(ent_cnt);
   //          ent_cnt++;
   //       }
   //    }
   //    pumi_mesh->end(itr);  
   //    mesh->SetAttributes();
   //    pumi_mesh->destroyNative();
   //    apf::destroyMesh(pumi_mesh);
   //    PCU_Comm_Free();

      EulerSolver solver(opt_file_name, nullptr, dim);
      solver.setInitialCondition(uexact);
      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver.calcL2Error(uexact, 0) << '\n' << endl;
      mfem::out << "\ninitial residual norm = " << solver.calcResidualNorm()
                << endl;
      solver.solveForState();
      mfem::out << "\nfinal residual norm = " << solver.calcResidualNorm()
                << endl;
      mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
                << solver.calcL2Error(uexact, 0) << '\n' << endl;

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

// Exact solution; 
void uinit(const Vector &x, Vector& u)
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
   //if(x(0) <= thresh+.5)
   {
      theta = 0;
      Ma = Mai;
      rho = rhoi;
      press = prsi;
      a = sqrt(euler::gamma*press/rho);
   }

   u(0) = .99*rho;
   u(1) = .99*rho*a*Ma*cos(theta);
   u(2) = .99*rho*a*Ma*sin(theta);
   u(3) = .99*(press/euler::gami + 0.5*rho*a*a*Ma*Ma);
   
   // u(0) = 1;
   // u(1) = 0;
   // u(2) = 0;
   // u(3) = 1;
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
   
   // u(0) = 1;
   // u(1) = 0;
   // u(2) = 0;
   // u(3) = 1;
}

