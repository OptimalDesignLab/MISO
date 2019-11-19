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


int main(int argc, char *argv[])
{
#ifdef MFEM_USE_MPI
   // Initialize MPI if parallel
   MPI_Init(&argc, &argv);
#endif
   // Parse command-line options
   const char *mesh_file = "../../cad/wedge1.smb";
#ifdef MFEM_USE_SIMMETRIX
   const char *model_file = "../../cad/wedge1.x_t";
#else
   const char *model_file = "../../cad/wedge1.dmg";
#endif
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

      // read the SCOREC Mesh.
      PCU_Comm_Init();
   #ifdef MFEM_USE_SIMMETRIX
      Sim_readLicenseFile(0);
      gmi_sim_start();
      gmi_register_sim();
   #endif
      gmi_register_mesh();

      apf::Mesh2* pumi_mesh;
      pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);


      pumi_mesh->verify();

      // create the MFEM mesh object from the PUMI mesh
      unique_ptr<Mesh> mesh = unique_ptr<Mesh>(new PumiMesh(pumi_mesh, 1, 1));

      // reorder boundary faces, just in case
      apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
      apf::MeshEntity* ent ;
      int ent_cnt = 0;
      while ((ent = pumi_mesh->iterate(itr)))
      {
         apf::ModelEntity *me = pumi_mesh->toModel(ent);
         if (pumi_mesh->getModelType(me) == (dim-1))
         {
            //Get tag from model by  reverse classification
            int tag = pumi_mesh->getModelTag(me);
            (mesh->GetBdrElement(ent_cnt))->SetAttribute(ent_cnt);
            ent_cnt++;
         }
      }
      pumi_mesh->end(itr);  
      mesh->SetAttributes();
      pumi_mesh->destroyNative();
      apf::destroyMesh(pumi_mesh);
      PCU_Comm_Free();

      EulerSolver solver(opt_file_name, move(mesh), dim);
      solver.setInitialCondition(uexact);
      //solver.printSolution("init", degree+1);
      // mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
      //           << solver.calcL2Error(uexact, 0) << '\n' << endl;
      // mfem::out << "\ninitial residual norm = " << solver.calcResidualNorm()
      //           << endl;
      solver.solveForState();
      // mfem::out << "\nfinal residual norm = " << solver.calcResidualNorm()
      //           << endl;
      // mfem::out << "\n|| rho_h - rho ||_{L^2} = " 
      //           << solver.calcL2Error(uexact, 0) << '\n' << endl;

      // ofstream fout("wedge1_sol.vtk");
      // fout.precision(8); 
      // //mesh->Print(fout);
      // mesh->PrintVTK(fout, 0);
      // GridFunction x = Tesla.GetVectorPotential();
      // x.SaveVTK(mesh_ofs, "sol", 0);

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
void uexact(const Vector &x, Vector& u)
{
   u.SetSize(4);
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
   u(1) = rho*a*Ma*sin(theta);
   u(2) = -rho*a*Ma*cos(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;
}

