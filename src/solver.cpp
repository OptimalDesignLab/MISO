#include <fstream>
#include <iostream>
#include "solver.hpp"
#include "default_options.hpp"
#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#ifdef MFEM_USE_PUMI
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>
#endif
#include "sbp_fe.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

AbstractSolver::AbstractSolver(const string &opt_file_name)
{
   // Set the options; the defaults are overwritten by the values in the file
   // using the merge_patch method
   options = default_options;
   nlohmann::json file_options;
   ifstream options_file(opt_file_name);
   options_file >> file_options;
   options.merge_patch(file_options);
   cout << setw(3) << options << endl;
   ConstructMesh();
   int dim = mesh->Dimension();

   cout << "problem space dimension = " << dim << endl;

   // Define the ODE solver used for time integration (possibly not used)
   ode_solver = NULL;
   cout << "ode-solver type = " << options["ode-solver"].get<string>() << endl;
   if (options["ode-solver"].get<string>() == "RK1")
   {
      ode_solver.reset(new ForwardEulerSolver);
   }
   if (options["ode-solver"].get<string>() == "RK4")
   {
      ode_solver.reset(new RK4Solver);
   }
   else
   {
      throw MachException("Unknown ODE solver type " +
                          options["ode-solver"].get<string>());
      // TODO: parallel exit
   }

   // Refine the mesh here, or have a separate member function?
   for (int l = 0; l < options["mesh"]["refine"].get<int>(); l++)
   {
      mesh->UniformRefinement();
   }

   // Define the SBP elements and finite-element space; eventually, we will want
   // to have a case or if statement here for both CSBP and DSBP, and (?) standard FEM.
   fec.reset(new SBPCollection(options["degree"].get<int>(), dim));
}

AbstractSolver::~AbstractSolver() 
{
   cout << "Deleting Abstract Solver..." << endl;
}

void AbstractSolver::setInitialCondition(
   void (*u_init)(const Vector &, Vector &))
{
   // TODO: Need to verify that this is ok for scalar fields
   VectorFunctionCoefficient u0(num_state, u_init);
   u->ProjectCoefficient(u0);
}

double AbstractSolver::calcL2Error(
   void (*u_exact)(const Vector &, Vector &))
{
   // TODO: need to generalize to parallel
   VectorFunctionCoefficient exsol(num_state, u_exact);
   //return u->ComputeL2Error(ue);

   double loc_norm = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
      T = fes->GetElementTransformation(i);
      u->GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      vals.Norm2(loc_errs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         loc_norm += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
      }
   }
   double norm;
   #ifdef MFEM_USE_MPI
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
   #else
   norm = loc_norm;
   #endif
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

void AbstractSolver::solveForState()
{
   // TODO: This is not general enough.

   double t = 0.0;
   evolver->SetTime(t);
   ode_solver->Init(*evolver);

   // output the mesh and initial condition
   // TODO: need to swtich to vtk for SBP
   int precision = 8;
   {
      ofstream omesh("adv.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("adv-init.gf");
      osol.precision(precision);
      u->Save(osol);
   }

   bool done = false;
   double t_final = options["t-final"].get<double>();
   double dt = options["dt"].get<double>();
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(*u, t, dt_real);
      ti++;

      done = (t >= t_final - 1e-8*dt);

/*       if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (visualization)
         {
            sout << "solution\n" << mesh << u << flush;
         }

         if (visit)
         {
            dc->SetCycle(ti);
            dc->SetTime(t);
            dc->Save();
         }
      } */
   }

   // Save the final solution. This output can be viewed later using GLVis:
   // glvis -m unitGridTestMesh.msh -g adv-final.gf".
   {
      ofstream osol("adv-final.gf");
      osol.precision(precision);
      u->Save(osol);
   }
}

void AbstractSolver::ConstructMesh()
{
   #ifdef MFEM_USE_MPI
   comm = MPI_COMM_WORLD; // TODO: how to pass as an argument?
   MPI_Comm_rank(comm, &rank);
#ifdef MFEM_USE_PUMI  // if using pumi mesh
   // problem with using these in loadMdsMesh
   const char *model_file = options["model-file"].get<string>().c_str();
   const char *mesh_file= options["mesh-file"].get<string>().c_str();
   PCU_Comm_Init();
   #ifdef MFEM_USE_SIMMETRIX
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
   #endif
   gmi_register_mesh();

   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(options["model-file"].get<string>().c_str(), options["mesh-file"].get<string>().c_str());
   int dim = pumi_mesh->getDimension();
   int nEle = pumi_mesh->count(dim);
   int ref_levels = (int)floor(log(10000./nEle)/log(2.)/dim);
   // Perform Uniform refinement
   // if (ref_levels > 1)
   // {
   //    ma::Input* uniInput = ma::configureUniformRefine(pumi_mesh, ref_levels);
   //    ma::adapt(uniInput);
   // }
   pumi_mesh->verify();
   mesh.reset(new MeshType(comm, pumi_mesh));
   PCU_Comm_Free();
   #ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
   #endif
#else
   //Read the mesh from the given mesh file
   Mesh *smesh = new Mesh(options["mesh-file"].get<string>().c_str(), 1, 1);
   mesh.reset(new MeshType(comm, *smesh));
#endif //MFEM_USE_PUMI
#else
   mesh.reset(new MeshType(options["mesh-file"].get<string>().c_str(), 1, 1));
#endif //MFEM_USE_MPI
}

} // namespace mach
