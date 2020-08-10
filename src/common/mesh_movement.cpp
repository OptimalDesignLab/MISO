#include "mesh_movement.hpp"

#include <fstream>
#include "../../build/_config.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

#ifdef MFEM_USE_PUMI
#ifdef MACH_USE_EGADS
LEAnalogySolver::LEAnalogySolver(
   const std::string &opt_file_name,
   std::unique_ptr<mfem::Mesh> smesh,
   int dim)
   : MeshMovementSolver(opt_file_name, move(smesh))
{
   //testing
   if(options["test-removed-bound"].get<bool>())
   {
      mesh->RemoveInternalBoundaries();
   }

   int fe_order = options["space-dis"]["degree"].get<int>();

   /// Create the H(Grad) finite element collection
   h_grad_coll.reset(new H1_FECollection(fe_order, dim));

   /// Create the H(Grad) finite element space
   h_grad_space.reset(new SpaceType(mesh.get(), h_grad_coll.get(), 3));

   /// Create temperature grid function
   u.reset(new GridFunType(h_grad_space.get()));

   /// Set static variables
   setStaticMembers();

   //determine the list of essential boundary dofs, in this case, all of them
   // TODO: Can we make bdr_attributes contain geometric interfaces as well?
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 1;
   h_grad_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   cout << "Number of Essential DOF's: " << ess_tdof_list.Size() << endl;

   /// zero right hand side
   bs.reset(new LinearForm(h_grad_space.get()));
   bs->Set(0, *bs);
   
   /// assign stiffness
   if(options["uniform-stiff"]["on"].template get<bool>())
   {
      double lambda = options["uniform-stiff"]["lambda"].template get<double>();
      double mu = options["uniform-stiff"]["mu"].template get<double>();
      lambda_c.reset(new ConstantCoefficient(lambda));
      mu_c.reset(new ConstantCoefficient(mu));
   }
   else
   {
      mesh_copy = mesh.get();
      lambda_c.reset(new ElementFunctionCoefficient(LambdaFunc));
      mu_c.reset(new ElementFunctionCoefficient(MuFunc));
   }

   /// for loading model files
   string model_file_old = options["model-file"].template get<string>();
   string model_file_new = options["model-file-new"].template get<string>();
   string mesh_file = options["mesh"]["file"].template get<string>();
   string tess_file = options["tess-file-old"].template get<string>();

   /// assemble stiffness matrix
   k.reset(new BilinearFormType(h_grad_space.get()));
   k->AddDomainIntegrator(new ElasticityIntegrator(*lambda_c, *mu_c));
   k->Assemble();

   /// set surface node boundary conditions
   std::cout << "Computing Boundary Node Displacements..." << std::endl;
   
   getBoundaryNodeDisplacement(model_file_old, model_file_new, tess_file, 
                              pumi_mesh.get(), &disp_list);
   /// replicating ProjectCoefficient
   int el = -1;
   const FiniteElement *fe = NULL;
   
   Vector val;
   
   for (int dof = 1; dof < disp_list.Size(); dof++)
   {
      // int j = fes->GetElementForDof(dof);
      // if (el != j)
      // {
      //    el = j;
      //    fe = fes->GetFE(el);
      // }
      // int ld = fes->GetLocalDofForDof(dof);
      // const IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      // T->SetIntPoint(&ip);
      // vcoeff.Eval(val, *T, ip);
      val = disp_list[dof];

      for (int vd = 0; vd < h_grad_space->GetVDim(); vd ++)
      {
         int vdof = h_grad_space->DofToVDof(dof-1, vd);
         (*u)(vdof) = val(vd);
      }
   }

   cout << "Number of Surface Nodes: " << disp_list.Size() << endl;
   cout << "Total Nodes: " << mesh->GetNV() << endl;


   /// set solver
   solver.reset(new CGSolver());
   prec.reset(new HypreSmoother());
   solver->iterative_mode = false;
   solver->SetRelTol(options["lin-solver"]["rel-tol"].get<double>());
   solver->SetAbsTol(options["lin-solver"]["abs-tol"].get<double>());
   solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
   solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
   solver->SetPreconditioner(*prec);

   k->FormLinearSystem(ess_tdof_list, *u, *bs, K, U, B);

   solver->SetOperator(K);
}

void LEAnalogySolver::solveSteady()
{
   solver->Mult(B, U);

   k->RecoverFEMSolution(U, *bs, *u);

   // save the displaced mesh
   {   

      // save original mesh
      ofstream mesh_ofs_1("unmoved_mesh.vtk");
      mesh_ofs_1.precision(8);
      mesh->PrintVTK(mesh_ofs_1, options["space-dis"]["degree"].get<int>());


      GridFunction *nodes = u.get();
      double coord[3];
      for(int nv = 0; nv < mesh->GetNV(); nv++)
      {
            mesh->GetNode(nv, coord);
            for (int vd = 0; vd < h_grad_space->GetVDim(); vd ++)
            {
               int vdof = h_grad_space->DofToVDof(nv, vd);
               (*nodes)(vdof) += coord[vd];
            }
      }
      int own_nodes = 0;
      mesh->SwapNodes(nodes, own_nodes);
      ofstream mesh_ofs_2("moved_mesh.vtk");
      mesh_ofs_2.precision(8);
      mesh->PrintVTK(mesh_ofs_2, options["space-dis"]["degree"].get<int>());
      //nodes->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>()); 

      //update pumi mesh and write to file
      string model_file_new = options["model-file-new"].template get<string>();
      string mesh_file_new = options["mesh"]["moved-file"].template get<string>();
      moved_mesh = getNewMesh(model_file_new, mesh_file_new, mesh.get(), pumi_mesh.get());
   }
}

double LEAnalogySolver::LambdaFunc(const mfem::Vector &x, int ie)
{
   return 1.0/mesh_copy->GetElementVolume(ie);
}

double LEAnalogySolver::MuFunc(const mfem::Vector &x, int ie)
{
   return 1.0;//mesh_copy->GetElementVolume(ie);
}

mfem::Mesh* LEAnalogySolver::mesh_copy = 0;

#endif
#endif

} //namespace mach
