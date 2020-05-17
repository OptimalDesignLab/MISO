#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "json.hpp"

#include "pfem_extras.hpp"

#include "magnetostatic.hpp"
#include "electromag_integ.hpp"
#include "res_integ.hpp"

#include "electromag_test_data.hpp"

namespace
{

using namespace mach;

void getDivFreeCurrentMeshSens(mfem::ParFiniteElementSpace *mesh_fes,
                               mfem::ParFiniteElementSpace *h1_fes,
                               mfem::ParFiniteElementSpace *nd_fes,
                               const int p,
                               mfem::VectorFunctionCoefficient &current,
                               const mfem::ParGridFunction &adj,
                               mfem::Vector& mesh_sens)
{
   Array<int> ess_bdr, ess_bdr_tdofs;
   ess_bdr.SetSize(h1_fes->GetParMesh()->bdr_attributes.Max());
   ess_bdr = 1;
   h1_fes->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

   // int irOrder = h1_fes->GetElementTransformation(0)->OrderW()
   //                + 2 * p;
   // int geom = h1_fes->GetFE(0)->GetGeomType();
   // const IntegrationRule *ir = &IntRules.Get(geom, irOrder);

   /// compute \psi_k
   /// D \psi_k = G^T M^T \psi_j (\psi_j = -\psi_A)
   ParBilinearForm h_curl_mass(nd_fes);
   h_curl_mass.AddDomainIntegrator(new VectorFEMassIntegrator);
   // assemble mass matrix
   h_curl_mass.Assemble();
   h_curl_mass.Finalize();

   ParGridFunction MTpsi_j(nd_fes);
   MTpsi_j = 0.0;
   h_curl_mass.MultTranspose(adj, MTpsi_j);
   MTpsi_j *= -1.0; // (\psi_j = -\psi_A)

   mfem::common::ParDiscreteGradOperator grad(h1_fes, nd_fes);
   grad.Assemble();
   grad.Finalize();

   ParGridFunction GTMTpsi_j(h1_fes);
   GTMTpsi_j = 0.0;
   grad.MultTranspose(MTpsi_j, GTMTpsi_j);

   ParBilinearForm D(h1_fes);
   D.AddDomainIntegrator(new DiffusionIntegrator);
   D.Assemble();
   D.Finalize();
   
   // auto *Dmat = D.ParallelAssemble();
   auto *Dmat = new HypreParMatrix;

   ParGridFunction psi_k(h1_fes);
   psi_k = 0.0;
   {
      Vector PSIK;
      Vector RHS;
      D.FormLinearSystem(ess_bdr_tdofs, psi_k, GTMTpsi_j, *Dmat, PSIK, RHS);
      /// Diffusion matrix is symmetric, no need to transpose
      // auto *DmatT = Dmat->Transpose();
      HypreBoomerAMG amg(*Dmat);
      amg.SetPrintLevel(0);
      HypreGMRES gmres(*Dmat);
      gmres.SetTol(1e-14);
      gmres.SetMaxIter(200);
      gmres.SetPrintLevel(-1);
      gmres.SetPreconditioner(amg);
      gmres.Mult(RHS, PSIK);

      D.RecoverFEMSolution(PSIK, GTMTpsi_j, psi_k);
   }

   /// compute k
   ParMixedBilinearForm weakDiv(nd_fes, h1_fes);
   weakDiv.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
   weakDiv.Assemble();
   weakDiv.Finalize();

   ParGridFunction j(nd_fes);
   j.ProjectCoefficient(current);

   ParGridFunction Wj(h1_fes);
   Wj = 0.0;
   weakDiv.Mult(j, Wj);

   ParGridFunction k(h1_fes);
   k = 0.0;
   {
      Vector K;
      Vector RHS;
      D.FormLinearSystem(ess_bdr_tdofs, k, Wj, *Dmat, K, RHS);

      HypreBoomerAMG amg(*Dmat);
      amg.SetPrintLevel(0);
      HypreGMRES gmres(*Dmat);
      gmres.SetTol(1e-14);
      gmres.SetMaxIter(200);
      gmres.SetPrintLevel(-1);
      gmres.SetPreconditioner(amg);
      gmres.Mult(RHS, K);

      D.RecoverFEMSolution(K, Wj, k);
   }

   ParLinearForm Rk_mesh_sens(mesh_fes);
   /// add integrators R_k = Dk - Wj = 0
   /// \psi_k^T Dk
   ConstantCoefficient one(1.0);
   Rk_mesh_sens.AddDomainIntegrator(
      new DiffusionResIntegrator(one, &k, &psi_k));
   /// -\psi_k^T W j 
   Rk_mesh_sens.AddDomainIntegrator(
      new VectorFEWeakDivergencedJdXIntegrator(&j, &psi_k, &current, -1.0));
   Rk_mesh_sens.Assemble();

   ParLinearForm Rj_mesh_sens(mesh_fes);
   /// Add integrators R_{\hat{j}} = \hat{j} - MGk - Mj = 0
   ParGridFunction Gk(nd_fes);
   Gk = 0.0;
   grad.Mult(k, Gk);

   /// NOTE: Not using -1.0 here even though there are - signs in the residual
   /// because we're using adj, not psi_j, which would be -adj
   Rj_mesh_sens.AddDomainIntegrator(
      new VectorFEMassdJdXIntegerator(&Gk, &adj));
   Rj_mesh_sens.AddDomainIntegrator(
      new VectorFEMassdJdXIntegerator(&j, &adj, &current));
   Rj_mesh_sens.Assemble();

   mesh_sens.Add(1.0, *Rk_mesh_sens.ParallelAssemble());
   mesh_sens.Add(1.0, *Rj_mesh_sens.ParallelAssemble());   
}

}

TEST_CASE("Divergence free projection mesh sensitivities")
{
   using namespace mfem;
   using namespace electromag_data;
   using namespace mach;

   const int dim = 3;
   const double delta = 1e-5;
   const double fd_delta = 1e-7;

   int mesh_el = 8;

   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,1);
         ParMesh mesh(MPI_COMM_WORLD, *init_mesh);
         mesh.EnsureNodes();
         auto *mesh_fes = static_cast<ParFiniteElementSpace*>(
                                             mesh.GetNodes()->FESpace());


         // get the finite-element space for the current grid function
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> o_nd_fes(
            new ParFiniteElementSpace(&mesh, nd_fec.get()));
            
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> h1_fec(
            new H1_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> o_h1_fes(
               new ParFiniteElementSpace(&mesh, h1_fec.get()));
         
         ParGridFunction adj(o_nd_fes.get());
         VectorFunctionCoefficient v_rand(dim, randState);
         adj.ProjectCoefficient(v_rand);

         VectorFunctionCoefficient current(3, func, funcRevDiff);

         ParGridFunction mesh_sens(mesh_fes);
         mesh_sens = 0.0;
         getDivFreeCurrentMeshSens(mesh_fes, o_h1_fes.get(), o_nd_fes.get(), p,
                                   current, adj, mesh_sens);

         ParGridFunction v(mesh_fes);
         v.ProjectCoefficient(v_rand);

         double dJdX_v = mesh_sens * v;

         int irOrder = o_h1_fes->GetElementTransformation(0)->OrderW()
                        + 2 * p;

         /// now compute centered difference difference
         double dJdX_v_cd = 0.0;
         // back step
         {
            auto back_mesh = getMesh(mesh_el,1);
            ParMesh b_mesh(MPI_COMM_WORLD, *back_mesh);
            b_mesh.EnsureNodes();
            auto *b_mesh_nodes = static_cast<ParGridFunction*>(b_mesh.GetNodes());

            b_mesh_nodes->Add(-delta, v);
            back_mesh->SetNodes(*b_mesh_nodes);
            std::unique_ptr<ParFiniteElementSpace> b_nd_fes(
                  new ParFiniteElementSpace(&b_mesh, nd_fec.get()));
            std::unique_ptr<ParFiniteElementSpace> b_h1_fes(
                  new ParFiniteElementSpace(&b_mesh, h1_fec.get()));

            auto b_div_free_proj = mfem::common::DivergenceFreeProjector(
                                             *b_h1_fes,
                                             *b_nd_fes,
                                             irOrder,
                                             NULL, NULL,
                                             NULL);
            ParGridFunction j(b_nd_fes.get());
            j = 0.0;
            j.ProjectCoefficient(current);
            ParGridFunction div_free_j(b_nd_fes.get());
            div_free_j = 0.0;
            b_div_free_proj.Mult(j, div_free_j);

            ParBilinearForm h_curl_mass(b_nd_fes.get());
            h_curl_mass.AddDomainIntegrator(new VectorFEMassIntegrator);
            // assemble mass matrix
            h_curl_mass.Assemble();
            h_curl_mass.Finalize();
            ParGridFunction current_vec(b_nd_fes.get());
            current_vec = 0.0;
            h_curl_mass.AddMult(div_free_j, current_vec);

            dJdX_v_cd -= adj*current_vec;
         }

         // forward step
         {
            auto for_mesh = getMesh(mesh_el,1);
            ParMesh f_mesh(MPI_COMM_WORLD, *for_mesh);
            f_mesh.EnsureNodes();
            auto *f_mesh_nodes = static_cast<ParGridFunction*>(f_mesh.GetNodes());

            f_mesh_nodes->Add(delta, v);
            for_mesh->SetNodes(*f_mesh_nodes);
            std::unique_ptr<ParFiniteElementSpace> f_nd_fes(
                  new ParFiniteElementSpace(&f_mesh, nd_fec.get()));
            std::unique_ptr<ParFiniteElementSpace> f_h1_fes(
                  new ParFiniteElementSpace(&f_mesh, h1_fec.get()));

            auto f_div_free_proj = mfem::common::DivergenceFreeProjector(
                                             *f_h1_fes,
                                             *f_nd_fes,
                                             irOrder,
                                             NULL, NULL,
                                             NULL);
            ParGridFunction j(f_nd_fes.get());
            j = 0.0;
            j.ProjectCoefficient(current);
            ParGridFunction div_free_j(f_nd_fes.get());
            div_free_j = 0.0;
            f_div_free_proj.Mult(j, div_free_j);

            ParBilinearForm h_curl_mass(f_nd_fes.get());
            h_curl_mass.AddDomainIntegrator(new VectorFEMassIntegrator);
            // assemble mass matrix
            h_curl_mass.Assemble();
            h_curl_mass.Finalize();
            ParGridFunction current_vec(f_nd_fes.get());
            current_vec = 0.0;
            h_curl_mass.AddMult(div_free_j, current_vec);

            dJdX_v_cd += adj*current_vec;     
         }
         dJdX_v_cd /= (2*delta);

         // std::cout << "dJdX_v: " << dJdX_v << "\n";
         // std::cout << "dJdX_v_cd: " << dJdX_v_cd << "\n";
         REQUIRE(dJdX_v == Approx(dJdX_v_cd).margin(1e-10));

      }
   }
}

TEST_CASE("Divergence free projection mesh sensitivities - Solver Version")
{
   using namespace mfem;
   using namespace electromag_data;
   using namespace mach;

   const int dim = 3;
   const double delta = 1e-5;

   int mesh_el = 8;

   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,2);
         init_mesh->EnsureNodes();
         init_mesh->RemoveInternalBoundaries();

         nlohmann::json options = getBoxOptions(p);
         // generate initial tet mesh
         mach::MagnetostaticSolver solver(options, move(init_mesh));
         solver.initDerived();

         auto *mesh = solver.getMesh();

         // extract mesh nodes and get their finite-element space
         // GridFunction *x_nodes = mesh->GetNodes();
         // FiniteElementSpace *mesh_fes = x_nodes->FESpace();
         auto *mesh_fes = static_cast<ParFiniteElementSpace*>(
                                                   mesh->GetNodes()->FESpace());



         // get the finite-element space for the current grid function
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> nd_fes(
            new ParFiniteElementSpace(mesh, nd_fec.get()));
         
         ParGridFunction adj(nd_fes.get());
         VectorFunctionCoefficient v_rand(dim, randState);
         adj.ProjectCoefficient(v_rand);

         ParGridFunction mesh_sens(mesh_fes);
         mesh_sens = 0.0;
         solver.getCurrentSourceMeshSens(adj, mesh_sens);

         ParGridFunction v(mesh_fes);
         v.ProjectCoefficient(v_rand);

         double dJdX_v = mesh_sens * v;

         /// now compute centered difference difference
         double dJdX_v_cd = 0.0;
         // back step
         {
            auto back_mesh = getMesh(mesh_el,2);
            mach::MagnetostaticSolver back_solver(options, move(back_mesh));
            auto *b_mesh = back_solver.getMesh();
            auto *back_pert(static_cast<ParGridFunction*>(b_mesh->GetNodes()));
            back_pert->Add(-delta, v);
            b_mesh->SetNodes(*back_pert);

            back_solver.initDerived();
            back_solver.assembleCurrentSource();
            dJdX_v_cd -= adj * *back_solver.current_vec;
         }

         // forward step
         {
            auto forward_mesh = getMesh(mesh_el,2);

            mach::MagnetostaticSolver forward_solver(options, move(forward_mesh));

            auto *f_mesh = forward_solver.getMesh();
            auto *forward_pert(static_cast<ParGridFunction*>(f_mesh->GetNodes()));
            forward_pert->Add(delta, v);
            f_mesh->SetNodes(*forward_pert);

            forward_solver.initDerived();
            forward_solver.assembleCurrentSource();
            dJdX_v_cd += adj * *forward_solver.current_vec;
         }
         dJdX_v_cd /= (2*delta);

         std::cout << "dJdX_v: " << dJdX_v << "\n";
         std::cout << "dJdX_v_cd: " << dJdX_v_cd << "\n";
         REQUIRE(dJdX_v == Approx(dJdX_v_cd).margin(1e-10));
      }
   }
}

TEST_CASE("Magnetostatic Adjoint solved correctly")
{
   using namespace mfem;
   using namespace electromag_data;
   using namespace mach;

   const int dim = 3;
   const double delta = 1e-5;

   int mesh_el = 8;

   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,2);
         init_mesh->EnsureNodes();
         init_mesh->RemoveInternalBoundaries();

         nlohmann::json options = getBoxOptions(p);
         // generate initial tet mesh
         mach::MagnetostaticSolver solver(options, move(init_mesh));
         solver.initDerived();
         solver.solveForState();

         double dWdJ = solver.getFunctionalCurrentDensitySensitivity("co-energy");

         /// now compute centered difference difference
         double dWdJ_cd = 0.0;
         // back step
         {
            auto mesh = getMesh(mesh_el,2);
            nlohmann::json back_options = getBoxOptions(p);
            double J = back_options["problem-opts"]["current-density"].get<double>();
            J -= delta;
            back_options["problem-opts"]["current-density"] = J;

            mach::MagnetostaticSolver back_solver(back_options, move(mesh));
            back_solver.initDerived();
            back_solver.solveForState();
            dWdJ_cd -= back_solver.calcOutput("co-energy");

         }

         // forward step
         {
            auto forward_mesh = getMesh(mesh_el,2);
            nlohmann::json forward_options = getBoxOptions(p);
            double J = forward_options["problem-opts"]["current-density"].get<double>();
            J += delta;
            forward_options["problem-opts"]["current-density"] = J;

            mach::MagnetostaticSolver forward_solver(forward_options, move(forward_mesh));
            forward_solver.initDerived();
            forward_solver.solveForState();
            dWdJ_cd += forward_solver.calcOutput("co-energy");

         }
         dWdJ_cd /= (2*delta);

         std::cout << "dWdJ: " << dWdJ << "\n";
         std::cout << "dWdJ_cd: " << dWdJ_cd << "\n";
         REQUIRE(dWdJ == Approx(dWdJ_cd).margin(1e-10));
      }
   }
}

TEST_CASE("Rk = Dk - Wj Mesh Sensitivity")
{
   using namespace mfem;
   using namespace electromag_data;
   using namespace mach;

   const int dim = 3;
   const double delta = 1e-5;
   const double fd_delta = 1e-7;

   int mesh_el = 8;

   for (int p = 2; p <= 2; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,1);
         ParMesh mesh(MPI_COMM_WORLD, *init_mesh);
         mesh.EnsureNodes();
         auto *mesh_fes = static_cast<ParFiniteElementSpace*>(
                                             mesh.GetNodes()->FESpace());

         // get the finite-element space for the current grid function
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> nd_fes(
            new ParFiniteElementSpace(&mesh, nd_fec.get()));
            
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> h1_fec(
            new H1_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> h1_fes(
               new ParFiniteElementSpace(&mesh, h1_fec.get()));
         
         ParGridFunction adj(h1_fes.get());
         FunctionCoefficient rand(randState);
         adj.ProjectCoefficient(rand);

         VectorFunctionCoefficient v_rand(dim, randState);
         ParGridFunction v(mesh_fes);
         v.ProjectCoefficient(v_rand);

         /// Costruct coefficient
         mach::VectorMeshDependentCoefficient current(dim);
         std::unique_ptr<mfem::VectorCoefficient> coeff1(
            new VectorFunctionCoefficient(dim, func, funcRevDiff));
         std::unique_ptr<mfem::VectorCoefficient> coeff2(
            new VectorFunctionCoefficient(dim, func2, func2RevDiff));
         current.addCoefficient(1, move(coeff1));
         current.addCoefficient(2, move(coeff2));

         ParGridFunction j(nd_fes.get());
         j = 0.0;
         j.ProjectCoefficient(current);

         /// compute k
         ParMixedBilinearForm weakDiv(nd_fes.get(), h1_fes.get());
         weakDiv.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
         weakDiv.Assemble();
         weakDiv.Finalize();

         ParGridFunction Wj(h1_fes.get());
         Wj = 0.0;
         weakDiv.Mult(j, Wj);

         ParGridFunction k(h1_fes.get());
         k = 0.0;
         k.ProjectCoefficient(rand);

         ParLinearForm Rk_mesh_sens(mesh_fes);
         /// add integrators R_k = Dk - Wj = 0
         /// \psi_k^T Dk
         ConstantCoefficient one(1.0);
         Rk_mesh_sens.AddDomainIntegrator(
            new DiffusionResIntegrator(one, &k, &adj));
         /// -\psi_k^T W j 
         Rk_mesh_sens.AddDomainIntegrator(
            new VectorFEWeakDivergencedJdXIntegrator(&j, &adj, &current, -1.0));
         Rk_mesh_sens.Assemble();

         double dRkdX_v = Rk_mesh_sens * v;

         double dRkdX_v_cd = 0.0;
         /// backward step
         {
            auto back_mesh = getMesh(mesh_el,1);
            ParMesh b_mesh(MPI_COMM_WORLD, *back_mesh);
            b_mesh.EnsureNodes();
            auto *b_mesh_nodes = static_cast<ParGridFunction*>(b_mesh.GetNodes());

            b_mesh_nodes->Add(-delta, v);
            back_mesh->SetNodes(*b_mesh_nodes);
            std::unique_ptr<ParFiniteElementSpace> b_nd_fes(
                  new ParFiniteElementSpace(&b_mesh, nd_fec.get()));
            std::unique_ptr<ParFiniteElementSpace> b_h1_fes(
                  new ParFiniteElementSpace(&b_mesh, h1_fec.get()));

            /// compute k
            ParMixedBilinearForm weakDiv(b_nd_fes.get(), b_h1_fes.get());
            weakDiv.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
            weakDiv.Assemble();
            weakDiv.Finalize();

            ParGridFunction j(b_nd_fes.get());
            j = 0.0;
            j.ProjectCoefficient(current);

            ParGridFunction Wj(b_h1_fes.get());
            Wj = 0.0;
            weakDiv.Mult(j, Wj);

            // ParGridFunction k(b_h1_fes.get());
            // k = 0.0;
            ParGridFunction Dk(b_h1_fes.get());
            Dk = 0.0;
            {
               Array<int> ess_bdr, ess_bdr_tdofs;
               ess_bdr.SetSize(b_h1_fes->GetParMesh()->bdr_attributes.Max());
               ess_bdr = 1;
               b_h1_fes->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

               ParBilinearForm D(b_h1_fes.get());
               D.AddDomainIntegrator(new DiffusionIntegrator);
               D.Assemble();
               D.Finalize();

               D.Mult(k, Dk);
            }
            ParGridFunction Rk(b_h1_fes.get());
            Rk = 0.0;
            Rk.Add(1.0, Dk);
            Rk.Add(-1.0, Wj);
            dRkdX_v_cd -= adj*Rk;
         }
         /// forward step
         {
            auto back_mesh = getMesh(mesh_el,1);
            ParMesh b_mesh(MPI_COMM_WORLD, *back_mesh);
            b_mesh.EnsureNodes();
            auto *b_mesh_nodes = static_cast<ParGridFunction*>(b_mesh.GetNodes());

            b_mesh_nodes->Add(delta, v);
            back_mesh->SetNodes(*b_mesh_nodes);
            std::unique_ptr<ParFiniteElementSpace> f_nd_fes(
                  new ParFiniteElementSpace(&b_mesh, nd_fec.get()));
            std::unique_ptr<ParFiniteElementSpace> f_h1_fes(
                  new ParFiniteElementSpace(&b_mesh, h1_fec.get()));

            /// compute k
            ParMixedBilinearForm weakDiv(f_nd_fes.get(), f_h1_fes.get());
            weakDiv.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);
            weakDiv.Assemble();
            weakDiv.Finalize();

            ParGridFunction j(f_nd_fes.get());
            j = 0.0;
            j.ProjectCoefficient(current);

            ParGridFunction Wj(f_h1_fes.get());
            Wj = 0.0;
            weakDiv.Mult(j, Wj);

            // ParGridFunction k(f_h1_fes.get());
            // k = 0.0;
            ParGridFunction Dk(f_h1_fes.get());
            Dk = 0.0;
            {
               Array<int> ess_bdr, ess_bdr_tdofs;
               ess_bdr.SetSize(f_h1_fes->GetParMesh()->bdr_attributes.Max());
               ess_bdr = 1;
               f_h1_fes->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);

               ParBilinearForm D(f_h1_fes.get());
               D.AddDomainIntegrator(new DiffusionIntegrator);
               D.Assemble();
               D.Finalize();

               D.Mult(k, Dk);
            }
            ParGridFunction Rk(f_h1_fes.get());
            Rk = 0.0;
            Rk.Add(1.0, Dk);
            Rk.Add(-1.0, Wj);
            dRkdX_v_cd += adj*Rk;
         }
         dRkdX_v_cd /= (2*delta);

         REQUIRE(dRkdX_v == Approx(dRkdX_v_cd).margin(1e-10));
      }
   }
}

TEST_CASE("Discrete Gradient Operator - Should have no spatial dependence")
{
   using namespace mfem;
   using namespace electromag_data;
   using namespace mach;

   const int dim = 3;
   const double delta = 1e-5;
   const double fd_delta = 1e-7;

   int mesh_el = 4;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,2);
         ParMesh mesh(MPI_COMM_WORLD, *init_mesh);
         mesh.EnsureNodes();
         auto *mesh_fes = static_cast<ParFiniteElementSpace*>(
                                             mesh.GetNodes()->FESpace());

         // get the finite-element space for the current grid function
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> nd_fes(
            new ParFiniteElementSpace(&mesh, nd_fec.get()));
            
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> h1_fec(
            new H1_FECollection(p, dim));
         std::unique_ptr<ParFiniteElementSpace> h1_fes(
               new ParFiniteElementSpace(&mesh, h1_fec.get()));
         
         ParGridFunction adj(h1_fes.get());
         FunctionCoefficient rand(randState);
         adj.ProjectCoefficient(rand);

         VectorFunctionCoefficient v_rand(dim, randState);
         ParGridFunction v(mesh_fes);
         v.ProjectCoefficient(v_rand);

         VectorFunctionCoefficient current(3, func, funcRevDiff);
         ParGridFunction j(nd_fes.get());
         j = 0.0;
         j.ProjectCoefficient(current);

         /// now compute centered difference difference
         double dJdX_v_cd = 0.0;
         // back step
         {
            auto back_mesh = getMesh(mesh_el,2);
            ParMesh b_mesh(MPI_COMM_WORLD, *back_mesh);
            b_mesh.EnsureNodes();
            auto *b_mesh_nodes = static_cast<ParGridFunction*>(b_mesh.GetNodes());

            b_mesh_nodes->Add(-delta, v);
            back_mesh->SetNodes(*b_mesh_nodes);
            std::unique_ptr<ParFiniteElementSpace> b_nd_fes(
                  new ParFiniteElementSpace(&b_mesh, nd_fec.get()));
            std::unique_ptr<ParFiniteElementSpace> b_h1_fes(
                  new ParFiniteElementSpace(&b_mesh, h1_fec.get()));


            mfem::common::ParDiscreteGradOperator grad(b_h1_fes.get(), b_nd_fes.get());
            grad.Assemble();
            grad.Finalize();

            ParGridFunction GTj(b_h1_fes.get());
            GTj = 0.0;
            grad.MultTranspose(j, GTj);

            dJdX_v_cd -= adj*GTj;
         }

         // forward step
         {
            auto for_mesh = getMesh(mesh_el,2);
            ParMesh f_mesh(MPI_COMM_WORLD, *for_mesh);
            f_mesh.EnsureNodes();
            auto *f_mesh_nodes = static_cast<ParGridFunction*>(f_mesh.GetNodes());

            f_mesh_nodes->Add(delta, v);
            for_mesh->SetNodes(*f_mesh_nodes);
            std::unique_ptr<ParFiniteElementSpace> f_nd_fes(
                  new ParFiniteElementSpace(&f_mesh, nd_fec.get()));
            std::unique_ptr<ParFiniteElementSpace> f_h1_fes(
                  new ParFiniteElementSpace(&f_mesh, h1_fec.get()));

            mfem::common::ParDiscreteGradOperator grad(f_h1_fes.get(), f_nd_fes.get());
            grad.Assemble();
            grad.Finalize();

            ParGridFunction GTj(f_h1_fes.get());
            GTj = 0.0;
            grad.MultTranspose(j, GTj);

            dJdX_v_cd += adj*GTj;
         }
         dJdX_v_cd /= (2*delta);
          
         REQUIRE(0.0 == Approx(dJdX_v_cd).margin(1e-10));
      }
   }
}

/**
TEST_CASE("MagnetostaticSolver::getMeshSensitivities - interior only",
          "[MagnetostaticSolver]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   const double delta = 1e-5;
   const double fd_delta = 1e-7;

   int mesh_el = 8;

   for (int p = 2; p <= 2; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         // generate initial tet mesh
         auto init_mesh = getMesh(mesh_el,2);
         init_mesh->EnsureNodes();
         init_mesh->RemoveInternalBoundaries();

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = init_mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         Array<int> ess_bdr, ess_bdr_tdofs;
         ess_bdr.SetSize(mesh_fes->GetMesh()->bdr_attributes.Max());
         ess_bdr = 1;
         mesh_fes->GetEssentialTrueDofs(ess_bdr, ess_bdr_tdofs);


         nlohmann::json options = getBoxOptions(p);

         mach::MagnetostaticSolver solver(options, move(init_mesh));
         solver.initDerived();
         solver.solveForState();
         solver.printSolution("state", 0);
         double energy = solver.calcOutput("co-energy");
         GridFunction *dJdX = solver.getMeshSensitivities();

         GridFunction v(dJdX->FESpace());
         Vector zero;
         Vector rand(3);
         randState(zero, rand);
         // VectorConstantCoefficient v_rand(rand);
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         for (int i = 0; i < ess_bdr_tdofs.Size(); ++i)
         {
            v(ess_bdr_tdofs[i]) = 0.0;
         }

         double dJdX_v = (*dJdX) * v;

         /// now compute centered difference difference
         double dJdX_v_cd = 0.0;
         // back step
         {
            auto back_mesh = getMesh(mesh_el,2);

            mach::GridFunType back_pert(*static_cast<mach::GridFunType*>(back_mesh->GetNodes()));
            back_pert.Add(-delta, v);
            back_mesh->SetNodes(back_pert);
            mach::MagnetostaticSolver back_solver(options, move(back_mesh));
            back_solver.initDerived();
            back_solver.solveForState();
            dJdX_v_cd -= back_solver.calcOutput("co-energy");
         }

         // forward step
         {
            auto forward_mesh = getMesh(mesh_el,2);

            GridFunction forward_pert(*forward_mesh->GetNodes());
            forward_pert.Add(delta, v);
            forward_mesh->SetNodes(forward_pert);
            mach::MagnetostaticSolver forward_solver(options, move(forward_mesh));
            forward_solver.initDerived();
            forward_solver.solveForState();
            double forward_energy = forward_solver.calcOutput("co-energy");
            dJdX_v_cd += forward_energy;
         }
         dJdX_v_cd /= (2*delta);

         /// now compute forward difference
         double dJdX_v_fd = -energy;
         // forward step
         {
            auto forward_mesh = getMesh(mesh_el,2);

            GridFunction forward_pert(*forward_mesh->GetNodes());
            forward_pert.Add(fd_delta, v);
            forward_mesh->SetNodes(forward_pert);
            mach::MagnetostaticSolver forward_solver(options, move(forward_mesh));
            forward_solver.initDerived();
            forward_solver.solveForState();
            dJdX_v_fd += forward_solver.calcOutput("co-energy");
         }
         dJdX_v_fd /= fd_delta;

         std::cout << "Forward difference: " << dJdX_v_fd << "\n";
         std::cout << "Central difference: " << dJdX_v_cd << "\n";
         std::cout << "Analytic derivative: " << dJdX_v << "\n";

         // REQUIRE(dJdX_v == Approx(dJdX_v_cd).margin(1e-10));
         REQUIRE(dJdX_v == Approx(dJdX_v_fd).margin(1e-10));
      }
   }
}
*/
