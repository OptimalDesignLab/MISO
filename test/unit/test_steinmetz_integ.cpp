#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_test_data.hpp"

#include "coefficient.hpp"
#include "electromag_integ.hpp"
#include "material_library.hpp"

TEST_CASE("SteinmetzLossIntegrator::GetElementEnergy")
{
   const int dim = 3;
   int num_edge = 3;
   auto smesh = mfem::Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                            mfem::Element::TETRAHEDRON,
                                            1.0, 1.0, 1.0, true);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();


   mfem::L2_FECollection fec(1, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   mfem::NonlinearForm functional(&fes);

   mfem::ConstantCoefficient rho(1.0);
   mfem::ConstantCoefficient k_s(0.01);
   mfem::ConstantCoefficient alpha(1.21);
   mfem::ConstantCoefficient beta(1.62);
   auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
   setInputs(*integ, {
      {"frequency", 151.0},
      {"max_flux_magnitude", 2.2}
   });

   functional.AddDomainIntegrator(integ);

   mfem::Vector dummy_vec(fes.GetTrueVSize());
   auto core_loss = functional.GetEnergy(dummy_vec);

   /// Answer should be k_s * pow(freq, alpha) * pow(|B|)^beta 
   /// 0.01 * pow(151, 1.21) * pow(2.2, 1.62) -> 15.5341269187
   REQUIRE(core_loss == Approx(15.5341269187));
}

TEST_CASE("SteinmetzLossIntegratorFreqSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   mfem::FunctionCoefficient rho(f, f_rev_diff);
   mfem::FunctionCoefficient k_s(f, f_rev_diff);
   mfem::FunctionCoefficient alpha(f, f_rev_diff);
   mfem::FunctionCoefficient beta(f, f_rev_diff);
   // mfem::ConstantCoefficient rho(1.0);
   // mfem::ConstantCoefficient k_s(0.01);
   // mfem::ConstantCoefficient alpha(1.21);
   // mfem::ConstantCoefficient beta(1.62);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::SteinmetzLossIntegratorFreqSens(*integ));

         double frequency = 2.0 + randNumber();
         mach::MachInputs inputs{
            {"frequency", frequency},
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);

         // now compute the finite-difference approximation...
         inputs["frequency"] = frequency + delta;
         setInputs(*integ, inputs);
         double dfdp_fd_p = functional.GetEnergy(a);

         inputs["frequency"] = frequency - delta;
         setInputs(*integ, inputs);
         double dfdp_fd_m = functional.GetEnergy(a);

         double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));

      }
   }
}

TEST_CASE("SteinmetzLossIntegratorMaxFluxSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   mfem::FunctionCoefficient rho(f, f_rev_diff);
   mfem::FunctionCoefficient k_s(f, f_rev_diff);
   mfem::FunctionCoefficient alpha(f, f_rev_diff);
   mfem::FunctionCoefficient beta(f, f_rev_diff);
   // mfem::ConstantCoefficient rho(1.0);
   // mfem::ConstantCoefficient k_s(0.01);
   // mfem::ConstantCoefficient alpha(1.21);
   // mfem::ConstantCoefficient beta(1.62);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // evaluate dJdp and compute its product with pert
         NonlinearForm dJdp(&fes);
         dJdp.AddDomainIntegrator(
            new mach::SteinmetzLossIntegratorMaxFluxSens(*integ));

         double max_flux_magnitude = 2.0 + randNumber();
         mach::MachInputs inputs{
            {"max_flux_magnitude", max_flux_magnitude},
         };
         setInputs(*integ, inputs);
         double dfdp_fwd = dJdp.GetEnergy(a);

         // now compute the finite-difference approximation...
         inputs["max_flux_magnitude"] = max_flux_magnitude + delta;
         setInputs(*integ, inputs);
         double dfdp_fd_p = functional.GetEnergy(a);

         inputs["max_flux_magnitude"] = max_flux_magnitude - delta;
         setInputs(*integ, inputs);
         double dfdp_fd_m = functional.GetEnergy(a);

         double dfdp_fd = (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));

      }
   }
}

TEST_CASE("SteinmetzLossIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   auto f = [](const mfem::Vector &x)
   {
      double q = 0;
      for (int i = 0; i < x.Size(); ++i)
      {
         q += pow(x(i), 2);
      }
      return q;
   };

   auto f_rev_diff = [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   {
      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) += q_bar * 2 * x(i);
      }
   };

   mfem::FunctionCoefficient rho(f, f_rev_diff);
   mfem::FunctionCoefficient k_s(f, f_rev_diff);
   mfem::FunctionCoefficient alpha(f, f_rev_diff);
   mfem::FunctionCoefficient beta(f, f_rev_diff);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new mach::SteinmetzLossIntegrator(rho, k_s, alpha, beta);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new mach::SteinmetzLossIntegratorMeshSens(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(a);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(a);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}





/** not maintaining anymore
TEST_CASE("DomainResIntegrator::AssembleElementVector",
          "[DomainResIntegrator for Steinmetz]")
{
   using namespace mfem;
   using namespace euler_data;
   using namespace mach;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh = electromag_data::getMesh(2, 1);
                              //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                              //        true , 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         H1_FECollection fec(p, dim);
         ND_FECollection feca(p, dim);
         FiniteElementSpace fes(mesh.get(), &fec);
         FiniteElementSpace fesa(mesh.get(), &feca);

         // we use res for finite-difference approximation
         GridFunction A(&fesa);
         VectorFunctionCoefficient perta(dim, electromag_data::randVectorState);
         A.ProjectCoefficient(perta);
         // std::unique_ptr<Coefficient> q2(new FunctionCoefficient(func, funcRevDiff));
         std::unique_ptr<Coefficient> q2(new SteinmetzCoefficient(
                        1, 2, 4, 0.5, 0.6, A));
         // std::unique_ptr<mach::MeshDependentCoefficient> Q;
         // Q.reset(new mach::MeshDependentCoefficient());
         // Q->addCoefficient(1, move(q1)); 
         // Q->addCoefficient(2, move(q2));
         LinearForm res(&fes);
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*q2));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(electromag_data::randState);
         adjoint.ProjectCoefficient(pert);

         // extract mesh nodes and get their finite-element space
         auto *x_nodes = dynamic_cast<GridFunction*>(mesh->GetNodes());
         auto *mesh_fes = dynamic_cast<FiniteElementSpace*>(x_nodes->FESpace());

         // build the nonlinear form for d(psi^T R)/dx 
         NonlinearForm dfdx_form(mesh_fes);
         dfdx_form.AddDomainIntegrator(
            new mach::DomainResIntegrator(*q2, &adjoint));

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, electromag_data::randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
*/

/** not maintaining anymore
TEST_CASE("ThermalSensIntegrator::AssembleElementVector",
          "[ThermalSensIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;
   using namespace mach;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh = electromag_data::getMesh();
                              //(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                              //        true, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementCollection> feca(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));
         std::unique_ptr<FiniteElementSpace> fesa(new FiniteElementSpace(
             mesh.get(), feca.get()));

         // we use res for finite-difference approximation
         GridFunction A(fesa.get());
         VectorFunctionCoefficient perta(dim, electromag_data::randVectorState);
         A.ProjectCoefficient(perta);
         std::unique_ptr<Coefficient> q1(new ConstantCoefficient(1));
         std::unique_ptr<Coefficient> q2(new SteinmetzCoefficient(
                        1, 2, 4, 0.5, 0.6, A));
         std::unique_ptr<mach::MeshDependentCoefficient> Q;
         // Q.reset(new mach::MeshDependentCoefficient());
         // Q->addCoefficient(1, move(q1)); 
         // Q->addCoefficient(2, move(q2));
         std::unique_ptr<VectorCoefficient> QV(
            new SteinmetzVectorDiffCoefficient(1, 2, 4, 0.5, 0.6, A));
         LinearForm res(fes.get());
         res.AddDomainIntegrator(
            new DomainLFIntegrator(*q2));

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(fes.get()), adjoint(fes.get());
         FunctionCoefficient pert(electromag_data::randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);


         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx_form(fesa.get());
         dfdx_form.AddDomainIntegrator(
            new mach::ThermalSensIntegrator(*QV, &adjoint));

         // initialize the vector that we use to perturb the vector potential
         GridFunction v(fesa.get());
         VectorFunctionCoefficient v_rand(dim, electromag_data::randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         //GridFunction dfdx(*x_nodes);
         dfdx_form.Assemble();
         double dfdx_v = dfdx_form * v;

         // now compute the finite-difference approximation...
         //GridFunction a_pert(A);
         A.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         A.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         A.Add(delta, v);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
*/
