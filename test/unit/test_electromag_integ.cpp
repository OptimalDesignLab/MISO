#include <random>
#include <typeinfo>

#include "catch.hpp"
#include "coefficient.hpp"
#include "mfem.hpp"

#include "electromag_integ.hpp"
#include "electromag_test_data.hpp"

class TestMagnetizationSource : public miso::VectorStateCoefficient
{
public:
   void Eval(mfem::Vector &v,
             mfem::ElementTransformation &trans,
             const mfem::IntegrationPoint &ip,
             double state) override
   {
      int space_dim = trans.GetSpaceDim();
      double x_buffer[3] = {};
      mfem::Vector x(x_buffer, space_dim);
      trans.Transform(ip, x);

      for (int i = 0; i < x.Size(); ++i)
      {
         v(i) = pow(x(i), 2) * std::pow(state, 2);
      }
   }

   void EvalStateDeriv(mfem::Vector &vec_dot,
                       mfem::ElementTransformation &trans,
                       const mfem::IntegrationPoint &ip,
                       double state) override
   {
      int space_dim = trans.GetSpaceDim();
      double x_buffer[3] = {};
      mfem::Vector x(x_buffer, space_dim);
      trans.Transform(ip, x);

      for (int i = 0; i < x.Size(); ++i)
      {
         vec_dot(i) = pow(x(i), 2) * 2 * state;
      }
   }

   void  EvalRevDiff(const mfem::Vector &V_bar,
                    mfem::ElementTransformation &trans,
                    const mfem::IntegrationPoint &ip,
                    double state,
                    mfem::DenseMatrix &PointMat_bar) override
   {
      int space_dim = trans.GetSpaceDim();
      double x_buffer[3] = {};
      mfem::Vector x(x_buffer, space_dim);
      trans.Transform(ip, x);

      double x_bar_buffer[3] = {};
      mfem::Vector x_bar(x_bar_buffer, space_dim);

      for (int i = 0; i < x.Size(); ++i)
      {
         x_bar(i) = V_bar(i) * 2 * x(i) * std::pow(state, 2);
      }

      static_cast<mfem::IsoparametricTransformation &>(trans).TransformRevDiff(
         ip, x_bar, PointMat_bar);
   }

   TestMagnetizationSource(int vdim)
   : miso::VectorStateCoefficient(vdim)
   {}
};

TEST_CASE("NonlinearDiffusionIntegrator::AssembleElementGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   NonLinearCoefficient nu;
   // LinearCoefficient nu;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddDomainIntegrator(new miso::NonlinearDiffusionIntegrator(nu));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("NonlinearDiffusionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   NonLinearCoefficient nu;
   // LinearCoefficient nu;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new miso::NonlinearDiffusionIntegrator(nu);
         res.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::NonlinearDiffusionIntegratorMeshRevSens(state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("NonlinearDGDiffusionIntegrator::AssembleFaceVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   LinearCoefficient one_sc(10.0);
   ConstantCoefficient one(10.0);
   ConstantCoefficient bc_val(-10.0);
   double sigma = -1.0;
   double mu = 10;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddBdrFaceIntegrator(new miso::NonlinearDGDiffusionIntegrator(one_sc, bc_val, mu));

         GridFunction res_vec(&fes);
         res.Mult(state, res_vec);

         BilinearForm blf(&fes);
         auto *blfi = new DGDiffusionIntegrator(one, sigma, mu);
         blfi->SetIntRule(&IntRules.Get(Geometry::SEGMENT, 2*p));
         blf.AddBdrFaceIntegrator(blfi);
         blf.Assemble();

         LinearForm lf(&fes);
         auto *lfi = new DGDirichletLFIntegrator(bc_val, one, sigma, mu);
         lfi->SetIntRule(&IntRules.Get(Geometry::SEGMENT, 2*p));
         lf.AddBdrFaceIntegrator(lfi);
         lf.Assemble();

         GridFunction blf_res(&fes);
         blf.Mult(state, blf_res);
         blf_res -= lf;

         for (int i = 0; i < state.Size(); ++i)
         {
            REQUIRE(res_vec(i) == Approx(blf_res(i)).margin(1e-8));
         }
      }
   }
}

TEST_CASE("NonlinearDGDiffusionIntegrator::AssembleFaceGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   // LinearCoefficient one_sc(1.0);
   NonLinearCoefficient one_sc;
   ConstantCoefficient bc_val(10.0);
   double sigma = -1.0;
   double mu = 0.0;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddBdrFaceIntegrator(new miso::NonlinearDGDiffusionIntegrator(one_sc, bc_val, mu));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("NonlinearDGDiffusionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 3;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   // LinearCoefficient one_sc(1.0);
   NonLinearCoefficient one_sc;
   ConstantCoefficient bc_val(10.0);
   double mu = 10;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new miso::NonlinearDGDiffusionIntegrator(one_sc, bc_val, mu);
         res.AddBdrFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddBdrFaceIntegrator(
            new miso::NonlinearDGDiffusionIntegratorMeshRevSens(mesh_fes, state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

// TEST_CASE("DGInteriorFaceDiffusionIntegrator2::AssembleFaceVector")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 2;
//    auto mesh = Mesh::MakeCartesian2D(num_edge,
//                                      num_edge,
//                                      Element::TRIANGLE);
//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    mesh.SetCurvature(1);

//    auto &mesh_gf = *mesh.GetNodes();
//    auto *mesh_fespace = mesh_gf.FESpace();

//    mfem::GridFunction mesh_pert(mesh_fespace);
//    for (int i = 0; i < mesh_pert.Size(); ++i)
//    {
//       mesh_pert(i) = electromag_data::randNumber();
//    }
//    mesh_pert /= (100 * (num_edge / 32.0));
//    // mesh_pert = 0.0;

//    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    ess_bdr = 1;
//    mfem::Array<int> ess_tdof_list;
//    mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    mesh_pert.SetSubVector(ess_tdof_list, 0.0);
//    mesh_gf += mesh_pert;

//    LinearCoefficient one_sc(1.0);
//    ConstantCoefficient one(1.0);
//    double sigma = -1.0;
//    double mu = 10.0;

//    for (int p = 1; p <= 1; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {
//          L2_FECollection fec(p, dim);
//          FiniteElementSpace fes(&mesh, &fec);

//          GridFunction state(&fes);
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);

//          NonlinearForm res(&fes);
//          res.AddInteriorFaceIntegrator(
//             new miso::DGInteriorFaceDiffusionIntegrator2(one_sc, mu));

//          GridFunction res_vec(&fes);
//          res.Mult(state, res_vec);

//          NonlinearForm mfem_res(&fes);
//          mfem_res.AddInteriorFaceIntegrator(
//             new DGDiffusionIntegrator(one_sc, sigma, mu));

//          GridFunction mfem_res_vec(&fes);
//          mfem_res.Mult(state, mfem_res_vec);

//          // evaluate the Jacobian and compute its product with v
//          Operator& jac = res.GetGradient(state);
//          GridFunction jac_v(&fes);
//          GridFunction v(&fes);
//          DenseMatrix dJac(v.Size());
//          DenseMatrix dJac_fd(v.Size());
//          for (int i = 0; i < v.Size(); ++i)
//          {
//             v = 0.0;
//             v(i) = 1.0;
//             jac.Mult(v, jac_v);
//             // now compute the finite-difference approximation...
//             GridFunction r(&fes), jac_v_fd(&fes);
//             state.Add(-delta, v);
//             res.Mult(state, r);
//             state.Add(2*delta, v);
//             res.Mult(state, jac_v_fd);
//             jac_v_fd -= r;
//             jac_v_fd /= (2*delta);
//             // std::cout << "Jac_v:\n";
//             // jac_v.Print(mfem::out, 1); 
//             // std::cout << "Jac_v_fd:\n";
//             // jac_v_fd.Print(mfem::out, 1); 
//             state.Add(-delta, v);
//             for (int j = 0; j < v.Size(); ++j)
//             {
//                dJac(j, i) = jac_v(j);
//                dJac_fd(j, i) = jac_v_fd(j);
//             }
//          }
//          std::cout << "Jac:\n";
//          dJac.Print(mfem::out, v.Size()); 
//          std::cout << "Jac_fd:\n";
//          dJac_fd.Print(mfem::out, v.Size()); 

//          for (int i = 0; i < v.Size(); ++i)
//          {
//             for (int j = 0; j < v.Size(); ++j)
//             {
//                REQUIRE(dJac(i, j) == Approx(dJac_fd(i, j)).margin(1e-6));
//             }
//          }

//          std::cout << "res:\n";
//          res_vec.Print(mfem::out, v.Size()); 
//          std::cout << "mfem_res:\n";
//          mfem_res_vec.Print(mfem::out, v.Size()); 

//          for (int i = 0; i < state.Size(); ++i)
//          {
//             REQUIRE(res_vec(i) == Approx(mfem_res_vec(i)).margin(1e-8));
//          }
//       }
//    }
// }

// TEST_CASE("DGInteriorFaceDiffusionIntegrator2::AssembleFaceGrad")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 2;
//    auto mesh = Mesh::MakeCartesian2D(num_edge,
//                                      num_edge,
//                                      Element::TRIANGLE);
//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    mesh.SetCurvature(1);

//    auto &mesh_gf = *mesh.GetNodes();
//    auto *mesh_fespace = mesh_gf.FESpace();

//    mfem::GridFunction mesh_pert(mesh_fespace);
//    for (int i = 0; i < mesh_pert.Size(); ++i)
//    {
//       mesh_pert(i) = electromag_data::randNumber();
//    }
//    mesh_pert /= (100 * (num_edge / 32.0));
//    // mesh_pert = 0.0;

//    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
//    ess_bdr = 1;
//    mfem::Array<int> ess_tdof_list;
//    mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
//    mesh_pert.SetSubVector(ess_tdof_list, 0.0);
//    mesh_gf += mesh_pert;

//    // LinearCoefficient one_sc(1.0);
//    NonLinearCoefficient one_sc;
//    double mu = 10.0;

//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {
//          L2_FECollection fec(p, dim);
//          FiniteElementSpace fes(&mesh, &fec);

//          GridFunction state(&fes);
//          FunctionCoefficient pert(randState);
//          state.ProjectCoefficient(pert);

//          NonlinearForm res(&fes);
//          res.AddInteriorFaceIntegrator(
//             new miso::DGInteriorFaceDiffusionIntegrator2(one_sc, mu));

//          // initialize the vector that the Jacobian multiplies
//          GridFunction v(&fes);
//          v.ProjectCoefficient(pert);

//          // evaluate the Jacobian and compute its product with v
//          Operator& jac = res.GetGradient(state);
//          GridFunction jac_v(&fes);
//          jac.Mult(v, jac_v);

//          // now compute the finite-difference approximation...
//          GridFunction r(&fes), jac_v_fd(&fes);
//          state.Add(-delta, v);
//          res.Mult(state, r);
//          state.Add(2*delta, v);
//          res.Mult(state, jac_v_fd);
//          jac_v_fd -= r;
//          jac_v_fd /= (2*delta);

//          for (int i = 0; i < jac_v.Size(); ++i)
//          {
//             REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
//          }

//          // // evaluate the Jacobian and compute its product with v
//          // DenseMatrix dJac(v.Size());
//          // DenseMatrix dJac_fd(v.Size());
//          // for (int i = 0; i < v.Size(); ++i)
//          // {
//          //    v = 0.0;
//          //    v(i) = 1.0;
//          //    jac.Mult(v, jac_v);
//          //    // now compute the finite-difference approximation...
//          //    GridFunction r(&fes), jac_v_fd(&fes);
//          //    state.Add(-delta, v);
//          //    res.Mult(state, r);
//          //    state.Add(2*delta, v);
//          //    res.Mult(state, jac_v_fd);
//          //    jac_v_fd -= r;
//          //    jac_v_fd /= (2*delta);
//          //    // std::cout << "Jac_v:\n";
//          //    // jac_v.Print(mfem::out, 1); 
//          //    // std::cout << "Jac_v_fd:\n";
//          //    // jac_v_fd.Print(mfem::out, 1); 
//          //    state.Add(-delta, v);
//          //    for (int j = 0; j < v.Size(); ++j)
//          //    {
//          //       dJac(j, i) = jac_v(j);
//          //       dJac_fd(j, i) = jac_v_fd(j);
//          //    }
//          // }
//          // std::cout << "Jac:\n";
//          // dJac.Print(mfem::out, v.Size()); 
//          // std::cout << "Jac_fd:\n";
//          // dJac_fd.Print(mfem::out, v.Size()); 

//          // v.ProjectCoefficient(pert);
//          // // v = 0.0;
//          // // v(0) = 1.0;
//          // jac.Mult(v, jac_v);
//          // GridFunction r(&fes), jac_v_fd(&fes);
//          // state.Add(-delta, v);
//          // res.Mult(state, r);
//          // state.Add(2*delta, v);
//          // res.Mult(state, jac_v_fd);
//          // jac_v_fd -= r;
//          // jac_v_fd /= (2*delta);

//          // for (int i = 0; i < jac_v.Size(); ++i)
//          // {
//          //    REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
//          // }
//       }
//    }
// }

TEST_CASE("DGInteriorFaceDiffusionIntegrator::AssembleFaceVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   LinearCoefficient one_sc(1.0);
   ConstantCoefficient one(1.0);
   double sigma = -1.0;
   double mu = 10.0;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddInteriorFaceIntegrator(
            new miso::DGInteriorFaceDiffusionIntegrator(one_sc, mu));

         GridFunction res_vec(&fes);
         res.Mult(state, res_vec);

         NonlinearForm mfem_res(&fes);
         mfem_res.AddInteriorFaceIntegrator(
            new DGDiffusionIntegrator(one, sigma, mu));

         GridFunction mfem_res_vec(&fes);
         mfem_res.Mult(state, mfem_res_vec);

         // std::cout << "custom:\n";
         // res_vec.Print(mfem::out, 1);
         // std::cout << "\n";

         // std::cout << "mfem:\n";
         // mfem_res_vec.Print(mfem::out, 1);

         for (int i = 0; i < state.Size(); ++i)
         {
            REQUIRE(res_vec(i) == Approx(mfem_res_vec(i)).margin(1e-8));
         }
      }
   }
}

TEST_CASE("DGInteriorFaceDiffusionIntegrator::AssembleFaceGrad")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   // LinearCoefficient one_sc(1.0);
   NonLinearCoefficient one_sc;
   double mu = 10.0;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         GridFunction state(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         res.AddInteriorFaceIntegrator(
            new miso::DGInteriorFaceDiffusionIntegrator(one_sc, mu));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& jac = res.GetGradient(state);
         GridFunction jac_v(&fes);
         // jac.Mult(v, jac_v);

         // // now compute the finite-difference approximation...
         // GridFunction r(&fes), jac_v_fd(&fes);
         // state.Add(-delta, v);
         // res.Mult(state, r);
         // state.Add(2*delta, v);
         // res.Mult(state, jac_v_fd);
         // jac_v_fd -= r;
         // jac_v_fd /= (2*delta);

         // for (int i = 0; i < jac_v.Size(); ++i)
         // {
         //    REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         // }

         // // evaluate the Jacobian and compute its product with v
         // DenseMatrix dJac(v.Size());
         // DenseMatrix dJac_fd(v.Size());
         // for (int i = 0; i < v.Size(); ++i)
         // {
         //    v = 0.0;
         //    v(i) = 1.0;
         //    jac.Mult(v, jac_v);
         //    // now compute the finite-difference approximation...
         //    GridFunction r(&fes), jac_v_fd(&fes);
         //    state.Add(-delta, v);
         //    res.Mult(state, r);
         //    state.Add(2*delta, v);
         //    res.Mult(state, jac_v_fd);
         //    jac_v_fd -= r;
         //    jac_v_fd /= (2*delta);
         //    // std::cout << "Jac_v:\n";
         //    // jac_v.Print(mfem::out, 1); 
         //    // std::cout << "Jac_v_fd:\n";
         //    // jac_v_fd.Print(mfem::out, 1); 
         //    state.Add(-delta, v);
         //    for (int j = 0; j < v.Size(); ++j)
         //    {
         //       dJac(j, i) = jac_v(j);
         //       dJac_fd(j, i) = jac_v_fd(j);
         //    }
         // }
         // std::cout << "Jac:\n";
         // dJac.Print(mfem::out, v.Size()); 
         // std::cout << "Jac_fd:\n";
         // dJac_fd.Print(mfem::out, v.Size()); 

         v.ProjectCoefficient(pert);
         // v = 0.0;
         // v(0) = 1.0;
         jac.Mult(v, jac_v);
         GridFunction r(&fes), jac_v_fd(&fes);
         state.Add(-delta, v);
         res.Mult(state, r);
         state.Add(2*delta, v);
         res.Mult(state, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6));
         }
      }
   }
}

TEST_CASE("DGInteriorFaceDiffusionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mesh.SetCurvature(1);

   auto &mesh_gf = *mesh.GetNodes();
   auto *mesh_fespace = mesh_gf.FESpace();

   mfem::GridFunction mesh_pert(mesh_fespace);
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = electromag_data::randNumber();
   }
   mesh_pert /= (100 * (num_edge / 32.0));
   // mesh_pert = 0.0;

   mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;
   mfem::Array<int> ess_tdof_list;
   mesh_fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   mesh_pert.SetSubVector(ess_tdof_list, 0.0);
   mesh_gf += mesh_pert;

   NonLinearCoefficient one_sc;
   double mu = 10;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new miso::DGInteriorFaceDiffusionIntegrator(one_sc, mu);
         res.AddInteriorFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddInteriorFaceIntegrator(
            new miso::DGInteriorFaceDiffusionIntegratorMeshRevSens(mesh_fes, state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("TestBoundaryIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new miso::TestBoundaryIntegrator;
         res.AddBdrFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         // dfdx.AddBdrFaceIntegrator(
         dfdx.AddBoundaryIntegrator(
            new miso::TestBoundaryIntegratorMeshRevSens(state, adjoint, *integ));

         dfdx.AddBdrFaceIntegrator(
            new miso::TestBoundaryIntegratorMeshRevSens(state, adjoint, *integ));

         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("TestInteriorFaceIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         L2_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction state(&fes);
         GridFunction adjoint(&fes);
         FunctionCoefficient pert(randState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         NonlinearForm res(&fes);
         auto *integ = new miso::TestInteriorFaceIntegrator;
         res.AddInteriorFaceIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddInteriorFaceIntegrator(
            new miso::TestInteriorFaceIntegratorMeshRevSens(mesh_fes, state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("MagnetizationSource2DIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   TestMagnetizationSource model(dim);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize temperature field
         GridFunction temp(&fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);


         // initialize adjoint; here we randomly perturb a constant state
         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         LinearForm res(&fes);
         auto *integ = new miso::MagnetizationSource2DIntegrator(model, temp);
         res.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::MagnetizationSource2DIntegratorMeshRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("MagnetizationSource2DIntegratorTemperatureRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge,
                                     num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   TestMagnetizationSource model(dim);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize temperature field
         GridFunction temp(&fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);


         // initialize adjoint; here we randomly perturb a constant state
         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         LinearForm res(&fes);
         auto *integ = new miso::MagnetizationSource2DIntegrator(model, temp);
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::MagnetizationSource2DIntegratorTemperatureSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         // GridFunction r(&fes);
         temp.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         temp.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad - linear",
          "[CurlCurlNLFIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                      Element::TETRAHEDRON,
                                      1.0, 1.0, 1.0, true);

   ParMesh mesh(MPI_COMM_WORLD, smesh); 
   mesh.EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         ND_FECollection nd_fec(p, dim);
         ParFiniteElementSpace fes(&mesh, &nd_fec);

         ParNonlinearForm res(&fes);

         LinearCoefficient nu(1.0);

         res.AddDomainIntegrator(
            new miso::CurlCurlNLFIntegrator(nu));

         // initialize state; here we randomly perturb a constant state
         ParGridFunction A(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         A.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         ParGridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(A);
         ParGridFunction jac_v(&fes);
         Jac.Mult(v, jac_v);

         ParGridFunction res_v(&fes);
         res.Mult(v, res_v);

         ConstantCoefficient nu_linear(1.0);
         ParBilinearForm blf(&fes);
         blf.AddDomainIntegrator(new CurlCurlIntegrator(nu_linear));
         blf.Assemble();
         ParGridFunction blf_v(&fes);
         blf.Mult(v, blf_v);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( res_v(i) == Approx(blf_v(i)) );
            REQUIRE( jac_v(i) == Approx(blf_v(i)) );
         }
      }
   }
}

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad",
          "[CurlCurlNLFIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));

   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm res(fes.get());

         std::unique_ptr<miso::StateCoefficient> nu(
            new LinearCoefficient());

         res.AddDomainIntegrator(new miso::CurlCurlNLFIntegrator(*nu));

         // GridFunction A(fes.get());
         // VectorFunctionCoefficient a0(3, [](const Vector &x, Vector &a)
         // {
         //    a(0) = -0.05*x(1);
         //    a(1) = 0.05*x(0);
         //    a(2) = 0.0;
         // });
         // A.ProjectCoefficient(a0);


         GridFunction q(fes.get());
         // res.Mult(A, q);
         // Operator& tJac = res.GetGradient(A);


         // initialize state; here we randomly perturb a constant state
         VectorFunctionCoefficient pert(3, randBaselineVectorPert);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2*delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-8) );
         }
      }
   }
}

TEST_CASE("CurlCurlNLFIntegrator::AssembleElementGrad - Nonlinear",
          "[CurlCurlNLFIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction a(fes.get());
         VectorFunctionCoefficient pert(3, randBaselineVectorPert);
         a.ProjectCoefficient(pert);

         std::unique_ptr<miso::StateCoefficient> nu(
            new NonLinearCoefficient());

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new miso::CurlCurlNLFIntegrator(*nu));

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator& Jac = res.GetGradient(a);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction r(fes.get()), jac_v_fd(fes.get());
         a.Add(-delta, v);
         res.Mult(a, r);
         a.Add(2*delta, v);
         res.Mult(a, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2*delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE( jac_v(i) == Approx(jac_v_fd(i)).margin(1e-6) );
         }
      }
   }
}

TEST_CASE("CurlCurlNLFIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true);
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize state and adjoint; here we randomly perturb a constant state
         GridFunction state(&fes), adjoint(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         state.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         NonlinearForm res(&fes);
         auto *integ = new miso::CurlCurlNLFIntegrator(nu);
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::CurlCurlNLFIntegratorMeshRevSens(state, adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Mult(state, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

/** moved/replaced in mfem_common_integ.xpp
TEST_CASE("VectorFECurldJdXIntegerator::AssembleRHSElementVect",
          "[VectorFECurldJdXIntegerator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> nd_fes(new FiniteElementSpace(
            mesh.get(), nd_fec.get()));

         // get the finite-element space for the magnetization grid function
         std::unique_ptr<FiniteElementCollection> rt_fec(
            new RT_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> rt_fes(new FiniteElementSpace(
            mesh.get(), rt_fec.get()));

         /// We use a linear coefficient since the magnets are linear
         std::unique_ptr<miso::StateCoefficient> nu(
            new LinearCoefficient());

         // initialize magnetization source and adjoint; here we randomly perturb a constant state
         GridFunction M(rt_fes.get()), adjoint(nd_fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         VectorFunctionCoefficient mag(3, vectorFunc, vectorFuncRevDiff);

         // /// Costruct coefficient
         // miso::VectorMeshDependentCoefficient mag(dim);
         // std::unique_ptr<mfem::VectorCoefficient> coeff1(
         //    new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         // std::unique_ptr<mfem::VectorCoefficient> coeff2(
         //    new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         // mag.addCoefficient(1, move(coeff1));
         // mag.addCoefficient(2, move(coeff2));

         M.ProjectCoefficient(mag);
         // M.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         MixedBilinearForm res(rt_fes.get(), nd_fes.get());
         res.AddDomainIntegrator(new VectorFECurlIntegrator(*nu));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::VectorFECurldJdXIntegerator(nu.get(), &M, &adjoint, &mag));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(nd_fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         rt_fes->Update();
         nd_fes->Update();
         M.Update();
         M.ProjectCoefficient(mag);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(M, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         rt_fes->Update();
         nd_fes->Update();
         M.Update();
         M.ProjectCoefficient(mag);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(M, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         // std::cout << "dfdx_v " << dfdx_v << "\n";
         // std::cout << "dfdx_v_fd " << dfdx_v_fd << "\n";

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
TEST_CASE("VectorFEMassdJdXIntegerator::AssembleRHSElementVect",
          "[VectorFEMassdJdXIntegerator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize magnetization source and adjoint; here we randomly perturb a constant state
         GridFunction J(fes.get()), adjoint(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         VectorFunctionCoefficient current(3, vectorFunc, vectorFuncRevDiff);

         /// Costruct coefficient
         // miso::VectorMeshDependentCoefficient current(dim);
         // std::unique_ptr<mfem::VectorCoefficient> coeff1(
         //    new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         // std::unique_ptr<mfem::VectorCoefficient> coeff2(
         //    new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         // current.addCoefficient(1, move(coeff1));
         // current.addCoefficient(2, move(coeff2));

         J.ProjectCoefficient(current);
         // J.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         BilinearForm res(fes.get());
         res.AddDomainIntegrator(new VectorFEMassIntegrator);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::VectorFEMassdJdXIntegerator(&J, &adjoint, &current));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         J.Update();
         J.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(J, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         J.Update();
         J.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(J, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
TEST_CASE("VectorFEWeakDivergencedJdXIntegrator::AssembleRHSElementVect",
          "[VectorFEWeakDivergencedJdXIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the current grid function
         std::unique_ptr<FiniteElementCollection> nd_fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> nd_fes(new FiniteElementSpace(
            mesh.get(), nd_fec.get()));
            
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> h1_fec(
            new H1_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> h1_fes(new FiniteElementSpace(
            mesh.get(), h1_fec.get()));

         // initialize current source and adjoint
         // here we randomly perturb a constant state
         GridFunction c(nd_fes.get()), adjoint(h1_fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         FunctionCoefficient adj_pert(randState);
         VectorFunctionCoefficient current(3, vectorFunc, vectorFuncRevDiff);

         /// Costruct coefficient
         // miso::VectorMeshDependentCoefficient current(dim);
         // std::unique_ptr<mfem::VectorCoefficient> coeff1(
         //    new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         // std::unique_ptr<mfem::VectorCoefficient> coeff2(
         //    new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         // current.addCoefficient(1, move(coeff1));
         // current.addCoefficient(2, move(coeff2));

         c.ProjectCoefficient(current);
         // c.ProjectCoefficient(pert);
         adjoint.ProjectCoefficient(adj_pert);

         // we use res for finite-difference approximation
         MixedBilinearForm res(nd_fes.get(), h1_fes.get());
         res.AddDomainIntegrator(new VectorFEWeakDivergenceIntegrator);

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::VectorFEWeakDivergencedJdXIntegrator(&c, &adjoint, &current));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(h1_fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         nd_fes->Update();
         h1_fes->Update();
         c.Update();
         c.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(c, r);
         double dfdx_v_fd = adjoint * r;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         nd_fes->Update();
         h1_fes->Update();
         c.Update();
         c.ProjectCoefficient(current);
         res.Update();
         res.Assemble();
         res.Finalize();
         res.Mult(c, r);
         dfdx_v_fd -= adjoint * r;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}
*/

/** moved/replaced in mfem_common_integ.xpp
TEST_CASE("VectorFEDomainLFMeshSensInteg::AssembleRHSElementVect"
          "[VectorFEDomainLFMeshSensInteg]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;  // templating is hard here because mesh constructors
   // static adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = getMesh(num_edge, num_edge);
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the adjoint
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize adjoint; here we randomly perturb a constant state
         GridFunction adjoint(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);

         /// Costruct coefficient
         miso::VectorMeshDependentCoefficient current(dim);
         std::unique_ptr<mfem::VectorCoefficient> coeff1(
            new VectorFunctionCoefficient(dim, vectorFunc, vectorFuncRevDiff));
         std::unique_ptr<mfem::VectorCoefficient> coeff2(
            new VectorFunctionCoefficient(dim, vectorFunc2, vectorFunc2RevDiff));
         current.addCoefficient(1, move(coeff1));
         current.addCoefficient(2, move(coeff2));

         adjoint.ProjectCoefficient(pert);

         // we use res for finite-difference approximation
         LinearForm res(fes.get());
         res.AddDomainIntegrator(new VectorFEDomainLFIntegrator(current));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // build the nonlinear form for d(psi^T R)/dx 
         LinearForm dfdx(mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::VectorFEDomainLFMeshSensInteg(&adjoint, current));
         dfdx.Assemble();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // contract dfdx with v
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         res.Update();
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         res.Update();
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}
*/

// TEST_CASE("GridFuncMeshSensIntegrator::AssembleRHSElementVect",
//           "[GridFuncMeshSensIntegrator]")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    const int dim = 3;  // templating is hard here because mesh constructors
//    // static adept::Stack diff_stack;
//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 2;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
//                               Element::TETRAHEDRON, true /* gen. edges */, 1.0,
//                               1.0, 1.0, true));
//    mesh->EnsureNodes();
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {
//          // get the finite-element space for the current grid function
//          std::unique_ptr<FiniteElementCollection> nd_fec(
//             new ND_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> nd_fes(new FiniteElementSpace(
//             mesh.get(), nd_fec.get()));


//          // initialize current source and adjoint
//          // here we randomly perturb a constant state
//          GridFunction c(nd_fes.get()), adjoint(nd_fes.get());
//          VectorFunctionCoefficient pert(3, randVectorState);
//          VectorFunctionCoefficient current(3, vectorFunc, vectorFuncRevDiff);
//          adjoint.ProjectCoefficient(pert);

//          // extract mesh nodes and get their finite-element space
//          GridFunction *x_nodes = mesh->GetNodes();
//          FiniteElementSpace *mesh_fes = x_nodes->FESpace();

//          // build the linear form for d(psi^T R)/dx 
//          LinearForm dfdx(mesh_fes);
//          dfdx.AddDomainIntegrator(
//             new miso::GridFuncMeshSensIntegrator(&adjoint, &current));
//          dfdx.Assemble();

//          GridFunction dFdX_gf(mesh_fes);
//          dFdX_gf.ProjectCoefficientRevDiff(adjoint, current);

//          // initialize the vector that we use to perturb the mesh nodes
//          GridFunction v(mesh_fes);
//          VectorFunctionCoefficient v_rand(3, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // contract dfdx with v
//          double dfdx_v = dfdx * v;

//          double dFdX_gf_v = dFdX_gf * v;
//          std::cout << "dFdX_gf_v " << dFdX_gf_v << "\n";


//          // now compute the finite-difference approximation...
//          GridFunction x_pert(*x_nodes);
//          x_pert.Add(delta, v);
//          mesh->SetNodes(x_pert);
//          nd_fes->Update();
//          c.Update();
//          c.ProjectCoefficient(current);
//          double dfdx_v_fd = adjoint * c;
//          x_pert.Add(-2 * delta, v);
//          mesh->SetNodes(x_pert);
//          nd_fes->Update();
//          c.Update();
//          c.ProjectCoefficient(current);
//          dfdx_v_fd -= adjoint * c;
//          dfdx_v_fd /= (2 * delta);
//          mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
//       }
//    }
// }

TEST_CASE("MagneticEnergyIntegrator::GetEnergy - 2D")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   // auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
   //                                   Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         auto nonlin_energy = [](double B) { return 1.0/3.0 * (sqrt(B+1) * (B-2) + 2); };

         H1_FECollection fec(p, dim);
         // ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient a_func([](const mfem::Vector &x){
            return 1.5*x(1) - 0.5*x(0);
         });
         a.ProjectCoefficient(a_func);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(nu));

         const double fun = functional.GetEnergy(a);
         const double b_mag = sqrt(pow(1.5, 2) + pow(0.5, 2));
         const double energy = nonlin_energy(b_mag);
         const double area = 1.0;
         // std::cout << "fun: " << fun << " energy * area: " << energy*area << "\n";
         REQUIRE(fun == Approx(energy * area));
      }
   }
}

TEST_CASE("MagneticEnergyIntegrator::GetEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {

         auto nonlin_energy = [](double B) { return 1.0/3.0 * (sqrt(B+1) * (B-2) + 2); };

         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         GridFunction A(fes.get());
         VectorFunctionCoefficient pert(3, [](const Vector &x, Vector &a)
         {
            a(0) = 1.5*x(1);
            a(1) = -0.5*x(0);
            a(2) = 0.0;
         });
         A.ProjectCoefficient(pert);

         std::unique_ptr<miso::StateCoefficient> nu(
            new NonLinearCoefficient());

         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(*nu));

         const double fun = functional.GetEnergy(A);
         const double b_mag = 2.0;
         const double energy = nonlin_energy(b_mag);
         const double vol = 6.0;
         // std::cout << "fun: " << fun << " energy * vol: " << energy*vol << "\n";
         REQUIRE(fun == Approx(energy * vol));
      }
   }
}

TEST_CASE("MagneticEnergyIntegrator::AssembleElementVector - 2D")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   // auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
   //                                   Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         // ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(nu));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("MagneticEnergyIntegrator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true);
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(nu));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(A, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(A);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("MagneticEnergyIntegratorMeshSens::AssembleRHSElementVect - 2D")
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

   NonLinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new miso::MagneticEnergyIntegrator(nu);
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
            new miso::MagneticEnergyIntegratorMeshSens(a, *integ));
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

TEST_CASE("MagneticEnergyIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true);
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         auto *integ = new miso::MagneticEnergyIntegrator(nu);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::MagneticEnergyIntegratorMeshSens(A, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         double delta = 1e-5;
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(A);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(A);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

/** commenting out co-energy stuff since I'm stopping maintaining it
TEST_CASE("MagneticCoenergyIntegrator::AssembleElementVector",
          "[MagneticCoenergyIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge, Element::TETRAHEDRON,
                                       true, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         std::unique_ptr<miso::StateCoefficient> nu(
            // new LinearCoefficient());
            new NonLinearCoefficient());

         functional.AddDomainIntegrator(
            new miso::MagneticCoenergyIntegrator(q, nu.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         functional.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += functional.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);

         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("MagneticCoenergyIntegrator::AssembleElementRHSVect",
          "[MagneticCoenergyIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
                                       Element::TETRAHEDRON,
                                       true, 1.0, 1.0, 1.0, true));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         std::unique_ptr<miso::StateCoefficient> nu(
            new NonLinearCoefficient());

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::MagneticCoenergyIntegrator(q, nu.get()));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new miso::MagneticCoenergyIntegrator(q, nu.get()));

         double delta = 1e-5;
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         double dJdx_dot_v_fd = -functional.GetEnergy(q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(q);
         dJdx_dot_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}
*/

TEST_CASE("BNormIntegrator::GetElementEnergy",
          "[BNormIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 2.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         // initialize state; here we randomly perturb a constant state
         GridFunction A(fes.get());
         VectorFunctionCoefficient A_fun(3, [](const Vector &x, Vector &a)
         {
            a(0) = 1.5*x(1);
            a(1) = -0.5*x(0);
            a(2) = 0.0;
         });
         A.ProjectCoefficient(A_fun);

         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new miso::BNormIntegrator());


         const double fun = functional.GetEnergy(A);
         const double b_mag = 2.0;
         const double vol = 4.0;
         REQUIRE(fun == Approx(b_mag * vol));
      }
   }
}

TEST_CASE("BNormIntegrator::AssembleElementVector",
          "[BNormIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         const double scale = 0.01;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         functional.AddDomainIntegrator(
            new miso::BNormIntegrator());

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         functional.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += functional.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);

         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("BNormdJdX::AssembleRHSElementVect",
          "[BNormdJdX]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 3;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // initialize state; here we randomly perturb a constant state
         GridFunction A(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         A.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::BNormdJdx(A));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new miso::BNormIntegrator());

         double delta = 1e-5;
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         double dJdx_dot_v_fd = -functional.GetEnergy(A);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(A);
         dJdx_dot_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("nuBNormIntegrator::AssembleElementVector",
          "[nuBNormIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
            new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
            mesh.get(), fec.get()));

         NonlinearForm functional(fes.get());

         const double scale = 0.01;

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         std::unique_ptr<miso::StateCoefficient> nu(
            new NonLinearCoefficient());

         functional.AddDomainIntegrator(
            new miso::nuBNormIntegrator(nu.get()));

         // initialize the vector that dJdu multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(3, randVectorState);
         v.ProjectCoefficient(v_rand);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(fes.get());
         functional.Mult(q, dJdu);
         double dJdu_dot_v = InnerProduct(dJdu, v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q);
         q_pert.Add(-delta, v);
         double dJdu_dot_v_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, v);
         dJdu_dot_v_fd += functional.GetEnergy(q_pert);
         dJdu_dot_v_fd /= (2 * delta);

         REQUIRE(dJdu_dot_v == Approx(dJdu_dot_v_fd));
      }
   }
}

TEST_CASE("nuBNormdJdX::AssembleRHSElementVect",
          "[nuBNormdJdX]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         std::unique_ptr<miso::StateCoefficient> nu(
            new NonLinearCoefficient());

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::nuBNormdJdx(q, nu.get()));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new miso::nuBNormIntegrator(nu.get()));

         double delta = 1e-5;
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         double dJdx_dot_v_fd = -functional.GetEnergy(q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(q);
         dJdx_dot_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

TEST_CASE("nuFuncIntegrator::AssembleRHSElementVect",
          "[nuFuncIntegrator]")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true)));
   mesh->EnsureNodes();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state
         std::unique_ptr<FiniteElementCollection> fec(
             new ND_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // extract mesh nodes and get their finite-element space
         GridFunction *x_nodes = mesh->GetNodes();
         FiniteElementSpace *mesh_fes = x_nodes->FESpace();

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(3, randVectorState);
         q.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction v(mesh_fes);
         VectorFunctionCoefficient v_rand(dim, randVectorState);
         v.ProjectCoefficient(v_rand);

         std::unique_ptr<miso::StateCoefficient> nu(
            new NonLinearCoefficient());

         // evaluate dJdx and compute its product with v
         // GridFunction dJdx(*x_nodes);
         LinearForm dJdx(mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::nuFuncIntegrator(&q, nu.get()));
         dJdx.Assemble();
         double dJdx_dot_v = dJdx * v;

         // now compute the finite-difference approximation...
         NonlinearForm functional(fes.get());
         functional.AddDomainIntegrator(
            new miso::nuFuncIntegrator(nu.get()));

         double delta = 1e-5;
         GridFunction x_pert(*x_nodes);
         x_pert.Add(-delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         double dJdx_dot_v_fd = -functional.GetEnergy(q);
         x_pert.Add(2 * delta, v);
         mesh->SetNodes(x_pert);
         fes->Update();
         dJdx_dot_v_fd += functional.GetEnergy(q);
         dJdx_dot_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes
         fes->Update();

         REQUIRE(dJdx_dot_v == Approx(dJdx_dot_v_fd));
      }
   }
}

namespace miso
{
/// Compute the finite-difference approximation of the derivative of the
/// magnetic energy with respect to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the derivative of the magnetic energy with respect to B
double calcMagneticEnergyFD(
   mfem::ElementTransformation &trans,
   const mfem::IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   double delta = 1e-5;

   double fd_val;
   fd_val = calcMagneticEnergy(trans, ip, nu, B + delta);
   fd_val -= calcMagneticEnergy(trans, ip, nu, B - delta);
   return fd_val / (2*delta);
}

/// Compute the finite-difference approximation of the second derivative of the
/// magnetic energy with respect to B
/// \param[in] trans - element transformation for where to evaluate `nu`
/// \param[in] ip - integration point for where to evaluate `nu`
/// \param[in] nu - material dependent model describing reluctivity
/// \param[in] B - upper bound for integration
/// \return the second derivative of the magnetic energy with respect to B
double calcMagneticEnergyDoubleFD(
   mfem::ElementTransformation &trans,
   const mfem::IntegrationPoint &ip,
   StateCoefficient &nu,
   double B)
{
   double delta = 1e-5;

   double fd_val;
   fd_val = calcMagneticEnergyDot(trans, ip, nu, B + delta);
   fd_val -= calcMagneticEnergyDot(trans, ip, nu, B - delta);
   return fd_val / (2*delta);
}

}

TEST_CASE("calcMagneticEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   mesh->EnsureNodes();

   const double lin_nu_val = 0.42;
   std::unique_ptr<miso::StateCoefficient> lin_nu(new LinearCoefficient(lin_nu_val));
   std::unique_ptr<miso::StateCoefficient> nu(new NonLinearCoefficient());

   /// construct elements
   int p = 1;
   ND_FECollection fec(p, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   const double B_mag = 1.3;

   auto nonlin_energy = [](double B) { return 1.0/3.0 * (sqrt(B+1) * (B-2) + 2); };

   for (int j = 0; j < fes.GetNE(); j++)
   {
      const FiniteElement &el = *fes.GetFE(j);

      IsoparametricTransformation trans;
      mesh->GetElementTransformation(j, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double lin_en = miso::calcMagneticEnergy(trans, ip, *lin_nu, B_mag);
         REQUIRE(lin_en == Approx(0.5*pow(B_mag, 2)*lin_nu_val).epsilon(1e-8));

         double en = miso::calcMagneticEnergy(trans, ip, *nu, B_mag);
         REQUIRE(en == Approx(nonlin_energy(B_mag)).epsilon(1e-8));
         // std::cout << "lin_en: " << lin_en << " en: " << en << "\n";
      }
   }
}

TEST_CASE("calcMagneticEnergyDot")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   mesh->EnsureNodes();

   // std::unique_ptr<miso::StateCoefficient> nu(new LinearCoefficient(0.75));
   std::unique_ptr<miso::StateCoefficient> nu(new NonLinearCoefficient());

   /// construct elements
   int p = 1;
   ND_FECollection fec(p, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   const double B_mag = 0.5;

   for (int j = 0; j < fes.GetNE(); j++)
   {
      const FiniteElement &el = *fes.GetFE(j);

      IsoparametricTransformation trans;
      mesh->GetElementTransformation(j, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double dWdB = miso::calcMagneticEnergyDot(trans, ip, *nu, B_mag);
         double dWdB_fd = miso::calcMagneticEnergyFD(trans, ip, *nu, B_mag);
         REQUIRE(dWdB == Approx(dWdB_fd).epsilon(1e-10));
      }
   }
}

TEST_CASE("calcMagneticEnergyDoubleDot")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     1.0, 1.0, 1.0, true)));
   mesh->EnsureNodes();

   // std::unique_ptr<miso::StateCoefficient> nu(new LinearCoefficient(0.75));
   std::unique_ptr<miso::StateCoefficient> nu(new NonLinearCoefficient());

   /// construct elements
   int p = 1;
   ND_FECollection fec(p, dim);
   FiniteElementSpace fes(mesh.get(), &fec);

   const double B_mag = 0.25;

   for (int j = 0; j < fes.GetNE(); j++)
   {
      const FiniteElement &el = *fes.GetFE(j);

      IsoparametricTransformation trans;
      mesh->GetElementTransformation(j, &trans);

      int order = trans.OrderW() + 2 * el.GetOrder();
      const IntegrationRule *ir = &IntRules.Get(el.GetGeomType(), order);

      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         trans.SetIntPoint(&ip);

         double d2WdB2 = miso::calcMagneticEnergyDoubleDot(trans, ip, *nu, B_mag);
         double d2WdB2_fd = miso::calcMagneticEnergyDoubleFD(trans, ip, *nu, B_mag);
         REQUIRE(d2WdB2 == Approx(d2WdB2_fd).epsilon(1e-10));
      }
   }
}

TEST_CASE("DCLossFunctionalIntegrator: Resistivity for Analytical Temperature Field")
{
   using namespace mfem;
   using namespace electromag_data;

   //Create a temperature grid function
   //First, make a function coefficient
   //Then, project it onto a square 2D domain
   //Lastly, compare the resistivity solution found using the integrator to using assert

   //First case will be a simple 2D domain with a constant temperature of 20 deg C
   //The resistivity recovered should be 1/sigma_T_ref = 1/5.6497e7 = 1.770E-8 Ohm*m

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   //Function Coefficient model Representing the Temperature Field
   FunctionCoefficient model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double T = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            T = 37+273.15; //constant temperature throughout mesh
            // T = 77*x(0)+273.15; // temperature linearly dependent in the x(0) direction
            // T = 63*x(1)+273.15; // temperature linearly dependent in the x(1) direction
            // T = 30*std::pow(x(0),2)+273.15; // temperature quadratically dependent in the x(0) direction
            // T = 77*x(0)+63*x(1)+273.15; // temperature linearly dependent in both x(0) and x(1) directions
            // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2)+273.15; // temperature quadratically dependent in both x(0) and x(1) directions

         }
         return T;
      });

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {

         // Create the finite element collection and finite element space for the current order
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // Create the temperature_field grid function by mapping the function coefficient to a grid function
         GridFunction temperature_field_test(&fes);
         temperature_field_test.ProjectCoefficient(model);
         
         NonLinearCoefficient sigma(-1.0);

         // Define the functional integrator that will be used to compute the resistivity
         auto *integ = new miso::DCLossFunctionalIntegrator(sigma, temperature_field_test);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);                 

         // Compute the resistivity
         auto resistivity = functional.GetEnergy(temperature_field_test);
         // std::cout << "resistivity = " << resistivity << "\n";
 
         // Compare the computed resistivity to the value found by integration
         double expected_resistivity;
         // double alpha_resistivity = 3.8e-3; // only needed if using SigmaCoefficient
         // double T_ref = 293.15; // only needed if using SigmaCoefficient
         // double sigma_T_ref = 5.6497e7; // only needed if using SigmaCoefficient
         double state; // the "average" temperature over the simple 2D domain. (from either a temperature field, else the default temperature)
         state = 37.0+273.15; //constant temperature throughout mesh
         // state = 77.0/2+273.15; // temperature linearly dependent in the x(0) direction
         // state = 63.0/2+273.15; // temperature linearly dependent in the x(1) direction
         // state = 30.0/3+273.15; // temperature quadratically dependent in the x(0) direction; fails for p=1 as expected
         // state = 77.0/2+63.0/2+273.15; // temperature linearly dependent in both x(0) and x(1) directions
         // state = 30.0/3+3.0/3+273.15; // temperature linearly dependent in both x(0) and x(1) directions; fails for p=1 as expected

         // expected_resistivity = (1+alpha_resistivity*(state-T_ref))/sigma_T_ref; // the inverse of equation for sigma (for SigmaCoefficient)
         expected_resistivity = 2*state+2; // for NonLinearCoefficient

         // std::cout << "expected_resistivity = " << expected_resistivity << "\n";
         // std::cout << "diff = " << resistivity-expected_resistivity << "\n";
         REQUIRE(resistivity == Approx(expected_resistivity)); // Assert the DCLossFunctionalIntegrator is working as expected

      }
   }
}

TEST_CASE("DCLossFunctionalIntegratorMeshSens::AssembleRHSElementVect")
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

   NonLinearCoefficient sigma;

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection mvp_fec(p, dim);
         FiniteElementSpace mvp_fes(&mesh, &mvp_fec);

         // initialize peak mvp state
         GridFunction mvp(&mvp_fes);
         FunctionCoefficient pert(randState);
         mvp.ProjectCoefficient(pert);

         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         auto *integ = new miso::DCLossFunctionalIntegrator(sigma, temp);
         NonlinearForm functional(&mvp_fes);
         functional.AddDomainIntegrator(integ);
         
         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         p.ProjectCoefficient(v_pert);
         
         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::DCLossFunctionalIntegratorMeshSens(mvp, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         double dJdx_dot_p_fd = -functional.GetEnergy(mvp);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         dJdx_dot_p_fd += functional.GetEnergy(mvp);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         
         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("DCLossFunctionalIntegratorTemperatureSens::AssembleRHSElementVect")
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

   NonLinearCoefficient sigma;

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection mvp_fec(p, dim);
         FiniteElementSpace mvp_fes(&mesh, &mvp_fec);

         // initialize peak mvp state
         GridFunction mvp(&mvp_fes);
         FunctionCoefficient pert(randState);
         mvp.ProjectCoefficient(pert);

         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         auto *integ = new miso::DCLossFunctionalIntegrator(sigma, temp);
         NonlinearForm functional(&mvp_fes);
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdx multiplies
         GridFunction p(&temp_fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with pert
         LinearForm dJdu(&temp_fes);
         dJdu.AddDomainIntegrator(
            new miso::DCLossFunctionalIntegratorTemperatureSens(mvp, *integ));
         dJdu.Assemble();
         double dJdu_dot_p = dJdu * p;

         // now compute the finite-difference approximation...
         temp.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(mvp);
         temp.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(mvp);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));

      }
   }
}

TEST_CASE("DCLossDistributionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0}
         });
         res.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorMeshRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorTemperatureRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0}
         });
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the temp
         GridFunction v(&temp_fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorTemperatureRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         temp.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         temp.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorStrandRadiusRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorStrandRadiusRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"strand_radius", sqrt(1/M_PI) + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"strand_radius", sqrt(1/M_PI) - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorWireLengthRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorWireLengthRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"wire_length", 1.0 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"wire_length", 1.0 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorStrandsInHandRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorStrandsInHandRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"strands_in_hand", 1.0 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"strands_in_hand", 1.0 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorCurrentRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorCurrentRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"rms_current", sqrt(1.0/sqrt(2.0)) + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"rms_current", sqrt(1.0/sqrt(2.0)) - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorStackLengthRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0},
            {"stack_length", 0.34}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorStackLengthRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"stack_length", 0.34 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"stack_length", 0.34 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("DCLossDistributionIntegratorWindingVolumeRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         FunctionCoefficient pert(randState);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         GridFunction adjoint(&temp_fes);
         adjoint.ProjectCoefficient(pert);
         
         LinearForm res(&temp_fes);
         auto *integ = new miso::DCLossDistributionIntegrator(sigma, temp);
         setInputs(*integ, {
            {"wire_length", 1.0},
            {"rms_current", sqrt(1.0/sqrt(2.0))},
            {"strand_radius", sqrt(1/M_PI)},
            {"strands_in_hand", 1.0},
            {"winding_volume", 0.34}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&temp_fes);
         dfdx.AddDomainIntegrator(
            new miso::DCLossDistributionIntegratorWindingVolumeRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"winding_volume", 0.34 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"winding_volume", 0.34 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

// Added test case for ACLossFunctionalIntegrator (sigma*B^2 value)
TEST_CASE("ACLossFunctionalIntegrator::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   //Function Coefficient model Representing the B Field (flux density)
   FunctionCoefficient Bfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double B = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            B = 1.7; //constant flux density throughout mesh
            // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
            // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
            // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
            // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
            // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

         }
         return B;
      });

   //Function Coefficient model Representing the Temperature Field
   FunctionCoefficient Tfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double T = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            // T = 37+273.15; //constant temperature throughout mesh
            // T = 77*x(0)+273.15; // temperature linearly dependent in the x(0) direction
            // T = 63*x(1)+273.15; // temperature linearly dependent in the x(1) direction
            T = 30*std::pow(x(0),2)+273.15; // temperature quadratically dependent in the x(0) direction
            // T = 77*x(0)+63*x(1)+273.15; // temperature linearly dependent in both x(0) and x(1) directions
            // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2) +273.15; // temperature quadratically dependent in both x(0) and x(1) directions

         }
         return T;
      });

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {

         // Create the finite element collection and finite element space for the current order
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // Create the temperature_field grid function by mapping the function coefficient to a grid function
         GridFunction temp(&fes);
         temp.ProjectCoefficient(Tfield_model);

         // Create the flux density field (B) grid function by mapping the function coefficient to a grid function
         GridFunction flux_density_field(&fes);
         flux_density_field.ProjectCoefficient(Bfield_model);
         
         // Handling the coefficient for the sigma in the same way the StateCoefficient nu was handled in other tests
         // std::unique_ptr<miso::StateCoefficient> sigma(new SigmaCoefficient()); // using default parameters for alpha_resistivity, T_ref, and sigma_T_ref
         double state=30.0/3+273.15; // the "average" temperature over the simple 2D domain. (from either a temperature field, else the default temperature)
         // In order of temperature field, state=37.0+273.15, 77.0/2+273.15, 63.0/2+273.15, 30.0/3+273.15, 77.0/2+63.0/2+273.15, 30.0/3+3.0/3+273.15
         
         LinearCoefficient sigma(state);

         // Define the functional integrator that will be used to compute the sigma_b2
         auto *integ = new miso::ACLossFunctionalIntegrator(sigma, temp);
         // auto *integ = new miso::ACLossFunctionalIntegrator(*sigma); //confirms that ACLossFunctional integrator works as intended if don't pass in a temperature field
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);                 

         // Compute the value of sigma_b2 (sigma*B^2)
         auto sigma_b2 = functional.GetEnergy(flux_density_field);
         // std::cout << "p=" << p << "sigma_b2 = " << sigma_b2 << "\n";
 
         // Compare the computed sigma_b2 to the value found by integration (analytically derived, checked with WolframAlpha)
         double expected_sigma_b2;
         // double x0=1; // the upper limit for integration in the x(0) direction // only needed if using SigmaCoefficient
         // double x1=1; // the upper limit for integration in the x(1) direction // only needed if using SigmaCoefficient
         // double alpha_resistivity = 3.8e-3; // only needed if using SigmaCoefficient
         // double T_ref = 20+273.15; // only needed if using SigmaCoefficient
         // double sigma_T_ref = 5.6497e7; // only needed if using SigmaCoefficient
         // expected_sigma_b2 = (sigma_T_ref/(1+alpha_resistivity*(37+273.15-T_ref)))*std::pow(1.7,2)*x0*x1; // for B = 1.7; T = 37; passes for all degrees;         
         // expected_sigma_b2 = (1.0189201578057486e8)*std::pow(x0,3)*x1; // for B = 2.4*x(0); T = 37; fails for degree p=1 as expected (p=1sigma_b2 = 9.76465e+07, expected_sigma_b2 = 1.01892e+08, diff=-4.2455e+06)
         // expected_sigma_b2 = 1.5351283527722308e8; // for B = 1.7; T = 77*x(0); fails for degree p=1 (sigma_b2 = 1.53349e+08, expected_sigma_b2 = 1.53513e+08, diff=-163514)
         // expected_sigma_b2 = 1.0439171176578794e8; // for B = 2.4*x(0); T = 63*x(1); fails for degree p=1 (sigma_b2 = 9.5404e+07, expected_sigma_b2 = 9.55238e+07, diff=-119773)
         // expected_sigma_b2 = 2.0686903638877016e7; // for B = 1.1*x(1); T = 63*x(1); fails for degrees p=1 and p=2
         // expected_sigma_b2 = (9.552376479428893e7)*std::pow(x0,5)*x1; // for B = 3.0*std::pow(x(0),2); T = 37; fails for degrees p=1 and p=2
         // Temporarily changing ACLFI loss = (1e8/sigma_val) + b_mag...
         // expected_sigma_b2 = 100000000*std::pow(((sigma_T_ref)/(1+alpha_resistivity*(37-T_ref))),-1)+2.4/2; // for temporarily ACLFI loss, T=37, B = 2.4*x(0); passes for all orders, including p=1
         // expected_sigma_b2 = 3.5944368727543055; // for temporarily ACLFI loss, T=77*x(0), B = 1.7; passes for all orders, including p=1
         // expected_sigma_b2 = 3.094436872754305538347; // for temporarily ACLFI loss, T=77*x(0), B = 2.4*x(0); passes for all orders, including p=1
         // Have not tested other combinations of B and T fields using SigmaCoefficient

         expected_sigma_b2 = pow(1.7,2)*state; // for B=1.7 when using LinearCoefficient prescribed above
         // expected_sigma_b2 = pow(2.4,2)*state/3; // for B=2.4*x(0) when using LinearCoefficient prescribed above
         // expected_sigma_b2 = pow(3.0,2)*state/5; // for B=3.0*std::pow(x(0),2) when using LinearCoefficient prescribed above

         // std::cout << "p=" << p << "expected_sigma_b2 = " << expected_sigma_b2 << "\n";
         // std::cout << "p=" << p << "diff=" <<sigma_b2-expected_sigma_b2 << "\n";
         REQUIRE(sigma_b2 == Approx(expected_sigma_b2)); // Assert the ACLossFunctionalIntegrator is working as expected

      }
   }
}

TEST_CASE("ACLossFunctionalIntegratorMeshSens::AssembleRHSElementVect (2D)")
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

   // mfem::FunctionCoefficient model(
   //    [](const mfem::Vector &x)
   //    {
   //       double q = 0;
   //       for (int i = 0; i < x.Size(); ++i)
   //       {
   //          q += pow(x(i), 2);
   //          // q += sqrt(x(i));
   //          // q += pow(x(i), 5);
   //       }
   //       return q;
   //    },
   //    [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
   //    {
   //       for (int i = 0; i < x.Size(); ++i)
   //       {
   //          x_bar(i) += q_bar * 2 * x(i);
   //          // x_bar(i) += q_bar * 0.5 / sqrt(x(i));
   //          // x_bar(i) += q_bar * 5 * pow(x(i), 4);
   //       }
   //    });

   //Function Coefficient model Representing the B Field (flux density)
   FunctionCoefficient Bfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double B = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            B = 1.7; //constant flux density throughout mesh
            // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
            // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
            // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
            // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
            // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

         }
         return B;
      });

   //Function Coefficient model Representing the Temperature Field
   FunctionCoefficient Tfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double T = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            T = 37; //constant temperature throughout mesh
            // T = 77*x(0); // temperature linearly dependent in the x(0) direction
            // T = 63*x(1); // temperature linearly dependent in the x(1) direction
            // T = 30*std::pow(x(0),2); // temperature quadratically dependent in the x(0) direction
            // T = 77*x(0)+63*x(1); // temperature linearly dependent in both x(0) and x(1) directions
            // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2); // temperature quadratically dependent in both x(0) and x(1) directions

         }
         return T;
      });

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // mesh.SetCurvature(p);

         L2_FECollection fec(p, dim); // Keep the L2 elements or use other FEs?
         FiniteElementSpace fes(&mesh, &fec);

         ///NOTE: state initialization may no longer be necessary because of the flux density field
         // initialize state 
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // Create the temperature_field grid function by mapping the function coefficient to a grid function
         GridFunction temp(&fes);
         temp.ProjectCoefficient(Tfield_model);

         // Create the flux density field (B) grid function by mapping the function coefficient to a grid function
         GridFunction flux_density_field(&fes);
         flux_density_field.ProjectCoefficient(Bfield_model);
         
         NonLinearCoefficient sigma;

         // Define the primal integrator (ACLossFunctionalIntegrator)
         auto *integ = new miso::ACLossFunctionalIntegrator(sigma, temp);
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
         // dJdx.AddDomainIntegrator(
         //    new miso::ACLossFunctionalIntegratorMeshSens(a, *integ)); // pre-peak flux
         dJdx.AddDomainIntegrator(
            new miso::ACLossFunctionalIntegratorMeshSens(flux_density_field, *integ)); // the state is the peak flux
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;
         std::cout << "dJdx_dot_p=" << dJdx_dot_p << "\n";

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         // double dJdx_dot_p_fd = -functional.GetEnergy(a); // pre-peak flux
         double dJdx_dot_p_fd = -functional.GetEnergy(flux_density_field); // the state is the peak flux
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         // dJdx_dot_p_fd += functional.GetEnergy(a); // pre-peak flux
         dJdx_dot_p_fd += functional.GetEnergy(flux_density_field);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         std::cout << "dJdx_dot_p_fd=" << dJdx_dot_p_fd << "\n";
         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

TEST_CASE("ACLossFunctionalIntegratorPeakFluxSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   // auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
   //                                   Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   // mfem::ConstantCoefficient sigma(1.0);

   // //Function Coefficient model Representing the B Field (flux density)
   // FunctionCoefficient Bfield_model(
   //    [](const mfem::Vector &x)
   //    {
   //       // x will be the point in space
   //       double B = 0;
   //       for (int i = 0; i < x.Size(); ++i)
   //       {
   //          B = 1.7; //constant flux density throughout mesh
   //          // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
   //          // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
   //          // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
   //          // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
   //          // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

   //       }
   //       return B;
   //    });

   //Function Coefficient model Representing the Temperature Field
   FunctionCoefficient Tfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double T = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            // T = 37+273.15; //constant temperature throughout mesh
            T = 77*x(0)+273.15; // temperature linearly dependent in the x(0) direction
            // T = 63*x(1)+273.15; // temperature linearly dependent in the x(1) direction
            // T = 30*std::pow(x(0),2)+273.15; // temperature quadratically dependent in the x(0) direction
            // T = 77*x(0)+63*x(1)+273.15; // temperature linearly dependent in both x(0) and x(1) directions
            // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2) +273.15; // temperature quadratically dependent in both x(0) and x(1) directions

         }
         return T;
      });

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // L2_FECollection fec(p, dim);
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // Create the temperature_field grid function by mapping the function coefficient to a grid function
         GridFunction temperature_field_test(&fes);
         temperature_field_test.ProjectCoefficient(Tfield_model);

         // // Create the flux density field (B) grid function by mapping the function coefficient to a grid function
         // GridFunction flux_density_field(&fes);
         // flux_density_field.ProjectCoefficient(Bfield_model);

         // Handling the coefficient for the sigma in the same way the StateCoefficient nu was handled in other tests
         double state=77.0/2; // the "average" temperature over the simple 2D domain. (from either a temperature field, else the default temperature)
         // In order of temperature field, state=37.0, 77.0/2, 63.0/2, 30.0/3, 77.0/2+63.0/2, 30.0/3+3.0/3
         LinearCoefficient sigma(state);

         auto *integ = new miso::ACLossFunctionalIntegrator(sigma, temperature_field_test);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize state for the 
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // create v displacement field
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // initialize the vector that dJdx multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdu(&fes);
         dJdu.AddDomainIntegrator(
            new miso::ACLossFunctionalIntegratorPeakFluxSens(a, *integ));
         dJdu.Assemble();
         double dJdu_dot_p = dJdu * p;
         std::cout << "ACLFIPFS dJdu_dot_p = " << dJdu_dot_p <<  "\n";

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("ACLossFunctionalIntegratorTemperatureSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   // auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
   //                                   Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   // Loop over various degrees of elements (1 to 4)
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         // initialize peak flux state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         H1_FECollection temp_fec(p, dim);
         FiniteElementSpace temp_fes(&mesh, &temp_fec);

         // initialize temp state
         GridFunction temp(&temp_fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;

         auto *integ = new miso::ACLossFunctionalIntegrator(sigma, temp);
         NonlinearForm functional(&flux_fes);
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdx multiplies
         GridFunction p(&temp_fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with pert
         LinearForm dJdu(&temp_fes);
         dJdu.AddDomainIntegrator(
            new miso::ACLossFunctionalIntegratorTemperatureSens(flux, *integ));
         dJdu.Assemble();
         double dJdu_dot_p = dJdu * p;

         // now compute the finite-difference approximation...
         temp.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(flux);
         temp.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(flux);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));

      }
   }
}

TEST_CASE("ACLosslDistributionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   // LinearCoefficient sigma(1.0);
   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorMeshRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLosslDistributionIntegratorPeakFluxRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   // LinearCoefficient sigma(1.0);
   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&flux_fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&flux_fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorPeakFluxRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         // GridFunction r(&fes);
         flux.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         flux.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLosslDistributionIntegratorTemperatureRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   // LinearCoefficient sigma(1.0);
   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorTemperatureRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         // GridFunction r(&fes);
         temp.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         temp.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLossDistributionIntegratorFrequencyRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"frequency", 100.0},
            {"strand_radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorFrequencyRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"frequency", 100 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"frequency", 100 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLossDistributionIntegratorStrandRadiusRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"strand_radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorStrandRadiusRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"strand_radius", 0.345 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"strand_radius", 0.345 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLossDistributionIntegratorStrandsInHandRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"strand_radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorStrandsInHandRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"strands_in_hand", 1.0 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"strands_in_hand", 1.0 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLossDistributionIntegratorNumTurnsRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"strand_radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorNumTurnsRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"num_turns", 810.0 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"num_turns", 810.0 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLossDistributionIntegratorNumSlotsRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"strand_radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorNumSlotsRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"num_slots", 24.0 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"num_slots", 24.0 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("ACLossDistributionIntegratorWindingVolumeRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma(1.0);

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::ACLossDistributionIntegrator(flux, temp, sigma);
         setInputs(*integ, {
            {"freq", 100.0},
            {"strand_radius", 0.345},
            {"stack_length", 0.41},
            {"strands_in_hand", 1.0},
            {"num_turns", 810.0},
            {"num_slots", 24.0},
            {"winding_volume", 0.23}
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::ACLossDistributionIntegratorWindingVolumeRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"winding_volume", 0.23 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"winding_volume", 0.23 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("CAL2CoreLossDistributionIntegratorMeshRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   mfem::ConstantCoefficient rho(1.0);
   CAL2Coefficient kh;
   CAL2Coefficient ke;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::CAL2CoreLossDistributionIntegrator(rho, kh, ke, flux, temp);
         setInputs(*integ, {
            {"freq", 100.0},
            {"stack_length", 0.41},
         });
         res.AddDomainIntegrator(integ);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&mesh_fes);
         dfdx.AddDomainIntegrator(
            new miso::CAL2CoreLossDistributionIntegratorMeshRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(x_nodes);
         GridFunction r(&fes);
         x_pert.Add(delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh.SetNodes(x_pert);
         fes.Update();
         res.Update();
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("CAL2CoreLossDistributionIntegratorPeakFluxRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   mfem::ConstantCoefficient rho(1.0);
   CAL2Coefficient kh;
   CAL2Coefficient ke;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::CAL2CoreLossDistributionIntegrator(rho, kh, ke, flux, temp);
         setInputs(*integ, {
            {"freq", 100.0},
            {"stack_length", 0.41},
         });
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&flux_fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&flux_fes);
         dfdx.AddDomainIntegrator(
            new miso::CAL2CoreLossDistributionIntegratorPeakFluxRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         // GridFunction r(&fes);
         flux.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         flux.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("CAL2CoreLossDistributionIntegratorTemperatureRevSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();
   
   mfem::ConstantCoefficient rho(1.0);
   CAL2Coefficient kh;
   CAL2Coefficient ke;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::CAL2CoreLossDistributionIntegrator(rho, kh, ke, flux, temp);
         setInputs(*integ, {
            {"freq", 100.0},
            {"stack_length", 0.41},
         });
         res.AddDomainIntegrator(integ);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(&fes);
         v.ProjectCoefficient(pert);

         // evaluate d(psi^T R)/dx and contract with v
         LinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::CAL2CoreLossDistributionIntegratorTemperatureRevSens(adjoint, *integ));
         dfdx.Assemble();
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         // GridFunction r(&fes);
         temp.Add(delta, v);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         temp.Add(-2 * delta, v);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

TEST_CASE("CAL2CoreLossDistributionIntegratorFrequencyRevSens::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mfem::ConstantCoefficient rho(1.0);
   CAL2Coefficient kh;
   CAL2Coefficient ke;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         L2_FECollection flux_fec(p, dim);
         FiniteElementSpace flux_fes(&mesh, &flux_fec);

         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // initialize state
         GridFunction flux(&flux_fes);
         FunctionCoefficient pert(randState);
         flux.ProjectCoefficient(pert);

         GridFunction adjoint(&fes);
         adjoint.ProjectCoefficient(pert);

         GridFunction temp(&fes);
         temp.ProjectCoefficient(pert);
         temp += 10.0;
         
         LinearForm res(&fes);
         auto *integ = new miso::CAL2CoreLossDistributionIntegrator(rho, kh, ke, flux, temp);
         setInputs(*integ, {
            {"frequency", 100.0},
         });
         res.AddDomainIntegrator(integ);

         // evaluate d(psi^T R)/dx and contract with v
         NonlinearForm dfdx(&fes);
         dfdx.AddDomainIntegrator(
            new miso::CAL2CoreLossDistributionIntegratorFrequencyRevSens(adjoint, *integ));

         // random perturbation
         double v = 0.3042434;
         double dfdx_v = dfdx.GetEnergy(temp) * v;

         // now compute the finite-difference approximation...
         setInputs(*integ, {
            {"frequency", 100 + delta * v}
         });
         res.Assemble();
         double dfdx_v_fd = adjoint * res;

         setInputs(*integ, {
            {"frequency", 100 - delta * v}
         });
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }   
}

///NOTE: Dropped efforts for incomplete loss functional distribution integrator test case. Not the appropriate place, nor the most meaningful test if/when completed.
/*
// Added test case for ACLossFunctionalDistributionIntegrator
///TODO: Finish test case in conjunction with implementation itself
TEST_CASE("ACLossFunctionalDistributionIntegrator::AssembleRHSElementVect")
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

      // Not using the previous function coefficient model for sigma

   //Function Coefficient model Representing the B Field (flux density)
   FunctionCoefficient Bfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double B = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            B = 1.7; //constant flux density throughout mesh
            // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
            // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
            // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
            // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
            // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

         }
         return B;
      });

   //Function Coefficient model Representing the Temperature Field
   FunctionCoefficient Tfield_model(
      [](const mfem::Vector &x)
      {
         // x will be the point in space
         double T = 0;
         for (int i = 0; i < x.Size(); ++i)
         {
            T = 37; //constant temperature throughout mesh
            // T = 77*x(0); // temperature linearly dependent in the x(0) direction
            // T = 63*x(1); // temperature linearly dependent in the x(1) direction
            // T = 30*std::pow(x(0),2); // temperature quadratically dependent in the x(0) direction
            // T = 77*x(0)+63*x(1); // temperature linearly dependent in both x(0) and x(1) directions
            // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2); // temperature quadratically dependent in both x(0) and x(1) directions

         }
         return T;
      });

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // mesh.SetCurvature(p);

         L2_FECollection fec(p, dim); // Keep the L2 elements or use other FEs?
         FiniteElementSpace fes(&mesh, &fec);

         ///NOTE: state initialization may no longer be necessary because of the flux density field
         // initialize state 
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         // Create the temperature_field grid function by mapping the function coefficient to a grid function
         GridFunction temperature_field_test(&fes);
         temperature_field_test.ProjectCoefficient(Tfield_model);

         // Create the flux density field (B) grid function by mapping the function coefficient to a grid function
         GridFunction flux_density_field(&fes);
         flux_density_field.ProjectCoefficient(Bfield_model);
         
         // Handling the coefficient for the sigma in the same way the StateCoefficient nu was handled in other tests
         // std::unique_ptr<miso::StateCoefficient> sigma(new SigmaCoefficient()); // using default parameters for alpha_resistivity, T_ref, and sigma_T_ref
         std::unique_ptr<miso::StateCoefficient> sigma(new LinearCoefficient()); // no longer using SigmaCoefficient (unnecessary) 

         // No need for primal integrator

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
            new miso::ACLossFunctionalDistributionIntegrator(flux_density_field, *sigma, &temperature_field_test)); // if temperature field is present
         // dJdx.AddDomainIntegrator(
         //    new miso::ACLossFunctionalDistributionIntegrator(flux_density_field, *sigma, &temperature_field_test)); // if no temperature field
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;
         std::cout << "dJdx_dot_p=" << dJdx_dot_p << "\n";

         ///TODO: compute the finite-difference approximation or other value as needed to assert LinearForm
                 
         ///TODO: Add Assertion
         // std::cout << "dJdx_dot_p_fd=" << dJdx_dot_p_fd << "\n";
         // REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}
*/

TEST_CASE("ForceIntegrator3::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(dim, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction a(&fes);
         a.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::ForceIntegrator3(nu, v));

         auto force = functional.GetEnergy(a);

         NonlinearForm energy(&fes);
         energy.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(nu));
         
         add(x_nodes, -delta, v, x_nodes);
         auto dWds = -energy.GetEnergy(a);
         add(x_nodes, 2*delta, v, x_nodes);
         dWds += energy.GetEnergy(a);
         dWds /= 2*delta;

         // std::cout << "-dWds: " << -dWds << " Force: " << force << "\n";
         REQUIRE(force == Approx(-dWds));
      }
   }
}

TEST_CASE("ForceIntegrator3::GetElementEnergy - 2D")
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

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(dim, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction a(&fes);
         FunctionCoefficient apert(randState);
         a.ProjectCoefficient(apert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::ForceIntegrator3(nu, v));

         auto force = functional.GetEnergy(a);

         NonlinearForm energy(&fes);
         energy.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(nu));
         
         add(x_nodes, -delta, v, x_nodes);
         auto dWds = -energy.GetEnergy(a);
         add(x_nodes, 2*delta, v, x_nodes);
         dWds += energy.GetEnergy(a);
         dWds /= 2*delta;

         // std::cout << "-dWds: " << -dWds << " Force: " << force << "\n";
         REQUIRE(force == Approx(-dWds));
      }
   }
}

TEST_CASE("ForceIntegrator3::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(dim, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state
         GridFunction a(&fes);
         a.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::ForceIntegrator3(nu, v));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("ForceIntegrator3::AssembleElementVector - 2D")
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

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         // ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::ForceIntegrator3(nu, v));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(a, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(a);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("ForceIntegratorMeshSens3::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON);
   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction a(&fes);
         a.ProjectCoefficient(pert);

         auto *integ = new miso::ForceIntegrator3(nu, v);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::ForceIntegratorMeshSens3(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         double delta = 1e-5;
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

TEST_CASE("ForceIntegratorMeshSens3::AssembleRHSElementVect - 2D")
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

   NonLinearCoefficient nu;
   // LinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         H1_FECollection fec(p, dim);
         // ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         v.ProjectCoefficient(v_pert);

         // initialize state
         GridFunction a(&fes);
         FunctionCoefficient pert(randState);
         a.ProjectCoefficient(pert);

         auto *integ = new miso::ForceIntegrator3(nu, v);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(v_pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::ForceIntegratorMeshSens3(a, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         double delta = 1e-5;
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

TEST_CASE("ForceIntegrator::GetElementEnergy")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true);
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::ForceIntegrator(nu, v));

         auto force = functional.GetEnergy(A);

         NonlinearForm energy(&fes);
         energy.AddDomainIntegrator(
            new miso::MagneticEnergyIntegrator(nu));
         
         add(x_nodes, -delta, v, x_nodes);
         auto dWds = -energy.GetEnergy(A);
         add(x_nodes, 2*delta, v, x_nodes);
         dWds += energy.GetEnergy(A);
         dWds /= 2*delta;

         // std::cout << "-dWds: " << -dWds << " Force: " << force << "\n";
         REQUIRE(force == Approx(-dWds));
      }
   }
}

TEST_CASE("ForceIntegrator::AssembleElementVector")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   // int num_edge = 2;
   // auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
   //                                   Element::TETRAHEDRON,
   //                                   1.0, 1.0, 1.0, true);
   int num_edge = 1;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON);
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   /// construct elements
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(
            new miso::ForceIntegrator(nu, v));

         // initialize the vector that dJdu multiplies
         GridFunction p(&fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdu and compute its product with v
         GridFunction dJdu(&fes);
         functional.Mult(A, dJdu);
         double dJdu_dot_p = InnerProduct(dJdu, p);

         // now compute the finite-difference approximation...
         GridFunction q_pert(A);
         q_pert.Add(-delta, p);
         double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
         q_pert.Add(2 * delta, p);
         dJdu_dot_p_fd += functional.GetEnergy(q_pert);
         dJdu_dot_p_fd /= (2 * delta);

         REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
      }
   }
}

TEST_CASE("ForceIntegratorMeshSens::AssembleRHSElementVect")
{
   using namespace mfem;
   using namespace electromag_data;

   const int dim = 3;
   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 2;
   auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                     Element::TETRAHEDRON,
                                     2.0, 3.0, 1.0, true);
   mesh.EnsureNodes();

   NonLinearCoefficient nu;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         ND_FECollection fec(p, dim);
         FiniteElementSpace fes(&mesh, &fec);

         // extract mesh nodes and get their finite-element space
         auto &x_nodes = *mesh.GetNodes();
         auto &mesh_fes = *x_nodes.FESpace();

         // create v displacement field
         GridFunction v(&mesh_fes);
         VectorFunctionCoefficient pert(3, randVectorState);
         v.ProjectCoefficient(pert);

         // initialize state; here we randomly perturb a constant state
         GridFunction A(&fes);
         A.ProjectCoefficient(pert);

         auto *integ = new miso::ForceIntegrator(nu, v);
         NonlinearForm functional(&fes);
         functional.AddDomainIntegrator(integ);

         // initialize the vector that dJdx multiplies
         GridFunction p(&mesh_fes);
         p.ProjectCoefficient(pert);

         // evaluate dJdx and compute its product with p
         LinearForm dJdx(&mesh_fes);
         dJdx.AddDomainIntegrator(
            new miso::ForceIntegratorMeshSens(A, *integ));
         dJdx.Assemble();
         double dJdx_dot_p = dJdx * p;

         // now compute the finite-difference approximation...
         double delta = 1e-5;
         GridFunction x_pert(x_nodes);
         x_pert.Add(-delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         double dJdx_dot_p_fd = -functional.GetEnergy(A);
         x_pert.Add(2 * delta, p);
         mesh.SetNodes(x_pert);
         fes.Update();
         dJdx_dot_p_fd += functional.GetEnergy(A);
         dJdx_dot_p_fd /= (2 * delta);
         mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
         fes.Update();

         REQUIRE(dJdx_dot_p == Approx(dJdx_dot_p_fd));
      }
   }
}

/// commenting out because I removed support for nonlinear magnets for now
// TEST_CASE("MagnetizationIntegrator::AssembleElementGrad - Nonlinear", "[MagnetizationIntegrator]")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    const int dim = 3;  // templating is hard here because mesh constructors
//    // static adept::Stack diff_stack;
//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 1;
//    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, num_edge,
//                               Element::TETRAHEDRON, true /* gen. edges */, 1.0,
//                               1.0, 1.0, true));

//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {
//          std::unique_ptr<FiniteElementCollection> fec(
//             new ND_FECollection(p, dim));
//          std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
//             mesh.get(), fec.get()));

//          // initialize state; here we randomly perturb a constant state
//          GridFunction a(fes.get());
//          VectorFunctionCoefficient pert(3, randBaselineVectorPert);
//          a.ProjectCoefficient(pert);

//          std::unique_ptr<miso::StateCoefficient> nu(
//             new NonLinearCoefficient());

//          std::unique_ptr<mfem::VectorCoefficient> mag(
//             new mfem::VectorFunctionCoefficient(3, mag_func));

//          NonlinearForm res(fes.get());
//          res.AddDomainIntegrator(new miso::MagnetizationIntegrator(nu.get(),
//                                                                    mag.get()));

//          // initialize the vector that the Jacobian multiplies
//          GridFunction v(fes.get());
//          VectorFunctionCoefficient v_rand(3, randVectorState);
//          v.ProjectCoefficient(v_rand);

//          // evaluate the Jacobian and compute its product with v
//          Operator& Jac = res.GetGradient(a);
//          GridFunction jac_v(fes.get());
//          Jac.Mult(v, jac_v);

//          // now compute the finite-difference approximation...
//          GridFunction r(fes.get()), jac_v_fd(fes.get());
//          a.Add(-delta, v);
//          res.Mult(a, r);
//          a.Add(2*delta, v);
//          res.Mult(a, jac_v_fd);
//          jac_v_fd -= r;
//          jac_v_fd /= (2*delta);

//          for (int i = 0; i < jac_v.Size(); ++i)
//          {
//             REQUIRE( jac_v(i) == Approx(jac_v_fd(i)) );
//          }
//       }
//    }
// }

// // Added test case for PMDemagIntegrator::GetElementEnergy
// TEST_CASE("PMDemagIntegrator::GetElementEnergy")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    // Very similar structure to ACLossFunctionalIntegrator test
//    // Computes the integral of C(B,T)

//    // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
//    int num_edge = 2;
//    auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
//                                      Element::TRIANGLE);
//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    //Function Coefficient model Representing the B Field (flux density)
//    FunctionCoefficient Bfield_model(
//       [](const mfem::Vector &x)
//       {
//          // x will be the point in space
//          double B = 0;
//          for (int i = 0; i < x.Size(); ++i)
//          {
//             B = 1.7; //constant flux density throughout mesh
//             // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
//             // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
//             // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
//             // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
//             // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

//          }
//          return B;
//       });

//    //Function Coefficient model Representing the Temperature Field
//    FunctionCoefficient Tfield_model(
//       [](const mfem::Vector &x)
//       {
//          // x will be the point in space
//          double T = 0;
//          for (int i = 0; i < x.Size(); ++i)
//          {
//             T = 37; //constant temperature throughout mesh
//             // T = 77*x(0); // temperature linearly dependent in the x(0) direction
//             // T = 63*x(1); // temperature linearly dependent in the x(1) direction
//             // T = 30*std::pow(x(0),2); // temperature quadratically dependent in the x(0) direction
//             // T = 77*x(0)+63*x(1); // temperature linearly dependent in both x(0) and x(1) directions
//             // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2); // temperature quadratically dependent in both x(0) and x(1) directions

//          }
//          return T;
//       });

//    // Loop over various degrees of elements (1 to 4)
//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {

//          // Create the finite element collection and finite element space for the current order
//          H1_FECollection fec(p, dim);
//          FiniteElementSpace fes(&mesh, &fec);

//          // extract mesh nodes and get their finite-element space
//          auto &x_nodes = *mesh.GetNodes();
//          auto &mesh_fes = *x_nodes.FESpace();

//          // Create the temperature_field grid function by mapping the function coefficient to a grid function
//          GridFunction temperature_field_test(&fes);
//          temperature_field_test.ProjectCoefficient(Tfield_model);

//          // Create the flux density field (B) grid function by mapping the function coefficient to a grid function
//          GridFunction flux_density_field(&fes);
//          flux_density_field.ProjectCoefficient(Bfield_model);
         
//          // Using a simple two state coefficient from electromag_test_data for the constraint
//          std::unique_ptr<miso::TwoStateCoefficient> TwoStateCoefficient_Constraint(new SimpleTwoStateCoefficient());

//          // Define the functional integrator that will be used to compute C(B,T)
//          auto *integ = new miso::PMDemagIntegrator(*TwoStateCoefficient_Constraint,&temperature_field_test);
//          // auto *integ = new miso::PMDemagIntegrator(*TwoStateCoefficient_Constraint); // confirms that PMDemagIntegrator works as intended if don't pass in a temperature field
//          NonlinearForm functional(&fes);
//          functional.AddDomainIntegrator(integ);                 

//          // Compute the value of C(B,T)
//          auto C_B_T = functional.GetEnergy(flux_density_field);
//          // std::cout << "p=" << p << ",C(B,T) = " << C_B_T << "\n";
 
//          // Compare the computed C(B,T) to the value found by integration (analytically derived, checked with WolframAlpha)
//          double expected_C_B_T = pow(1.7,2)*37; // for B=1.7, T=37 (both constant), passes for all degrees as expected
//          // double expected_C_B_T = pow(1.7,2)*100; // for B=1.7 (constant), no temperature field (so T=100 const.), passes for all degrees as expected
//          // double expected_C_B_T = pow(1.7,2)*77/2; // for B=1.7, T=77*x(0), should pass for all degrees
//          // double expected_C_B_T = pow(2.4,2)*37/3; // for B=2.4*x(0), T=37, should pass for p=2,3,4
//          // double expected_C_B_T = pow(3.0,2)*37/5; // for B=3.0*std::pow(x(0),2), T=37, should pass for p=3,4
//          // double expected_C_B_T = pow(3.0,2)*77/6; // for B=3.0*std::pow(x(0),2), T=77*x(0), should pass for p=3,4
//          // double expected_C_B_T = pow(3.0,2)*30/7; // for B=3.0*std::pow(x(0),2), T=77*x(0), should pass for p=4 only

//          // std::cout << "p=" << p << ",C(B,T) = " << C_B_T << "\n";
//          // std::cout << "p=" << p << ",diff=" << C_B_T-expected_C_B_T << "\n";
//          REQUIRE(C_B_T == Approx(expected_C_B_T)); // Assert the PMDemagIntegrator is working as expected
//       }
//    }
// }

// // Added test case for PMDemagIntegrator::AssembleElementVector
// TEST_CASE("PMDemagIntegrator::AssembleElementVector")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    // Similar structure to ACLossFunctionalDisributionIntegrator test
//    // Also adapting from MagneticEnergyIntegrator::AssembleElementVector - 2D
//    // Computes the distribution of C(B,T)

//    std::cout << "PMDemagIntegrator::AssembleElementVector.................................\n"; 
//    double delta = 1e-5;

//    // generate a 8 element mesh
//    int num_edge = 2;
//    auto mesh = Mesh::MakeCartesian2D(num_edge, num_edge,
//                                      Element::TRIANGLE);
//    // auto mesh = Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
//    //                                   Element::TETRAHEDRON);
//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    //Function Coefficient model Representing the B Field (flux density)
//    FunctionCoefficient Bfield_model(
//       [](const mfem::Vector &x)
//       {
//          // x will be the point in space
//          double B = 0;
//          for (int i = 0; i < x.Size(); ++i)
//          {
//             B = 1.0; //constant flux density throughout mesh
//             // B = 2.4*x(0); // flux density linearly dependent in the x(0) direction
//             // B = 1.1*x(1); // flux density linearly dependent in the x(1) direction
//             // B = 3.0*std::pow(x(0),2); // flux density quadratically dependent in the x(0) direction
//             // B = 2.4*x(0)+1.1*x(1); // flux density linearly dependent in both x(0) and x(1) directions
//             // B = 3.0*std::pow(x(0),2) + 0.3*std::pow(x(1),2); // flux density quadratically dependent in both x(0) and x(1) directions

//          }
//          return B;
//       });

//    //Function Coefficient model Representing the Temperature Field
//    FunctionCoefficient Tfield_model(
//       [](const mfem::Vector &x)
//       {
//          // x will be the point in space
//          double T = 0;
//          for (int i = 0; i < x.Size(); ++i)
//          {
//             T = 10; //constant temperature throughout mesh
//             // T = 77*x(0); // temperature linearly dependent in the x(0) direction
//             // T = 63*x(1); // temperature linearly dependent in the x(1) direction
//             // T = 30*std::pow(x(0),2); // temperature quadratically dependent in the x(0) direction
//             // T = 77*x(0)+63*x(1); // temperature linearly dependent in both x(0) and x(1) directions
//             // T = 30*std::pow(x(0),2) + 3*std::pow(x(1),2); // temperature quadratically dependent in both x(0) and x(1) directions

//          }
//          return T;
//       });

//    ///TODO: Loop over various degrees of elements (1 to 4)
//    for (int p = 1; p <= 1; ++p)
//    {
//       DYNAMIC_SECTION("...for degree p = " << p)
//       {

//          // Create the finite element collection and finite element space for the current order
//          H1_FECollection fec(p, dim);
//          // ND_FECollection fec(p, dim);
//          FiniteElementSpace fes(&mesh, &fec);

//          // initialize state (not currently used, B field is)
//          GridFunction a(&fes);
//          FunctionCoefficient pert(randState);
//          a.ProjectCoefficient(pert);
//          std::cout << "a=np.array([";
//          for (int j = 0; j < a.Size(); j++) {std::cout << a.Elem(j) << ", ";}
//          std::cout << "])\n";

//          // Create the temperature_field grid function by mapping the function coefficient to a grid function
//          GridFunction temperature_field_test(&fes);
//          temperature_field_test.ProjectCoefficient(Tfield_model);

//          // Create the flux density field (B) grid function by mapping the function coefficient to a grid function
//          GridFunction flux_density_field(&fes);
//          flux_density_field.ProjectCoefficient(Bfield_model);

//          // Using a simple two state coefficient from electromag_test_data for the constraint
//          std::unique_ptr<miso::TwoStateCoefficient> TwoStateCoefficient_Constraint(new SimpleTwoStateCoefficient());

//          // Define the functional integrator that will be used to compute C(B,T) values for distribution
//          auto *integ = new miso::PMDemagIntegrator(*TwoStateCoefficient_Constraint,&temperature_field_test);
//          // auto *integ = new miso::PMDemagIntegrator(*TwoStateCoefficient_Constraint); // confirms that PMDemagIntegrator works as intended if don't pass in a temperature field
//          NonlinearForm functional(&fes);
//          functional.AddDomainIntegrator(integ);                 

//          // initialize the vector that dJdu multiplies (not used as of yet)
//          GridFunction p(&fes);
//          p.ProjectCoefficient(pert);
//          std::cout << "p=np.array([";
//          for (int j = 0; j < p.Size(); j++) {std::cout << p.Elem(j) << ", ";}
//          std::cout << "])\n";
         
//          // evaluate dJdu and compute its product with v
//          GridFunction dJdu(&fes);
//          functional.Mult(flux_density_field, dJdu);
//          double dJdu_dot_p = InnerProduct(dJdu, p);

//          ///TODO: Add an appropriate assertion. As is, the assertion and logic from MagneticEnergyIntegrator::AssembleElementVector - 2D test case do not carry over
//          // // now compute the finite-difference approximation...
//          // GridFunction q_pert(a);
//          // q_pert.Add(-delta, p);
//          // double dJdu_dot_p_fd = -functional.GetEnergy(q_pert);
//          // q_pert.Add(2 * delta, p);
//          // dJdu_dot_p_fd += functional.GetEnergy(q_pert);
//          // dJdu_dot_p_fd /= (2 * delta);

//          // std::cout << "dJdu_dot_p = " << dJdu_dot_p << "\n";
//          // std::cout << "dJdu_dot_p_fd = " << dJdu_dot_p_fd << "\n";
//          // REQUIRE(dJdu_dot_p == Approx(dJdu_dot_p_fd));
//       }
//    }
// }