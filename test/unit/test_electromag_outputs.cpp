#include <random>

#include "catch.hpp"
#include "finite_element_state.hpp"
#include "functional_output.hpp"
#include "mfem.hpp"

#include "electromag_outputs.hpp"

#include "electromag_test_data.hpp"

// TEST_CASE("DCLossFunctional::vectorJacobianProduct wrt mesh_coords")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 1;
//    auto smesh = Mesh::MakeCartesian2D(num_edge,
//                                       num_edge,
//                                       Element::TRIANGLE);
//    mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    mfem::FunctionCoefficient model(
//       [](const mfem::Vector &x)
//       {
//          // double q = 0;
//          // for (int i = 0; i < x.Size(); ++i)
//          // {

//          //    // q += pow(x(i), 2);
//          //    // q += sqrt(x(i));
//          //    // q += pow(x(i), 5);
//          // }
//          // return q;
//          return exp(-pow(x(0),2));
//       },
//       [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
//       {
//          // for (int i = 0; i < x.Size(); ++i)
//          // {
//          //    x_bar(i) += q_bar * 2 * x(i);
//          //    // x_bar(i) += q_bar * 0.5 / sqrt(x(i));
//          //    // x_bar(i) += q_bar * 5 * pow(x(i), 4);
//          // }
//          x_bar(0) -= q_bar * 2 * x(0) * exp(-pow(x(0),2));

//       });

//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {

//          mfem::H1_FECollection fec(p, dim);
//          mfem::ParFiniteElementSpace fes(&mesh, &fec);

//          std::map<std::string, mach::FiniteElementState> fields;
//          fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));

//          auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
//          auto *mesh_fespace = mesh_gf.ParFESpace();
//          /// create new state vector copying the mesh's fe space
//          fields.emplace("mesh_coords",
//                         mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
//          auto &mesh_coords = fields.at("mesh_coords");
//          /// set the values of the new GF to those of the mesh's old nodes
//          mesh_coords.gridFunc() = mesh_gf;
//          /// tell the mesh to use this GF for its Nodes
//          /// (and that it doesn't own it)
//          mesh.NewNodes(mesh_coords.gridFunc(), false);

//          mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
//          mesh_coords.setTrueVec(mesh_coords_tv);

//          mach::DCLossFunctional fun(fields, model, {});
//          mach::MachInputs inputs{
//             {"mesh_coords", mesh_coords_tv}
//          };

//          double dc_loss = calcOutput(fun, inputs);
//          std::cout << "dc_loss: " << dc_loss << "\n";

//          // initialize the vector that we use to perturb the mesh nodes
//          VectorFunctionCoefficient v_pert(dim, randVectorState);

//          mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
//          v_tv = 0.0;
//          v_tv(0) = 1.0;
//          // mesh_coords.project(v_pert, v_tv);

//          // mesh_coords.distributeSharedDofs(mesh_coords_tv);

//          mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
//          wrt_bar = 0.0;
//          // evaluate d(psi^T R)/dx and contract with v
//          mfem::Vector adjoint_tv(1);
//          adjoint_tv(0) = 1.0;
//          setInputs(fun, inputs);
//          vectorJacobianProduct(fun, adjoint_tv, "mesh_coords", wrt_bar);
//          double dfdx_v = wrt_bar * v_tv;

//          // now compute the finite-difference approximation...
//          mesh_coords_tv.Add(delta, v_tv);
//          double dfdx_v_fd_p = calcOutput(fun, inputs);
//          std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

//          mesh_coords_tv.Add(-2 * delta, v_tv);
//          // mesh_coords_tv.Add(-delta, v_tv);
//          double dfdx_v_fd_m = calcOutput(fun, inputs);
//          std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

//          double dfdx_v_fd = adjoint_tv(0) * (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

//          // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
//          std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
//       }
//    }
// }

// TEST_CASE("Resistivity::vectorJacobianProduct wrt mesh_coords")
// {
//    using namespace mfem;
//    using namespace electromag_data;

//    double delta = 1e-5;

//    // generate a 6 element mesh
//    int num_edge = 1;
//    auto smesh = Mesh::MakeCartesian2D(num_edge,
//                                       num_edge,
//                                       Element::TRIANGLE);
//    mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

//    mesh.EnsureNodes();
//    const auto dim = mesh.SpaceDimension();

//    mfem::FunctionCoefficient model(
//       [](const mfem::Vector &x)
//       {
//          // double q = 0;
//          // for (int i = 0; i < x.Size(); ++i)
//          // {

//          //    // q += pow(x(i), 2);
//          //    // q += sqrt(x(i));
//          //    // q += pow(x(i), 5);
//          // }
//          // return q;
//          return exp(-pow(x(0),2));
//       },
//       [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
//       {
//          // for (int i = 0; i < x.Size(); ++i)
//          // {
//          //    x_bar(i) += q_bar * 2 * x(i);
//          //    // x_bar(i) += q_bar * 0.5 / sqrt(x(i));
//          //    // x_bar(i) += q_bar * 5 * pow(x(i), 4);
//          // }
//          x_bar(0) -= q_bar * 2 * x(0) * exp(-pow(x(0),2));

//       });

//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {

//          mfem::H1_FECollection fec(p, dim);
//          mfem::ParFiniteElementSpace fes(&mesh, &fec);

//          std::map<std::string, mach::FiniteElementState> fields;
//          fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));

//          auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
//          auto *mesh_fespace = mesh_gf.ParFESpace();
//          /// create new state vector copying the mesh's fe space
//          fields.emplace("mesh_coords",
//                         mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
//          auto &mesh_coords = fields.at("mesh_coords");
//          /// set the values of the new GF to those of the mesh's old nodes
//          mesh_coords.gridFunc() = mesh_gf;
//          /// tell the mesh to use this GF for its Nodes
//          /// (and that it doesn't own it)
//          mesh.NewNodes(mesh_coords.gridFunc(), false);

//          mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
//          mesh_coords.setTrueVec(mesh_coords_tv);

//          mach::FunctionalOutput fun(fields.at("state").space(), fields);
//          fun.addOutputDomainIntegrator(new mach::DCLossFunctionalIntegrator(model));

//          mach::MachInputs inputs{
//             {"mesh_coords", mesh_coords_tv}
//          };

//          double dc_loss = calcOutput(fun, inputs);
//          std::cout << "dc_loss: " << dc_loss << "\n";

//          // initialize the vector that we use to perturb the mesh nodes
//          VectorFunctionCoefficient v_pert(dim, randVectorState);

//          mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
//          v_tv = 0.0;
//          v_tv(0) = 1.0;
//          // mesh_coords.project(v_pert, v_tv);

//          // mesh_coords.distributeSharedDofs(mesh_coords_tv);

//          mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
//          wrt_bar = 0.0;
//          // evaluate d(psi^T R)/dx and contract with v
//          mfem::Vector adjoint_tv(1);
//          adjoint_tv(0) = 1.0;
//          setInputs(fun, inputs);
//          vectorJacobianProduct(fun, adjoint_tv, "mesh_coords", wrt_bar);
//          double dfdx_v = wrt_bar * v_tv;

//          // now compute the finite-difference approximation...
//          mesh_coords_tv.Add(delta, v_tv);
//          double dfdx_v_fd_p = calcOutput(fun, inputs);
//          std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

//          mesh_coords_tv.Add(-2 * delta, v_tv);
//          // mesh_coords_tv.Add(-delta, v_tv);
//          double dfdx_v_fd_m = calcOutput(fun, inputs);
//          std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

//          double dfdx_v_fd = adjoint_tv(0) * (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

//          // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
//          std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
//          REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
//       }
//    }
// }

TEST_CASE("Resistivity::jacobianVectorProduct wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-6;

   // generate a 6 element mesh
   int num_edge = 1;
   auto smesh = Mesh::MakeCartesian2D(num_edge,
                                      num_edge,
                                      Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   mfem::FunctionCoefficient model(
      [](const mfem::Vector &x)
      {
         // double q = 0;
         // for (int i = 0; i < x.Size(); ++i)
         // {

         //    // q += pow(x(i), 2);
         //    // q += sqrt(x(i));
         //    // q += pow(x(i), 5);
         // }
         // return q;
         return exp(-pow(x(0),2));
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
         // for (int i = 0; i < x.Size(); ++i)
         // {
         //    x_bar(i) += q_bar * 2 * x(i);
         //    // x_bar(i) += q_bar * 0.5 / sqrt(x(i));
         //    // x_bar(i) += q_bar * 5 * pow(x(i), 4);
         // }
         x_bar(0) -= q_bar * 2 * x(0) * exp(-pow(x(0),2));

      });

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {

         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));

         auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
         auto *mesh_fespace = mesh_gf.ParFESpace();
         /// create new state vector copying the mesh's fe space
         fields.emplace("mesh_coords",
                        mach::FiniteElementState(mesh, *mesh_fespace, "mesh_coords"));
         auto &mesh_coords = fields.at("mesh_coords");
         /// set the values of the new GF to those of the mesh's old nodes
         mesh_coords.gridFunc() = mesh_gf;
         /// tell the mesh to use this GF for its Nodes
         /// (and that it doesn't own it)
         mesh.NewNodes(mesh_coords.gridFunc(), false);

         mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.setTrueVec(mesh_coords_tv);

         mach::FunctionalOutput fun(fields.at("state").space(), fields);
         fun.addOutputDomainIntegrator(new mach::DCLossFunctionalIntegrator(model));

         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv}
         };

         double dc_loss = calcOutput(fun, inputs);
         std::cout << "dc_loss: " << dc_loss << "\n";

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);

         mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
         v_tv = 0.0;
         v_tv(0) = 1.0;
         // mesh_coords.project(v_pert, v_tv);

         // mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(fun, inputs);
         double dfdx_v = jacobianVectorProduct(fun, v_tv, "mesh_coords");

         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(2*delta, v_tv);
         double dfdx_v_fd_p_2 = calcOutput(fun, inputs);

         mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_p = calcOutput(fun, inputs);
         std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

         mesh_coords_tv.Add(-2 * delta, v_tv);
         // mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_m = calcOutput(fun, inputs);
         std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

         mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_m_2 = calcOutput(fun, inputs);

         double dfdx_v_fd = (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);
         double dfdx_v_fd_4 = (-dfdx_v_fd_p_2 + 8*dfdx_v_fd_p - 8*dfdx_v_fd_m + dfdx_v_fd_m_2) / (12 * delta);

         // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << " dfdx_v_fd_4: " << dfdx_v_fd_4 << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}