#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "catch.hpp"
#include "mfem.hpp"

#include "electromag_test_data.hpp"

#include "common_outputs.hpp"

TEST_CASE("VolumeFunctional::jacobianVectorProduct wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

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
         return exp(-pow(x(0),2));
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
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

         mach::VolumeFunctional fun(fields, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, v_tv);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(fun, inputs);
         double dfdx_v = jacobianVectorProduct(fun, v_tv, "mesh_coords");

         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(delta, v_tv);
         double dfdx_v_fd_p = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

         mesh_coords_tv.Add(-2 * delta, v_tv);
         // mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_m = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

         double dfdx_v_fd = (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

         // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("VolumeFunctional::vectorJacobianProduct wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

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
         return exp(-pow(x(0),2));
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
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

         mach::VolumeFunctional fun(fields, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv}
         };

         // double dc_loss = calcOutput(fun, inputs);
         // std::cout << "dc_loss: " << dc_loss << "\n";

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);

         mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
         // v_tv = 0.0;
         // v_tv(0) = 1.0;
         mesh_coords.project(v_pert, v_tv);

         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
         wrt_bar = 0.0;
         // evaluate d(psi^T R)/dx and contract with v
         mfem::Vector adjoint_tv(1);
         adjoint_tv(0) = 1.0;
         setInputs(fun, inputs);
         vectorJacobianProduct(fun, adjoint_tv, "mesh_coords", wrt_bar);
         double dfdx_v = wrt_bar * v_tv;

         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(delta, v_tv);
         double dfdx_v_fd_p = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

         mesh_coords_tv.Add(-2 * delta, v_tv);
         // mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_m = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

         double dfdx_v_fd = adjoint_tv(0) * (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

         // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("MassFunctional::jacobianVectorProduct wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

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
         return exp(-pow(x(0),2));
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
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

         auto components = R"({
            "box1": {
               "attrs": [1],
               "material": "box1"
            }
         })"_json;

         auto materials = R"({
            "box1": {
               "rho": 1.0,
               "ks": 1.0,
               "alpha": 1.0,
               "beta": 1.0
            }
         })"_json;
         mach::MassFunctional fun(fields, components, materials, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, v_tv);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(fun, inputs);
         double dfdx_v = jacobianVectorProduct(fun, v_tv, "mesh_coords");

         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(delta, v_tv);
         double dfdx_v_fd_p = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

         mesh_coords_tv.Add(-2 * delta, v_tv);
         // mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_m = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

         double dfdx_v_fd = (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

         // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("MassFunctional::vectorJacobianProduct wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

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
         return exp(-pow(x(0),2));
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
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

         auto components = R"({
            "box1": {
               "attrs": [1],
               "material": "box1"
            }
         })"_json;

         auto materials = R"({
            "box1": {
               "rho": 1.0,
               "ks": 1.0,
               "alpha": 1.0,
               "beta": 1.0
            }
         })"_json;
         mach::MassFunctional fun(fields, components, materials, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv}
         };

         // double dc_loss = calcOutput(fun, inputs);
         // std::cout << "dc_loss: " << dc_loss << "\n";

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);

         mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
         // v_tv = 0.0;
         // v_tv(0) = 1.0;
         mesh_coords.project(v_pert, v_tv);

         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
         wrt_bar = 0.0;
         // evaluate d(psi^T R)/dx and contract with v
         mfem::Vector adjoint_tv(1);
         adjoint_tv(0) = 1.0;
         setInputs(fun, inputs);
         vectorJacobianProduct(fun, adjoint_tv, "mesh_coords", wrt_bar);
         double dfdx_v = wrt_bar * v_tv;

         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(delta, v_tv);
         double dfdx_v_fd_p = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

         mesh_coords_tv.Add(-2 * delta, v_tv);
         // mesh_coords_tv.Add(-delta, v_tv);
         double dfdx_v_fd_m = calcOutput(fun, inputs);
         // std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

         double dfdx_v_fd = adjoint_tv(0) * (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

         // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
         std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("StateAverageFunctional::calcOutput (3D)")
{
   int num_edge = 3;
   auto smesh = mfem::Mesh::MakeCartesian3D(num_edge, num_edge, num_edge,
                                            mfem::Element::TETRAHEDRON,
                                            2*M_PI, 1.0, 1.0, true);

   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   auto dim = mesh.Dimension();

   auto p = 1;
   mfem::H1_FECollection fec(p, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   std::map<std::string, mach::FiniteElementState> fields;

   fields.emplace(
      std::piecewise_construct,
      std::forward_as_tuple("state"),
      std::forward_as_tuple(mesh, fes, "state"));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();
   /// create new state vector copying the mesh's fe space
   fields.emplace(
         std::piecewise_construct,
         std::forward_as_tuple("mesh_coords"),
         std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));

   mach::StateAverageFunctional out(fes, fields);

   auto &state = fields.at("state");
   mfem::Vector state_tv(state.space().GetTrueVSize());

   state.project([](const mfem::Vector &p)
   {
      return sin(p(0)) * sin(p(0));
   }, state_tv);

   mach::MachInputs inputs{{"state", state_tv}};
   double rms = sqrt(calcOutput(out, inputs));

   REQUIRE(rms == Approx(sqrt(2)/2).margin(1e-10));
}

TEST_CASE("IEAggregateFunctional::calcOutput")
{
   auto smesh = mfem::Mesh::MakeCartesian2D(3, 3, mfem::Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   auto dim = mesh.Dimension();

   auto p = 2;

   // get the finite-element space for the state
   mfem::H1_FECollection fec(p, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   std::map<std::string, mach::FiniteElementState> fields;
   fields.emplace(
      std::piecewise_construct,
      std::forward_as_tuple("state"),
      std::forward_as_tuple(mesh, fes, "state"));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();
   /// create new state vector copying the mesh's fe space
   fields.emplace(
         std::piecewise_construct,
         std::forward_as_tuple("mesh_coords"),
         std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));

   auto &state = fields.at("state");
   mfem::Vector state_tv(state.space().GetTrueVSize());

   nlohmann::json output_opts{{"rho", 50.0}};
   mach::IEAggregateFunctional out(fes, fields, output_opts);

   state.project([](const mfem::Vector &p)
   {
      const double x = p(0);
      const double y = p(1);

      return - pow(x - 0.5, 2) - pow(x - 0.5, 2) + 1;
   }, state_tv);

   mach::MachInputs inputs{{"state", state_tv}};
   double max_state = calcOutput(out, inputs);

   /// Should be 1.0
   REQUIRE(max_state == Approx(0.9919141335));

   output_opts["rho"] = 1.0;
   setOptions(out, output_opts);

   max_state = calcOutput(out, inputs);
   /// Should be 1.0
   REQUIRE(max_state == Approx(0.8544376503));
}

TEST_CASE("IEAggregateFunctional sensitivity wrt state")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   auto smesh = Mesh::MakeCartesian2D(num_edge,
                                      num_edge,
                                      Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));

         auto &state = fields.at("state");
         mfem::Vector state_tv(state.space().GetTrueVSize());
         state.project(randState, state_tv);

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

         auto fun_opts = R"({
            "rho": 10
         })"_json;
         mach::IEAggregateFunctional fun(state.space(), fields, fun_opts);
         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv}
         };

         // initialize the vector that we use to perturb the state
         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         state.project(randState, pert_vec);
         state.distributeSharedDofs(state_tv);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "state");

         mfem::Vector wrt_bar(state.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(fun, adjoint_vec, "state", wrt_bar);
         double dfdp_rev = wrt_bar * pert_vec;


         // now compute the finite-difference approximation...
         state_tv.Add(delta, pert_vec);
         double dfdp_fd_p = calcOutput(fun, inputs);

         state_tv.Add(-2 * delta, pert_vec);
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         state_tv.Add(delta, pert_vec); // remember to reset the state
         std::cout << "dfdp_fwd: " << dfdp_fwd << " dfdp_fd: " << dfdp_fd << " dfdp_rev: " << dfdp_rev << "\n";

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("IEAggregateFunctional sensitivity wrt mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 6 element mesh
   int num_edge = 1;
   auto smesh = Mesh::MakeCartesian2D(num_edge,
                                      num_edge,
                                      Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));

         auto &state = fields.at("state");
         mfem::Vector state_tv(state.space().GetTrueVSize());
         state.project(randState, state_tv);

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

         auto fun_opts = R"({
            "rho": 10
         })"_json;
         mach::IEAggregateFunctional fun(state.space(), fields, fun_opts);
         mach::MachInputs inputs{
            {"state", state_tv},
            {"mesh_coords", mesh_coords_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector pert_vec(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, pert_vec);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "mesh_coords");

         mfem::Vector wrt_bar(mesh_coords.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(fun, adjoint_vec, "mesh_coords", wrt_bar);
         double dfdp_rev = wrt_bar * pert_vec;


         // now compute the finite-difference approximation...
         mesh_coords_tv.Add(delta, pert_vec);
         double dfdp_fd_p = calcOutput(fun, inputs);

         mesh_coords_tv.Add(-2 * delta, pert_vec);
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         mesh_coords_tv.Add(delta, pert_vec); // remember to reset the mesh nodes
         std::cout << "dfdp_fwd: " << dfdp_fwd << " dfdp_fd: " << dfdp_fd << " dfdp_rev: " << dfdp_rev << "\n";

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("IECurlMagnitudeAggregateFunctional::calcOutput")
{
   auto smesh = mfem::Mesh::MakeCartesian3D(3, 3, 3, mfem::Element::TETRAHEDRON);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();
   auto dim = mesh.Dimension();

   auto p = 2;

   // get the finite-element space for the state
   mfem::ND_FECollection fec(p, dim);
   mfem::ParFiniteElementSpace fes(&mesh, &fec);

   std::map<std::string, mach::FiniteElementState> fields;
   fields.emplace(
      std::piecewise_construct,
      std::forward_as_tuple("state"),
      std::forward_as_tuple(mesh, fes, "state"));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();
   /// create new state vector copying the mesh's fe space
   fields.emplace(
         std::piecewise_construct,
         std::forward_as_tuple("mesh_coords"),
         std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));

   auto &state = fields.at("state");
   mfem::Vector state_tv(state.space().GetTrueVSize());

   nlohmann::json output_opts{{"rho", 50.0}};
   mach::IECurlMagnitudeAggregateFunctional out(fes, fields, output_opts);

   state.project([](const mfem::Vector &p, mfem::Vector &A)
   {
      const double x = p(0);
      const double y = p(1);
      // const double z = p(2);

      A = 0.0;
      A(2) = sin(x) + cos(y);
   }, state_tv);

   mach::MachInputs inputs{{"state", state_tv}};
   double max_state = calcOutput(out, inputs);

   /// Should be sqrt(sin(1.0)^2 + 1.0)
   REQUIRE(max_state == Approx(1.2908573815));

   output_opts["rho"] = 1.0;
   setOptions(out, output_opts);

   max_state = calcOutput(out, inputs);
   /// Should be sqrt(sin(1.0)^2 + 1.0)
   REQUIRE(max_state == Approx(1.0152746915));

}
