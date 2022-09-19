#include <random>

#include "catch.hpp"
#include "finite_element_state.hpp"
#include "functional_output.hpp"
#include "magnetostatic_load.hpp"
#include "mfem.hpp"

#include "electromag_outputs.hpp"

#include "electromag_test_data.hpp"

TEST_CASE("DCLossFunctional sensitivity wrt mesh_coords")
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

         mach::DCLossFunctional fun(fields, model, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv}
         };

         // // initialize the vector that we use to perturb the mesh nodes
         // VectorFunctionCoefficient v_pert(dim, randVectorState);
         // mfem::Vector v_tv(mesh_coords.space().GetTrueVSize());
         // mesh_coords.project(v_pert, v_tv);
         // mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // // evaluate d(psi^T R)/dx and contract with v
         // setInputs(fun, inputs);
         // double dfdx_v = jacobianVectorProduct(fun, v_tv, "mesh_coords");

         // // now compute the finite-difference approximation...
         // mesh_coords_tv.Add(delta, v_tv);
         // double dfdx_v_fd_p = calcOutput(fun, inputs);
         // // std::cout << "dfdx_v_fd_p: " << dfdx_v_fd_p << "\n";

         // mesh_coords_tv.Add(-2 * delta, v_tv);
         // // mesh_coords_tv.Add(-delta, v_tv);
         // double dfdx_v_fd_m = calcOutput(fun, inputs);
         // // std::cout << "dfdx_v_fd_m: " << dfdx_v_fd_m << "\n";

         // double dfdx_v_fd = (dfdx_v_fd_p - dfdx_v_fd_m) / (2 * delta);

         // mesh_coords_tv.Add(delta, v_tv); // remember to reset the mesh nodes
         // std::cout << "dfdx_v: " << dfdx_v << " dfdx_v_fd: " << dfdx_v_fd << "\n";
         // REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-8));

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

TEST_CASE("ACLossFunctional sensitivity wrt strand_radius")
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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});

         double strand_radius = 1.0;
         mach::MachInputs inputs{
            {"strand_radius", strand_radius},
            {"peak_flux", peak_flux_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "strand_radius");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "strand_radius") * pert;

         // now compute the finite-difference approximation...
         inputs["strand_radius"] = strand_radius + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["strand_radius"] = strand_radius - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt frequency")
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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});

         double frequency = 1.0;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"peak_flux", peak_flux_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "frequency");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "frequency") * pert;

         // now compute the finite-difference approximation...
         inputs["frequency"] = frequency + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["frequency"] = frequency - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt stack_length")
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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});

         double stack_length = 1.0;
         mach::MachInputs inputs{
            {"stack_length", stack_length},
            {"peak_flux", peak_flux_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "stack_length");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "stack_length") * pert;

         // now compute the finite-difference approximation...
         inputs["stack_length"] = stack_length + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["stack_length"] = stack_length - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt strands_in_hand")
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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});

         double strands_in_hand = 1.0;
         mach::MachInputs inputs{
            {"strands_in_hand", strands_in_hand},
            {"peak_flux", peak_flux_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "strands_in_hand");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "strands_in_hand") * pert;

         // now compute the finite-difference approximation...
         inputs["strands_in_hand"] = strands_in_hand + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["strands_in_hand"] = strands_in_hand - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt num_turns")
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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});

         double num_turns = 1.0;
         mach::MachInputs inputs{
            {"num_turns", num_turns},
            {"peak_flux", peak_flux_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "num_turns");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "num_turns") * pert;

         // now compute the finite-difference approximation...
         inputs["num_turns"] = num_turns + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["num_turns"] = num_turns - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt num_slots")
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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});

         double num_slots = 1.0;
         mach::MachInputs inputs{
            {"num_slots", num_slots},
            {"peak_flux", peak_flux_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "num_slots");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "num_slots") * pert;

         // now compute the finite-difference approximation...
         inputs["num_slots"] = num_slots + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["num_slots"] = num_slots - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt mesh_coords")
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
         return 1.0;
         // return exp(-pow(x(0),2));
      },
      [](const mfem::Vector &x, const double q_bar, mfem::Vector &x_bar)
      {
         // x_bar(0) -= q_bar * 2 * x(0) * exp(-pow(x(0),2));
      });

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {

         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

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

         mach::ACLossFunctional fun(fields, model, {});
         mach::MachInputs inputs{
            {"state", peak_flux_tv},
            {"peak_flux", peak_flux_tv},
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
