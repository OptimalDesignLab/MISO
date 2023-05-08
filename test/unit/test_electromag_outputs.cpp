#include <random>

#include "catch.hpp"
#include "finite_element_state.hpp"
#include "finite_element_vector.hpp"
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossFunctional fun(fields, sigma, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
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

TEST_CASE("DCLossFunctional sensitivity wrt temperature")
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

   NonLinearCoefficient sigma;

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
         state_tv = 0.0;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossFunctional fun(fields, sigma, {});
         mach::MachInputs inputs{
            {"state", state_tv},
            {"peak_flux", peak_flux_tv},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb temperature
         FunctionCoefficient pert(randState);
         mfem::Vector pert_vec(temp.space().GetTrueVSize());
         temp.project(pert, pert_vec);
         temp.distributeSharedDofs(temp_tv);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "temperature");

         mfem::Vector wrt_bar(temp.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(fun, adjoint_vec, "temperature", wrt_bar);
         double dfdp_rev = wrt_bar * pert_vec;


         // now compute the finite-difference approximation...
         temp_tv.Add(delta, pert_vec);
         double dfdp_fd_p = calcOutput(fun, inputs);

         temp_tv.Add(-2 * delta, pert_vec);
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         temp_tv.Add(delta, pert_vec); // remember to reset the mesh nodes
         std::cout << "dfdp_fwd: " << dfdp_fwd << " dfdp_fd: " << dfdp_fd << " dfdp_rev: " << dfdp_rev << "\n";

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossFunctional sensitivity wrt wire_length")
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossFunctional fun(fields, sigma, {});

         double wire_length = 1.0;
         mach::MachInputs inputs{
            {"wire_length", wire_length},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "wire_length");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "wire_length") * pert;

         // now compute the finite-difference approximation...
         inputs["wire_length"] = wire_length + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["wire_length"] = wire_length - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossFunctional sensitivity wrt rms_current")
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossFunctional fun(fields, sigma, {});

         double rms_current = 1.0;
         mach::MachInputs inputs{
            {"rms_current", rms_current},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "rms_current");
         double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "rms_current") * pert;

         // now compute the finite-difference approximation...
         inputs["rms_current"] = rms_current + pert * delta;
         double dfdp_fd_p = calcOutput(fun, inputs);

         inputs["rms_current"] = rms_current - pert * delta;
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossFunctional sensitivity wrt strand_radius")
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossFunctional fun(fields, sigma, {});

         double strand_radius = 1.0;
         mach::MachInputs inputs{
            {"strand_radius", strand_radius},
            {"temperature", temp_tv}
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

TEST_CASE("DCLossFunctional sensitivity wrt strands_in_hand")
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossFunctional fun(fields, sigma, {});

         double strands_in_hand = 1.0;
         mach::MachInputs inputs{
            {"strands_in_hand", strands_in_hand},
            {"temperature", temp_tv}
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

TEST_CASE("DCLossDistribution::vectorJacobianProduct::mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector mesh_pert_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, mesh_pert_tv);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
         mesh_coords_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "mesh_coords", mesh_coords_bar);
         auto ddist_dmesh_v = mesh_pert_tv * mesh_coords_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         mesh_coords_tv.Add(delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dmesh_v_fd = adjoint_tv * dist_output;

         mesh_coords_tv.Add(-2 * delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dmesh_v_fd -= adjoint_tv * dist_output;

         ddist_dmesh_v_fd /= 2*delta;

         std::cout << "ddist_dmesh_v: " << ddist_dmesh_v << " ddist_dmesh_v_fd: " << ddist_dmesh_v_fd << "\n";
         REQUIRE(ddist_dmesh_v == Approx(ddist_dmesh_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossDistribution::vectorJacobianProduct::temperature")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector temp_pert_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_pert_tv);
         temp.distributeSharedDofs(temp_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector temperature_bar(temp.space().GetTrueVSize());
         temperature_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "temperature", temperature_bar);
         auto ddist_dtemp_v = temp_pert_tv * temperature_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         temp_tv.Add(delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dtemp_v_fd = adjoint_tv * dist_output;

         temp_tv.Add(-2 * delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dtemp_v_fd -= adjoint_tv * dist_output;

         ddist_dtemp_v_fd /= 2*delta;

         std::cout << "ddist_dtemp_v: " << ddist_dtemp_v << " ddist_dtemp_v_fd: " << ddist_dtemp_v_fd << "\n";
         REQUIRE(ddist_dtemp_v == Approx(ddist_dtemp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossDistribution sensitivity wrt wire_length")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});

         double wire_length = 1.0;
         mach::MachInputs inputs{
            {"wire_length", wire_length},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto wire_length_bar = vectorJacobianProduct(distribution, adjoint_tv, "wire_length");
         auto ddist_dp_v = pert * wire_length_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["wire_length"] = wire_length + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["wire_length"] = wire_length - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossDistribution sensitivity wrt rms_current")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});

         double rms_current = 1.0;
         mach::MachInputs inputs{
            {"rms_current", rms_current},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto rms_current_bar = vectorJacobianProduct(distribution, adjoint_tv, "rms_current");
         auto ddist_dp_v = pert * rms_current_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["rms_current"] = rms_current + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["rms_current"] = rms_current - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossDistribution sensitivity wrt strand_radius")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});

         double strand_radius = 1.0;
         mach::MachInputs inputs{
            {"strand_radius", strand_radius},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto strand_radius_bar = vectorJacobianProduct(distribution, adjoint_tv, "strand_radius");
         auto ddist_dp_v = pert * strand_radius_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["strand_radius"] = strand_radius + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["strand_radius"] = strand_radius - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossDistribution sensitivity wrt strands_in_hand")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});

         double strands_in_hand = 1.0;
         mach::MachInputs inputs{
            {"strands_in_hand", strands_in_hand},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto strands_in_hand_bar = vectorJacobianProduct(distribution, adjoint_tv, "strands_in_hand");
         auto ddist_dp_v = pert * strands_in_hand_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["strands_in_hand"] = strands_in_hand + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["strands_in_hand"] = strands_in_hand - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("DCLossDistribution sensitivity wrt stack_length")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::DCLossDistribution distribution(fields, sigma, {});

         double stack_length = 1.0;
         mach::MachInputs inputs{
            {"stack_length", stack_length},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto stack_length_bar = vectorJacobianProduct(distribution, adjoint_tv, "stack_length");
         auto ddist_dp_v = pert * stack_length_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["stack_length"] = stack_length + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["stack_length"] = stack_length - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});

         double strand_radius = 1.0;
         mach::MachInputs inputs{
            {"strand_radius", strand_radius},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});

         double frequency = 1.0;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
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

         std::cout << "dfdp_fwd: " << dfdp_fwd << " dfdp_fd: " << dfdp_fd << " dfdp_rev: " << dfdp_rev << "\n";
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});

         double stack_length = 1.0;
         mach::MachInputs inputs{
            {"stack_length", stack_length},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});

         double strands_in_hand = 1.0;
         mach::MachInputs inputs{
            {"strands_in_hand", strands_in_hand},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});

         double num_turns = 1.0;
         mach::MachInputs inputs{
            {"num_turns", num_turns},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});

         double num_slots = 1.0;
         mach::MachInputs inputs{
            {"num_slots", num_slots},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
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

   NonLinearCoefficient sigma;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});
         mach::MachInputs inputs{
            {"state", peak_flux_tv},
            {"peak_flux", peak_flux_tv},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
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

TEST_CASE("ACLossFunctional sensitivity wrt peak_flux")
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

   NonLinearCoefficient sigma;

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
         state_tv = 0.0;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});
         mach::MachInputs inputs{
            {"state", state_tv},
            {"peak_flux", peak_flux_tv},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb peak_flux
         FunctionCoefficient pert(randState);
         mfem::Vector pert_vec(peak_flux.space().GetTrueVSize());
         peak_flux.project(pert, pert_vec);
         peak_flux.distributeSharedDofs(peak_flux_tv);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "peak_flux");

         mfem::Vector wrt_bar(peak_flux.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(fun, adjoint_vec, "peak_flux", wrt_bar);
         double dfdp_rev = wrt_bar * pert_vec;


         // now compute the finite-difference approximation...
         peak_flux_tv.Add(delta, pert_vec);
         double dfdp_fd_p = calcOutput(fun, inputs);

         peak_flux_tv.Add(-2 * delta, pert_vec);
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         peak_flux_tv.Add(delta, pert_vec); // remember to reset the mesh nodes
         std::cout << "dfdp_fwd: " << dfdp_fwd << " dfdp_fd: " << dfdp_fd << " dfdp_rev: " << dfdp_rev << "\n";

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossFunctional sensitivity wrt temperature")
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

   NonLinearCoefficient sigma;

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
         state_tv = 0.0;

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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossFunctional fun(fields, sigma, {});
         mach::MachInputs inputs{
            {"state", state_tv},
            {"peak_flux", peak_flux_tv},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb peak_flux
         FunctionCoefficient pert(randState);
         mfem::Vector pert_vec(temp.space().GetTrueVSize());
         temp.project(pert, pert_vec);
         temp.distributeSharedDofs(temp_tv);

         double adjoint = randNumber();
         mfem::Vector adjoint_vec(&adjoint, 1);

         setInputs(fun, inputs);
         double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "temperature");

         mfem::Vector wrt_bar(temp.space().GetTrueVSize());
         wrt_bar = 0.0;
         vectorJacobianProduct(fun, adjoint_vec, "temperature", wrt_bar);
         double dfdp_rev = wrt_bar * pert_vec;


         // now compute the finite-difference approximation...
         temp_tv.Add(delta, pert_vec);
         double dfdp_fd_p = calcOutput(fun, inputs);

         temp_tv.Add(-2 * delta, pert_vec);
         double dfdp_fd_m = calcOutput(fun, inputs);

         double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

         temp_tv.Add(delta, pert_vec); // remember to reset the mesh nodes
         std::cout << "dfdp_fwd: " << dfdp_fwd << " dfdp_fd: " << dfdp_fd << " dfdp_rev: " << dfdp_rev << "\n";

         REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
         REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution::vectorJacobianProduct::mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector mesh_pert_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, mesh_pert_tv);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
         mesh_coords_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "mesh_coords", mesh_coords_bar);
         auto ddist_dmesh_v = mesh_pert_tv * mesh_coords_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         mesh_coords_tv.Add(delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dmesh_v_fd = adjoint_tv * dist_output;

         mesh_coords_tv.Add(-2 * delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dmesh_v_fd -= adjoint_tv * dist_output;

         ddist_dmesh_v_fd /= 2*delta;

         std::cout << "ddist_dmesh_v: " << ddist_dmesh_v << " ddist_dmesh_v_fd: " << ddist_dmesh_v_fd << "\n";
         REQUIRE(ddist_dmesh_v == Approx(ddist_dmesh_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution::vectorJacobianProduct::peak_flux")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector pert_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, pert_tv);
         peak_flux.distributeSharedDofs(peak_flux_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector peak_flux_bar(peak_flux.space().GetTrueVSize());
         peak_flux_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "peak_flux", peak_flux_bar);
         auto ddist_dflux_v = pert_tv * peak_flux_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         peak_flux_tv.Add(delta, pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dflux_v_fd = adjoint_tv * dist_output;

         peak_flux_tv.Add(-2 * delta, pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dflux_v_fd -= adjoint_tv * dist_output;

         ddist_dflux_v_fd /= 2*delta;

         std::cout << "ddist_dflux_v: " << ddist_dflux_v << " ddist_dflux_v_fd: " << ddist_dflux_v_fd << "\n";
         REQUIRE(ddist_dflux_v == Approx(ddist_dflux_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution::vectorJacobianProduct::temperature")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector temp_pert_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_pert_tv);
         temp.distributeSharedDofs(temp_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector temperature_bar(temp.space().GetTrueVSize());
         temperature_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "temperature", temperature_bar);
         auto ddist_dtemp_v = temp_pert_tv * temperature_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         temp_tv.Add(delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dtemp_v_fd = adjoint_tv * dist_output;

         temp_tv.Add(-2 * delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dtemp_v_fd -= adjoint_tv * dist_output;

         ddist_dtemp_v_fd /= 2*delta;

         std::cout << "ddist_dtemp_v: " << ddist_dtemp_v << " ddist_dtemp_v_fd: " << ddist_dtemp_v_fd << "\n";
         REQUIRE(ddist_dtemp_v == Approx(ddist_dtemp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution sensitivity wrt frequency")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});

         double frequency = 1.0;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto frequency_bar = vectorJacobianProduct(distribution, adjoint_tv, "frequency");
         auto ddist_dp_v = pert * frequency_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["frequency"] = frequency + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["frequency"] = frequency - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution sensitivity wrt strand_radius")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});

         double strand_radius = 1.0;
         mach::MachInputs inputs{
            {"strand_radius", strand_radius},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto strand_radius_bar = vectorJacobianProduct(distribution, adjoint_tv, "strand_radius");
         auto ddist_dp_v = pert * strand_radius_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["strand_radius"] = strand_radius + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["strand_radius"] = strand_radius - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution sensitivity wrt strands_in_hand")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});

         double strands_in_hand = 1.0;
         mach::MachInputs inputs{
            {"strands_in_hand", strands_in_hand},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto strands_in_hand_bar = vectorJacobianProduct(distribution, adjoint_tv, "strands_in_hand");
         auto ddist_dp_v = pert * strands_in_hand_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["strands_in_hand"] = strands_in_hand + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["strands_in_hand"] = strands_in_hand - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution sensitivity wrt num_turns")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});

         double num_turns = 1.0;
         mach::MachInputs inputs{
            {"num_turns", num_turns},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto num_turns_bar = vectorJacobianProduct(distribution, adjoint_tv, "num_turns");
         auto ddist_dp_v = pert * num_turns_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["num_turns"] = num_turns + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["num_turns"] = num_turns - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("ACLossDistribution sensitivity wrt num_slots")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         mach::ACLossDistribution distribution(fields, sigma, {});

         double num_slots = 1.0;
         mach::MachInputs inputs{
            {"num_slots", num_slots},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto num_slots_bar = vectorJacobianProduct(distribution, adjoint_tv, "num_slots");
         auto ddist_dp_v = pert * num_slots_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["num_slots"] = num_slots + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["num_slots"] = num_slots - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("CoreLossFunctional sensitivity wrt frequency")
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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         // mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
         // mesh_coords.setTrueVec(mesh_coords_tv);

         // auto components = R"({
         //    "box1": {
         //       "attrs": [1],
         //       "material": "box1"
         //    }
         // })"_json;

         // auto materials = R"({
         //    "box1": {
         //       "rho": 1.0,
         //       "ks": 1.0,
         //       "alpha": 1.0,
         //       "beta": 1.0
         //    }
         // })"_json;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         mach::CoreLossFunctional fun(fields, components, materials, {});

         double frequency = 2.0 + randNumber();
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"state", state_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
            // {"mesh_coords", mesh_coords_tv}
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

// TEST_CASE("CoreLossFunctional sensitivity wrt max_flux_magnitude")
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

//    for (int p = 1; p <= 4; ++p)
//    {
//       DYNAMIC_SECTION( "...for degree p = " << p )
//       {

//          mfem::H1_FECollection fec(p, dim);
//          mfem::ParFiniteElementSpace fes(&mesh, &fec);

//          std::map<std::string, mach::FiniteElementState> fields;
//          fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
//          auto &state = fields.at("state");
//          mfem::Vector state_tv(state.space().GetTrueVSize());
//          state.project(randState, state_tv);

//          fields.emplace("peak_flux",
//                         mach::FiniteElementState(mesh,
//                                                  {{"degree", p},
//                                                   {"basis-type", "dg"}},
//                                                  1,
//                                                  "peak_flux"));

//          auto &peak_flux = fields.at("peak_flux");
//          mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
//          peak_flux.project(randState, peak_flux_tv);

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

//          // mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
//          // mesh_coords.setTrueVec(mesh_coords_tv);

//          auto components = R"({
//             "box1": {
//                "attrs": [1],
//                "material": "box1"
//             }
//          })"_json;

//          auto materials = R"({
//             "box1": {
//                "rho": 1.0,
//                "ks": 1.0,
//                "alpha": 1.0,
//                "beta": 1.0
//             }
//          })"_json;

//          mach::CoreLossFunctional fun(fields, components, materials, {});

//          double max_flux_magnitude = 2.0 + randNumber();
//          mach::MachInputs inputs{
//             {"max_flux_magnitude", max_flux_magnitude},
//             {"state", state_tv},
//             {"peak_flux", peak_flux_tv},
//             // {"mesh_coords", mesh_coords_tv}
//          };

//          double pert = randNumber();
//          mfem::Vector pert_vec(&pert, 1);

//          double adjoint = randNumber();
//          mfem::Vector adjoint_vec(&adjoint, 1);

//          setInputs(fun, inputs);
//          double dfdp_fwd = adjoint * jacobianVectorProduct(fun, pert_vec, "max_flux_magnitude");
//          double dfdp_rev = vectorJacobianProduct(fun, adjoint_vec, "max_flux_magnitude") * pert;

//          // now compute the finite-difference approximation...
//          inputs["max_flux_magnitude"] = max_flux_magnitude + pert * delta;
//          double dfdp_fd_p = calcOutput(fun, inputs);

//          inputs["max_flux_magnitude"] = max_flux_magnitude - pert * delta;
//          double dfdp_fd_m = calcOutput(fun, inputs);

//          double dfdp_fd = adjoint * (dfdp_fd_p - dfdp_fd_m) / (2 * delta);

//          REQUIRE(dfdp_fwd == Approx(dfdp_fd).margin(1e-8));
//          REQUIRE(dfdp_rev == Approx(dfdp_fd).margin(1e-8));
//       }
//    }
// }

TEST_CASE("CoreLossFunctional sensitivity wrt mesh_coords")
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

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

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

         mach::CoreLossFunctional fun(fields, components, materials, {});
         mach::MachInputs inputs{
            {"state", peak_flux_tv},
            {"peak_flux", peak_flux_tv},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
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

TEST_CASE("CAL2CoreLossDistribution::vectorJacobianProduct::mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         mach::CAL2CoreLossDistribution distribution(fields, components, materials, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector mesh_pert_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, mesh_pert_tv);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
         mesh_coords_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "mesh_coords", mesh_coords_bar);
         auto ddist_dmesh_v = mesh_pert_tv * mesh_coords_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         mesh_coords_tv.Add(delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dmesh_v_fd = adjoint_tv * dist_output;

         mesh_coords_tv.Add(-2 * delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dmesh_v_fd -= adjoint_tv * dist_output;

         ddist_dmesh_v_fd /= 2*delta;

         std::cout << "ddist_dmesh_v: " << ddist_dmesh_v << " ddist_dmesh_v_fd: " << ddist_dmesh_v_fd << "\n";
         REQUIRE(ddist_dmesh_v == Approx(ddist_dmesh_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("CAL2CoreLossDistribution::vectorJacobianProduct::peak_flux")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         mach::CAL2CoreLossDistribution distribution(fields, components, materials, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector pert_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, pert_tv);
         peak_flux.distributeSharedDofs(peak_flux_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector peak_flux_bar(peak_flux.space().GetTrueVSize());
         peak_flux_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "peak_flux", peak_flux_bar);
         auto ddist_dflux_v = pert_tv * peak_flux_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         peak_flux_tv.Add(delta, pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dflux_v_fd = adjoint_tv * dist_output;

         peak_flux_tv.Add(-2 * delta, pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dflux_v_fd -= adjoint_tv * dist_output;

         ddist_dflux_v_fd /= 2*delta;

         std::cout << "ddist_dflux_v: " << ddist_dflux_v << " ddist_dflux_v_fd: " << ddist_dflux_v_fd << "\n";
         REQUIRE(ddist_dflux_v == Approx(ddist_dflux_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("CAL2CoreLossDistribution::vectorJacobianProduct::temperature")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         mach::CAL2CoreLossDistribution distribution(fields, components, materials, {});
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector temp_pert_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_pert_tv);
         temp.distributeSharedDofs(temp_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector temperature_bar(temp.space().GetTrueVSize());
         temperature_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "temperature", temperature_bar);
         auto ddist_dtemp_v = temp_pert_tv * temperature_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         temp_tv.Add(delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dtemp_v_fd = adjoint_tv * dist_output;

         temp_tv.Add(-2 * delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dtemp_v_fd -= adjoint_tv * dist_output;

         ddist_dtemp_v_fd /= 2*delta;

         std::cout << "ddist_dtemp_v: " << ddist_dtemp_v << " ddist_dtemp_v_fd: " << ddist_dtemp_v_fd << "\n";
         REQUIRE(ddist_dtemp_v == Approx(ddist_dtemp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("CAL2CoreLossDistribution sensitivity wrt frequency")
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
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         mach::CAL2CoreLossDistribution distribution(fields, components, materials, {});
         double frequency = 1.0;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"mesh_coords", mesh_coords_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto frequency_bar = vectorJacobianProduct(distribution, adjoint_tv, "frequency");
         auto ddist_dp_v = pert * frequency_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["frequency"] = frequency + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["frequency"] = frequency - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput::vectorJacobianProduct::mesh_coords")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);

         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         VectorFunctionCoefficient v_pert(dim, randVectorState);
         mfem::Vector mesh_pert_tv(mesh_coords.space().GetTrueVSize());
         mesh_coords.project(v_pert, mesh_pert_tv);
         mesh_coords.distributeSharedDofs(mesh_coords_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
         mesh_coords_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "mesh_coords", mesh_coords_bar);
         auto ddist_dmesh_v = mesh_pert_tv * mesh_coords_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         mesh_coords_tv.Add(delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dmesh_v_fd = adjoint_tv * dist_output;

         mesh_coords_tv.Add(-2 * delta, mesh_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dmesh_v_fd -= adjoint_tv * dist_output;

         ddist_dmesh_v_fd /= 2*delta;

         std::cout << "ddist_dmesh_v: " << ddist_dmesh_v << " ddist_dmesh_v_fd: " << ddist_dmesh_v_fd << "\n";
         REQUIRE(ddist_dmesh_v == Approx(ddist_dmesh_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput::vectorJacobianProduct::peak_flux")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector pert_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, pert_tv);
         peak_flux.distributeSharedDofs(peak_flux_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector peak_flux_bar(peak_flux.space().GetTrueVSize());
         peak_flux_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "peak_flux", peak_flux_bar);
         auto ddist_dflux_v = pert_tv * peak_flux_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         peak_flux_tv.Add(delta, pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dflux_v_fd = adjoint_tv * dist_output;

         peak_flux_tv.Add(-2 * delta, pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dflux_v_fd -= adjoint_tv * dist_output;

         ddist_dflux_v_fd /= 2*delta;

         std::cout << "ddist_dflux_v: " << ddist_dflux_v << " ddist_dflux_v_fd: " << ddist_dflux_v_fd << "\n";
         REQUIRE(ddist_dflux_v == Approx(ddist_dflux_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput::vectorJacobianProduct::temperature")
{
   using namespace mfem;
   using namespace electromag_data;

   double delta = 1e-5;

   // generate a 8 element mesh, simple 2D domain, 0<=x<=1, 0<=y<=1
   int num_edge = 2;
   auto smesh = Mesh::MakeCartesian2D(num_edge, num_edge,
                                     Element::TRIANGLE);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);

   mesh.EnsureNodes();
   const auto dim = mesh.SpaceDimension();

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         mach::MachInputs inputs{
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         // initialize the vector that we use to perturb the mesh nodes
         mfem::Vector temp_pert_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_pert_tv);
         temp.distributeSharedDofs(temp_tv);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         mfem::Vector temperature_bar(temp.space().GetTrueVSize());
         temperature_bar = 0.0;
         vectorJacobianProduct(distribution, adjoint_tv, "temperature", temperature_bar);
         auto ddist_dtemp_v = temp_pert_tv * temperature_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         temp_tv.Add(delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         double ddist_dtemp_v_fd = adjoint_tv * dist_output;

         temp_tv.Add(-2 * delta, temp_pert_tv);
         calcOutput(distribution, inputs, dist_output);
         ddist_dtemp_v_fd -= adjoint_tv * dist_output;

         ddist_dtemp_v_fd /= 2*delta;

         std::cout << "ddist_dtemp_v: " << ddist_dtemp_v << " ddist_dtemp_v_fd: " << ddist_dtemp_v_fd << "\n";
         REQUIRE(ddist_dtemp_v == Approx(ddist_dtemp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt wire_length")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double wire_length = 1.0;
         mach::MachInputs inputs{
            {"wire_length", wire_length},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto wire_length_bar = vectorJacobianProduct(distribution, adjoint_tv, "wire_length");
         auto ddist_dp_v = pert * wire_length_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["wire_length"] = wire_length + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["wire_length"] = wire_length - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt rms_current")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double rms_current = 1.0;
         mach::MachInputs inputs{
            {"rms_current", rms_current},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto rms_current_bar = vectorJacobianProduct(distribution, adjoint_tv, "rms_current");
         auto ddist_dp_v = pert * rms_current_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["rms_current"] = rms_current + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["rms_current"] = rms_current - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         std::cout << "current fd: " << ddist_dp_v << ", " << ddist_dp_v_fd << "\n";
         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt strand_radius")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double strand_radius = 1.0;
         mach::MachInputs inputs{
            {"strand_radius", strand_radius},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto strand_radius_bar = vectorJacobianProduct(distribution, adjoint_tv, "strand_radius");
         auto ddist_dp_v = pert * strand_radius_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["strand_radius"] = strand_radius + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["strand_radius"] = strand_radius - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt strands_in_hand")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double strands_in_hand = 1.0;
         mach::MachInputs inputs{
            {"strands_in_hand", strands_in_hand},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto strands_in_hand_bar = vectorJacobianProduct(distribution, adjoint_tv, "strands_in_hand");
         auto ddist_dp_v = pert * strands_in_hand_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["strands_in_hand"] = strands_in_hand + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["strands_in_hand"] = strands_in_hand - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt num_turns")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double num_turns = 1.0;
         mach::MachInputs inputs{
            {"num_turns", num_turns},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto num_turns_bar = vectorJacobianProduct(distribution, adjoint_tv, "num_turns");
         auto ddist_dp_v = pert * num_turns_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["num_turns"] = num_turns + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["num_turns"] = num_turns - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt num_slots")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double num_slots = 1.0;
         mach::MachInputs inputs{
            {"num_slots", num_slots},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto num_slots_bar = vectorJacobianProduct(distribution, adjoint_tv, "num_slots");
         auto ddist_dp_v = pert * num_slots_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["num_slots"] = num_slots + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["num_slots"] = num_slots - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt frequency")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double frequency = 1.0;
         mach::MachInputs inputs{
            {"frequency", frequency},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto frequency_bar = vectorJacobianProduct(distribution, adjoint_tv, "frequency");
         auto ddist_dp_v = pert * frequency_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["frequency"] = frequency + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["frequency"] = frequency - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}

TEST_CASE("EMHeatSourceOutput sensitivity wrt stack_length")
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

   NonLinearCoefficient sigma;

   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         mfem::H1_FECollection fec(p, dim);
         mfem::ParFiniteElementSpace fes(&mesh, &fec);

         std::map<std::string, mach::FiniteElementState> fields;
         fields.emplace("state", mach::FiniteElementState(mesh, fes, "state"));
         fields.emplace("adjoint", mach::FiniteElementState(mesh, fes, "adjoint"));
         auto &adjoint = fields.at("adjoint");

         mfem::Vector adjoint_tv(adjoint.space().GetTrueVSize());
         adjoint.project(randState, adjoint_tv);

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

         fields.emplace("peak_flux",
                        mach::FiniteElementState(mesh,
                                                 {{"degree", p},
                                                  {"basis-type", "dg"}},
                                                 1,
                                                 "peak_flux"));

         auto &peak_flux = fields.at("peak_flux");
         mfem::Vector peak_flux_tv(peak_flux.space().GetTrueVSize());
         peak_flux.project(randState, peak_flux_tv);
         peak_flux_tv += 2.0;

         fields.emplace("temperature", mach::FiniteElementState(mesh, {.order=p}));
         auto &temp = fields.at("temperature");

         mfem::Vector temp_tv(temp.space().GetTrueVSize());
         temp.project(randState, temp_tv);
         temp_tv *= 25;
         temp_tv += 200;

         auto components = R"({
            "test": {
               "attrs": [1],
               "material": {
                  "name": "test",
                  "core_loss": {
                     "model": "CAL2",
                     "T0": 293.15,
                     "kh_T0": [5.97783049251564E-02, -6.58569751792524E-02, 3.52052785575931E-02, -6.54762513683037E-03],
                     "ke_T0": [3.83147202762929E-05, -4.19965038193089E-05, 2.09788988466414E-05, -3.88567697029196E-06],
                     "T1": 473.15,
                     "kh_T1": [5.78728253280150E-02, -7.94684973286488E-02, 5.09165213772802E-02, -1.11117379956941E-02],
                     "ke_T1": [3.20525407302126E-05, -1.43502199723297E-05, -3.74786590271071E-06, 2.68517704958978E-06]
                  }
               }
            }
         })"_json;

         auto materials = R"({
            "test": {
               "rho": 1.0
            }
         })"_json;

         auto options = R"({
            "dc_loss": {
               "attributes": [1]
            },
            "ac_loss": {
               "attributes": [1]
            },
            "core_loss": {
               "attributes": [1]
            }
         })"_json;

         mach::EMHeatSourceOutput distribution(fields, sigma, components, materials, options);
         double stack_length = 1.0;
         mach::MachInputs inputs{
            {"stack_length", stack_length},
            {"mesh_coords", mesh_coords_tv},
            {"peak_flux", peak_flux_tv},
            {"temperature", temp_tv}
         };

         double pert = randNumber();
         mfem::Vector pert_vec(&pert, 1);

         // evaluate d(psi^T R)/dx and contract with v
         setInputs(distribution, inputs);
         auto stack_length_bar = vectorJacobianProduct(distribution, adjoint_tv, "stack_length");
         auto ddist_dp_v = pert * stack_length_bar;

         // now compute the finite-difference approximation...
         mfem::Vector dist_output(getSize(distribution));
         inputs["stack_length"] = stack_length + delta * pert;
         calcOutput(distribution, inputs, dist_output);
         double ddist_dp_v_fd = adjoint_tv * dist_output;

         inputs["stack_length"] = stack_length - delta * pert;
         calcOutput(distribution, inputs, dist_output);
         ddist_dp_v_fd -= adjoint_tv * dist_output;

         ddist_dp_v_fd /= 2*delta;

         REQUIRE(ddist_dp_v == Approx(ddist_dp_v_fd).margin(1e-8));
      }
   }
}
