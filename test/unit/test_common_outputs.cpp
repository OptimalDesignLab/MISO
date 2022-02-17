#include <map>
#include <string>

#include "catch.hpp"
#include "mfem.hpp"

#include "common_outputs.hpp"

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
   REQUIRE(max_state == Approx(1.3026749725));

   output_opts["rho"] = 1.0;
   setOptions(out, output_opts);

   max_state = calcOutput(out, inputs);
   /// Should be sqrt(sin(1.0)^2 + 1.0)
   REQUIRE(max_state == Approx(1.0152746915));

}
