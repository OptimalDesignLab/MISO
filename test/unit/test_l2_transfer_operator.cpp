#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "irrotational_projector.hpp"
#include "l2_transfer_operator.hpp"

TEST_CASE("L2IdentityProjection::apply")
{
   int nxy = 2;
   int nz = 1;
   auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                          //   mfem::Element::TETRAHEDRON,
                                            mfem::Element::HEXAHEDRON,
                                            1.0, 1.0, (double)nz / (double)nxy,
                                            true);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 2;
   const auto dim = mesh.Dimension();

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "ND"}});

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}},
      dim);

   mach::L2IdentityProjection op(state, dg_state);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   mfem::Vector dg_state_tv(dg_state.space().GetTrueVSize());

   mfem::VectorFunctionCoefficient state_coeff(3,
   [](const mfem::Vector &x, mfem::Vector &A)
   {
      A(0) = x(1);
      A(1) = -x(0);
      A(2) = 0.0;
   });
   state.project(state_coeff, state_tv);

   op.apply(state_tv, dg_state_tv);

   dg_state.distributeSharedDofs(dg_state_tv);

   /// Print fields
   mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_identity", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(p+2);
   pv.SetDataFormat(mfem::VTKFormat::ASCII);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("state", &state.gridFunc());
   pv.RegisterField("dg_state", &dg_state.gridFunc());
   pv.Save();
}

TEST_CASE("L2CurlProjection::apply")
{
   int nxy = 2;
   int nz = 1;
   auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                          //   mfem::Element::TETRAHEDRON,
                                            mfem::Element::HEXAHEDRON,
                                            1.0, 1.0, (double)nz / (double)nxy,
                                            true);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 2;
   const auto dim = mesh.Dimension();

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "ND"}});

   mach::FiniteElementState curl(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "RT"}});

   mach::FiniteElementState dg_curl(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}},
      dim);

   mach::L2CurlProjection op(state, dg_curl);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   mfem::Vector dg_curl_tv(dg_curl.space().GetTrueVSize());

   mfem::VectorFunctionCoefficient state_coeff(3,
   [](const mfem::Vector &x, mfem::Vector &A)
   {
      A(0) = x(1);
      A(1) = -x(0);
      A(2) = 0.0;
   });
   state.project(state_coeff, state_tv);

   op.apply(state_tv, dg_curl_tv);
   dg_curl.distributeSharedDofs(dg_curl_tv);

   /// Compute curl conventionally
   mach::DiscreteCurlOperator curl_op(&state.space(), &curl.space());
   curl_op.Assemble();
   curl_op.Finalize();
   curl_op.Mult(state.gridFunc(), curl.gridFunc());

   /// Print fields
   mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_curl", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(p+2);
   pv.SetDataFormat(mfem::VTKFormat::ASCII);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("curl", &curl.gridFunc());
   pv.RegisterField("dg_curl", &dg_curl.gridFunc());
   pv.Save();
}

TEST_CASE("L2CurlMagnitudeProjection::apply")
{
   int nxy = 2;
   int nz = 1;
   auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                          //   mfem::Element::TETRAHEDRON,
                                            mfem::Element::HEXAHEDRON,
                                            1.0, 1.0, (double)nz / (double)nxy,
                                            true);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 2;
   const auto dim = mesh.Dimension();

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "ND"}});

   mach::FiniteElementState curl(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "RT"}});

   mach::FiniteElementState dg_curl(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}});

   mach::L2CurlMagnitudeProjection op(state, dg_curl);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   mfem::Vector dg_curl_tv(dg_curl.space().GetTrueVSize());

   mfem::VectorFunctionCoefficient state_coeff(3,
   [](const mfem::Vector &x, mfem::Vector &A)
   {
      A(0) = x(1)*x(1)*x(1);
      A(1) = -x(0)*x(0)*x(0);
      A(2) = 0.0;
   });
   state.project(state_coeff, state_tv);

   op.apply(state_tv, dg_curl_tv);
   dg_curl.distributeSharedDofs(dg_curl_tv);

   std::cout << "\nDG TV:\n";
   dg_curl_tv.Print(std::cout, 1);

   /// Compute curl conventionally
   mach::DiscreteCurlOperator curl_op(&state.space(), &curl.space());
   curl_op.Assemble();
   curl_op.Finalize();
   curl_op.Mult(state.gridFunc(), curl.gridFunc());

   /// Print fields
   mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_curl_magnitude", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(p+2);
   pv.SetDataFormat(mfem::VTKFormat::ASCII);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("curl", &curl.gridFunc());
   pv.RegisterField("dg_curl_mag", &dg_curl.gridFunc());
   pv.Save();
}