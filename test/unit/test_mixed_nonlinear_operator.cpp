#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "irrotational_projector.hpp"
#include "mixed_nonlinear_operator.hpp"

void identity_operator(
   const mfem::FiniteElement &domain_fe,
   const mfem::FiniteElement &range_fe,
   mfem::ElementTransformation &trans,
   const mfem::Vector &el_domain,
   mfem::Vector &el_range);

void curl_operator(
   const mfem::FiniteElement &domain_fe,
   const mfem::FiniteElement &range_fe,
   mfem::ElementTransformation &trans,
   const mfem::Vector &el_domain,
   mfem::Vector &el_range);

void curl_magnitude_operator(
   const mfem::FiniteElement &domain_fe,
   const mfem::FiniteElement &range_fe,
   mfem::ElementTransformation &trans,
   const mfem::Vector &el_domain,
   mfem::Vector &el_range);

TEST_CASE("L2TransferOperator::apply (identity)")
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

   mach::L2TransferOperator op(state, dg_state, identity_operator);

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

TEST_CASE("L2TransferOperator::apply (curl)")
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

   mach::L2TransferOperator op(state, dg_curl, curl_operator);

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

TEST_CASE("L2TransferOperator::apply (curl magnitude)")
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

   mach::L2TransferOperator op(state, dg_curl, curl_magnitude_operator);

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

void identity_operator(
   const mfem::FiniteElement &domain_fe,
   const mfem::FiniteElement &range_fe,
   mfem::ElementTransformation &trans,
   const mfem::Vector &el_domain,
   mfem::Vector &el_range)
{
   int domain_dof = domain_fe.GetDof();
   int range_dof = range_fe.GetDof();

   int space_dim = trans.GetSpaceDim();

   mfem::DenseMatrix vshape(domain_dof, space_dim);

   double shape_vec_buffer[3];
   mfem::Vector shape_vec(shape_vec_buffer, space_dim);

   mfem::DenseMatrix range(el_range.GetData(), range_dof, space_dim);

   const auto &ir = range_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      domain_fe.CalcVShape(trans, vshape);
      vshape.MultTranspose(el_domain, shape_vec);

      for (int j = 0; j < space_dim; j++)
      {
         range(i, j) = shape_vec(j);
      }
   }
}

void curl_operator(
   const mfem::FiniteElement &domain_fe,
   const mfem::FiniteElement &range_fe,
   mfem::ElementTransformation &trans,
   const mfem::Vector &el_domain,
   mfem::Vector &el_range)
{
   int domain_dof = domain_fe.GetDof();
   int range_dof = range_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(domain_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(domain_dof, curl_dim);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

   mfem::DenseMatrix range(el_range.GetData(), range_dof, curl_dim);

   const auto &ir = range_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      domain_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_domain, curl_vec);

      curl_vec /= trans.Weight();

      for (int j = 0; j < curl_dim; j++)
      {
         range(i, j) = curl_vec(j);
      }
   }
}

void curl_magnitude_operator(
   const mfem::FiniteElement &domain_fe,
   const mfem::FiniteElement &range_fe,
   mfem::ElementTransformation &trans,
   const mfem::Vector &el_domain,
   mfem::Vector &el_range)
{
   int domain_dof = domain_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(domain_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(domain_dof, curl_dim);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

   const auto &ir = range_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      domain_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_domain, curl_vec);

      const double curl_vec_norm = curl_vec.Norml2();
      const double curl_mag = curl_vec_norm / trans.Weight();
      if (curl_mag <= 0.0)
      {
         std::cout << "negative at ip: " << i << "\n";
      }
      std::cout << curl_mag << "\n";

      el_range(i) = curl_mag;
   }
}
