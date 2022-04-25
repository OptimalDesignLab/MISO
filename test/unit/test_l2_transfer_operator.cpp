#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "nlohmann/json.hpp"

#include "irrotational_projector.hpp"
#include "l2_transfer_operator.hpp"

TEST_CASE("ScalarL2IdentityProjection::apply")
{
   int nxy = 2;
   int nz = 1;
   // auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
   //                                        //   mfem::Element::TETRAHEDRON,
   //                                          mfem::Element::HEXAHEDRON,
   //                                          1.0, 1.0, (double)nz / (double)nxy,
   //                                          true);
   auto smesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                            mfem::Element::TRIANGLE);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 2;
   const auto dim = mesh.Dimension();

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "H1"}});

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}});

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::ScalarL2IdentityProjection op(state, mesh_coords, dg_state);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   mfem::Vector dg_state_tv(dg_state.space().GetTrueVSize());

   mfem::FunctionCoefficient state_coeff(
   [](const mfem::Vector &x)
   {
      return x(1) * x(0);
   });
   state.project(state_coeff, state_tv);

   op.apply(state_tv, dg_state_tv);

   dg_state.distributeSharedDofs(dg_state_tv);

   // /// Print fields
   // mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_scalar_identity", &mesh);
   // pv.SetPrefixPath("ParaView");
   // pv.SetLevelsOfDetail(p+2);
   // pv.SetDataFormat(mfem::VTKFormat::ASCII);
   // pv.SetHighOrderOutput(true);
   // pv.RegisterField("state", &state.gridFunc());
   // pv.RegisterField("dg_state", &dg_state.gridFunc());
   // pv.Save();
}

TEST_CASE("ScalarL2IdentityProjection::vectorJacobianProduct wrt state")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   int nxy = 3;
   auto smesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                          //   mfem::Element::TRIANGLE, true,
                                            mfem::Element::QUADRILATERAL, true,
                                            1.0, 1.0);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 4;
   const auto dim = mesh.Dimension();

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "H1"}});
      // {"basis-type", "DG"}});

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}});

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::ScalarL2IdentityProjection op(state, mesh_coords, dg_state);

   mfem::Vector state_tv(state.space().GetTrueVSize());
   mfem::Vector dg_state_tv(dg_state.space().GetTrueVSize());

   mfem::FunctionCoefficient state_coeff(
   [](const mfem::Vector &x)
   {
      return x(1) + x(0);
   });
   state.project(state_coeff, state_tv);

   mfem::FunctionCoefficient adj_coeff(
   [](const mfem::Vector &x)
   {
      return x(1)*x(1) + x(0)*x(0);
   });
   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   dg_state.project(adj_coeff, out_bar);
   // for (int i = 0; i < out_bar.Size(); ++i)
   // {
   //    out_bar(i) = uniform_rand(gen);
   // }

   mfem::Vector state_pert(state.space().GetTrueVSize());
   state_pert = 0.0;
   for (int i = 0; i < state_pert.Size(); ++i)
   {
      state_pert(i) = uniform_rand(gen);
   }

   mfem::Vector state_bar(state.space().GetTrueVSize());
   state_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "state", state_bar);

   auto dout_dstate_v_local = state_pert * state_bar;
   double dout_dstate_v;
   MPI_Allreduce(&dout_dstate_v_local,
                 &dout_dstate_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dstate_v_fd_local = 0.0;

   add(state_tv, delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local += out_bar * dg_state_tv;

   add(state_tv, -2*delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local -= out_bar * dg_state_tv;

   dout_dstate_v_fd_local /= 2*delta;
   double dout_dstate_v_fd;
   MPI_Allreduce(&dout_dstate_v_fd_local,
                 &dout_dstate_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dstate_v: " << dout_dstate_v << "\n";
      std::cout << "dout_dstate_v_fd: " << dout_dstate_v_fd << "\n";
   }

   REQUIRE(dout_dstate_v == Approx(dout_dstate_v_fd).margin(1e-8));
}

TEST_CASE("ScalarL2IdentityProjection::vectorJacobianProduct wrt mesh_coords")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   int nxy = 3;
   auto smesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                            mfem::Element::TRIANGLE, true,
                                          //   mfem::Element::QUADRILATERAL, true,
                                            1.0, 1.0);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 4;
   const auto dim = mesh.Dimension();

   /// create new state vector copying the mesh's fe space
   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto &mesh_fespace = *mesh_gf.ParFESpace();
   mach::FiniteElementState mesh_coords(mesh, mesh_fespace, "mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "H1"}});
      // {"basis-type", "DG"}});
   mfem::Vector state_tv(state.space().GetTrueVSize());
   mfem::FunctionCoefficient state_coeff(
   [](const mfem::Vector &x)
   {
      return x(1)*x(1) + x(0);
   });
   state.project(state_coeff, state_tv);

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p-1},
      {"basis-type", "DG"}});

   mach::ScalarL2IdentityProjection op(state, mesh_coords, dg_state);
   mfem::Vector dg_state_tv(dg_state.space().GetTrueVSize());
   mfem::FunctionCoefficient adj_coeff(
   [](const mfem::Vector &x)
   {
      return x(1)*x(1) + x(0)*x(0);
   });
   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   dg_state.project(adj_coeff, out_bar);
   // for (int i = 0; i < out_bar.Size(); ++i)
   // {
   //    out_bar(i) = uniform_rand(gen);
   // }

   mfem::Vector mesh_pert(mesh_coords.space().GetTrueVSize());
   mesh_pert = 0.0;
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = uniform_rand(gen);
   }

   mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
   mesh_coords_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "mesh_coords", mesh_coords_bar);

   auto dout_dmesh_v_local = mesh_pert * mesh_coords_bar;
   double dout_dmesh_v;
   MPI_Allreduce(&dout_dmesh_v_local,
                 &dout_dmesh_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dmesh_v_fd_local = 0.0;

   add(mesh_coords_tv, delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local += out_bar * dg_state_tv;

   add(mesh_coords_tv, -2*delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local -= out_bar * dg_state_tv;

   dout_dmesh_v_fd_local /= 2*delta;
   double dout_dmesh_v_fd;
   MPI_Allreduce(&dout_dmesh_v_fd_local,
                 &dout_dmesh_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dmesh_v: " << dout_dmesh_v << "\n";
      std::cout << "dout_dmesh_v_fd: " << dout_dmesh_v_fd << "\n";
   }

   REQUIRE(dout_dmesh_v == Approx(dout_dmesh_v_fd).margin(1e-8));
}

TEST_CASE("L2IdentityProjection::apply")
{
   int nxy = 2;
   int nz = 1;
   // auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
   //                                        //   mfem::Element::TETRAHEDRON,
   //                                          mfem::Element::HEXAHEDRON,
   //                                          1.0, 1.0, (double)nz / (double)nxy,
   //                                          true);
   auto smesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                            mfem::Element::TRIANGLE);
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

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::L2IdentityProjection op(state, mesh_coords, dg_state);

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

   // /// Print fields
   // mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_identity", &mesh);
   // pv.SetPrefixPath("ParaView");
   // pv.SetLevelsOfDetail(p+2);
   // pv.SetDataFormat(mfem::VTKFormat::ASCII);
   // pv.SetHighOrderOutput(true);
   // pv.RegisterField("state", &state.gridFunc());
   // pv.RegisterField("dg_state", &dg_state.gridFunc());
   // pv.Save();
}

TEST_CASE("L2IdentityProjection::vectorJacobianProduct wrt state")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   int nxy = 2;
   int nz = 1;
   // auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
   //                                        //   mfem::Element::TETRAHEDRON,
   //                                          mfem::Element::HEXAHEDRON,
   //                                          1.0, 1.0, (double)nz / (double)nxy,
   //                                          true);
   auto smesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                            mfem::Element::TRIANGLE);
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

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::L2IdentityProjection op(state, mesh_coords, dg_state);

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

   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   for (int i = 0; i < out_bar.Size(); ++i)
   {
      out_bar(i) = uniform_rand(gen);
   }
   mfem::Vector state_pert(state.space().GetTrueVSize());
   for (int i = 0; i < state_pert.Size(); ++i)
   {
      state_pert(i) = uniform_rand(gen);
   }

   mfem::Vector state_bar(state.space().GetTrueVSize());
   state_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "state", state_bar);

   auto dout_dstate_v_local = state_pert * state_bar;
   double dout_dstate_v;
   MPI_Allreduce(&dout_dstate_v_local,
                 &dout_dstate_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dstate_v_fd_local = 0.0;

   add(state_tv, delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local += out_bar * dg_state_tv;

   add(state_tv, -2*delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local -= out_bar * dg_state_tv;

   dout_dstate_v_fd_local /= 2*delta;
   double dout_dstate_v_fd;
   MPI_Allreduce(&dout_dstate_v_fd_local,
                 &dout_dstate_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dstate_v: " << dout_dstate_v << "\n";
      std::cout << "dout_dstate_v_fd: " << dout_dstate_v_fd << "\n";
   }

   REQUIRE(dout_dstate_v == Approx(dout_dstate_v_fd).margin(1e-8));
}

TEST_CASE("L2IdentityProjection::vectorJacobianProduct wrt mesh_coords")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   int nxy = 3;
   auto smesh = mfem::Mesh::MakeCartesian2D(nxy, nxy,
                                            mfem::Element::TRIANGLE, true,
                                          //   mfem::Element::QUADRILATERAL, true,
                                            1.0, 1.0);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 4;
   const auto dim = mesh.Dimension();

   /// create new state vector copying the mesh's fe space
   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto &mesh_fespace = *mesh_gf.ParFESpace();
   mach::FiniteElementState mesh_coords(mesh, mesh_fespace, "mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "ND"}});

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}},
      dim);

   mach::L2IdentityProjection op(state, mesh_coords, dg_state);

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

   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   for (int i = 0; i < out_bar.Size(); ++i)
   {
      out_bar(i) = uniform_rand(gen);
   }
   mfem::Vector mesh_pert(mesh_coords.space().GetTrueVSize());
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = uniform_rand(gen);
   }

   mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
   mesh_coords_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "mesh_coords", mesh_coords_bar);

   auto dout_dmesh_v_local = mesh_pert * mesh_coords_bar;
   double dout_dmesh_v;
   MPI_Allreduce(&dout_dmesh_v_local,
                 &dout_dmesh_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dmesh_v_fd_local = 0.0;

   add(mesh_coords_tv, delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local += out_bar * dg_state_tv;

   add(mesh_coords_tv, -2*delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local -= out_bar * dg_state_tv;

   dout_dmesh_v_fd_local /= 2*delta;
   double dout_dmesh_v_fd;
   MPI_Allreduce(&dout_dmesh_v_fd_local,
                 &dout_dmesh_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dmesh_v: " << dout_dmesh_v << "\n";
      std::cout << "dout_dmesh_v_fd: " << dout_dmesh_v_fd << "\n";
   }

   REQUIRE(dout_dmesh_v == Approx(dout_dmesh_v_fd).margin(1e-8));
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

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::L2CurlProjection op(state, mesh_coords, dg_curl);

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

   // /// Print fields
   // mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_curl", &mesh);
   // pv.SetPrefixPath("ParaView");
   // pv.SetLevelsOfDetail(p+2);
   // pv.SetDataFormat(mfem::VTKFormat::ASCII);
   // pv.SetHighOrderOutput(true);
   // pv.RegisterField("curl", &curl.gridFunc());
   // pv.RegisterField("dg_curl", &dg_curl.gridFunc());
   // pv.Save();
}

TEST_CASE("L2CurlProjection::vectorJacobianProduct wrt state")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

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

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::L2CurlProjection op(state, mesh_coords, dg_state);

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
   mfem::Vector state_tv_copy(state_tv);

   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   for (int i = 0; i < out_bar.Size(); ++i)
   {
      out_bar(i) = uniform_rand(gen);
   }
   mfem::Vector state_pert(state.space().GetTrueVSize());
   for (int i = 0; i < state_pert.Size(); ++i)
   {
      state_pert(i) = uniform_rand(gen);
   }

   mfem::Vector state_bar(state.space().GetTrueVSize());
   state_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "state", state_bar);

   auto dout_dstate_v_local = state_pert * state_bar;
   double dout_dstate_v;
   MPI_Allreduce(&dout_dstate_v_local,
                 &dout_dstate_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dstate_v_fd_local = 0.0;

   add(state_tv, delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local += out_bar * dg_state_tv;

   add(state_tv, -2*delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local -= out_bar * dg_state_tv;

   dout_dstate_v_fd_local /= 2*delta;
   double dout_dstate_v_fd;
   MPI_Allreduce(&dout_dstate_v_fd_local,
                 &dout_dstate_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dstate_v: " << dout_dstate_v << "\n";
      std::cout << "dout_dstate_v_fd: " << dout_dstate_v_fd << "\n";
   }

   REQUIRE(dout_dstate_v == Approx(dout_dstate_v_fd).margin(1e-8));
}

TEST_CASE("L2CurlProjection::vectorJacobianProduct wrt mesh_coords")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

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

   /// create new state vector copying the mesh's fe space
   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto &mesh_fespace = *mesh_gf.ParFESpace();
   mach::FiniteElementState mesh_coords(mesh, mesh_fespace, "mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "ND"}});

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}},
      dim);

   mach::L2CurlProjection op(state, mesh_coords, dg_state);

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
   mfem::Vector state_tv_copy(state_tv);

   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   for (int i = 0; i < out_bar.Size(); ++i)
   {
      out_bar(i) = uniform_rand(gen);
   }
   mfem::Vector mesh_pert(mesh_coords.space().GetTrueVSize());
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = uniform_rand(gen);
   }

   mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
   mesh_coords_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "mesh_coords", mesh_coords_bar);

   auto dout_dmesh_v_local = mesh_pert * mesh_coords_bar;
   double dout_dmesh_v;
   MPI_Allreduce(&dout_dmesh_v_local,
                 &dout_dmesh_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dmesh_v_fd_local = 0.0;

   add(mesh_coords_tv, delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local += out_bar * dg_state_tv;

   add(mesh_coords_tv, -2*delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local -= out_bar * dg_state_tv;

   dout_dmesh_v_fd_local /= 2*delta;
   double dout_dmesh_v_fd;
   MPI_Allreduce(&dout_dmesh_v_fd_local,
                 &dout_dmesh_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dmesh_v: " << dout_dmesh_v << "\n";
      std::cout << "dout_dmesh_v_fd: " << dout_dmesh_v_fd << "\n";
   }

   REQUIRE(dout_dmesh_v == Approx(dout_dmesh_v_fd).margin(1e-8));
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

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::L2CurlMagnitudeProjection op(state, mesh_coords, dg_curl);

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

   /// Compute curl conventionally
   mach::DiscreteCurlOperator curl_op(&state.space(), &curl.space());
   curl_op.Assemble();
   curl_op.Finalize();
   curl_op.Mult(state.gridFunc(), curl.gridFunc());

   // /// Print fields
   // mfem::ParaViewDataCollection pv("test_mixed_nonlinear_operator_curl_magnitude", &mesh);
   // pv.SetPrefixPath("ParaView");
   // pv.SetLevelsOfDetail(p+2);
   // pv.SetDataFormat(mfem::VTKFormat::ASCII);
   // pv.SetHighOrderOutput(true);
   // pv.RegisterField("curl", &curl.gridFunc());
   // pv.RegisterField("dg_curl_mag", &dg_curl.gridFunc());
   // pv.Save();
}

TEST_CASE("L2CurlMagnitudeProjection::vectorJacobianProduct wrt state")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   int nxy = 4;
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
      {"basis-type", "DG"}});

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();

   mach::FiniteElementState mesh_coords(mesh,  *mesh_fespace);
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mach::L2CurlMagnitudeProjection op(state, mesh_coords, dg_state);

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
   mfem::Vector state_tv_copy(state_tv);

   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   for (int i = 0; i < out_bar.Size(); ++i)
   {
      out_bar(i) = uniform_rand(gen);
   }
   mfem::Vector state_pert(state.space().GetTrueVSize());
   for (int i = 0; i < state_pert.Size(); ++i)
   {
      state_pert(i) = uniform_rand(gen);
   }

   mfem::Vector state_bar(state.space().GetTrueVSize());
   state_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "state", state_bar);

   auto dout_dstate_v_local = state_pert * state_bar;
   double dout_dstate_v;
   MPI_Allreduce(&dout_dstate_v_local,
                 &dout_dstate_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dstate_v_fd_local = 0.0;

   add(state_tv, delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local += out_bar * dg_state_tv;

   add(state_tv, -2*delta, state_pert, state_tv);
   op.apply(state_tv, dg_state_tv);
   dout_dstate_v_fd_local -= out_bar * dg_state_tv;

   dout_dstate_v_fd_local /= 2*delta;
   double dout_dstate_v_fd;
   MPI_Allreduce(&dout_dstate_v_fd_local,
                 &dout_dstate_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dstate_v: " << dout_dstate_v << "\n";
      std::cout << "dout_dstate_v_fd: " << dout_dstate_v_fd << "\n";
   }

   REQUIRE(dout_dstate_v == Approx(dout_dstate_v_fd).margin(1e-8));
}

TEST_CASE("L2CurlMagnitudeProjection::vectorJacobianProduct wrt mesh_coords")
{
   std::default_random_engine gen;
   std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   int nxy = 3;
   int nz = 1;
   auto smesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                          //   mfem::Element::TETRAHEDRON,
                                            mfem::Element::HEXAHEDRON,
                                            1.0, 1.0, (double)nz / (double)nxy,
                                            true);
   auto mesh = mfem::ParMesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto p = 4;
   const auto dim = mesh.Dimension();

   /// create new state vector copying the mesh's fe space
   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto &mesh_fespace = *mesh_gf.ParFESpace();
   mach::FiniteElementState mesh_coords(mesh, mesh_fespace, "mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   mfem::Vector mesh_coords_tv(mesh_coords.space().GetTrueVSize());
   mesh_coords.setTrueVec(mesh_coords_tv);

   mach::FiniteElementState state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "ND"}});

   mach::FiniteElementState dg_state(mesh, nlohmann::json{
      {"degree", p},
      {"basis-type", "DG"}});

   mach::L2CurlMagnitudeProjection op(state, mesh_coords, dg_state);

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
   mfem::Vector state_tv_copy(state_tv);

   mfem::Vector out_bar(dg_state.space().GetTrueVSize());
   for (int i = 0; i < out_bar.Size(); ++i)
   {
      out_bar(i) = uniform_rand(gen);
   }
   mfem::Vector mesh_pert(mesh_coords.space().GetTrueVSize());
   for (int i = 0; i < mesh_pert.Size(); ++i)
   {
      mesh_pert(i) = uniform_rand(gen);
   }

   mfem::Vector mesh_coords_bar(mesh_coords.space().GetTrueVSize());
   mesh_coords_bar = 0.0;
   setInputs(op, {{"state", state_tv}});
   op.vectorJacobianProduct(out_bar, "mesh_coords", mesh_coords_bar);

   auto dout_dmesh_v_local = mesh_pert * mesh_coords_bar;
   double dout_dmesh_v;
   MPI_Allreduce(&dout_dmesh_v_local,
                 &dout_dmesh_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dout_dmesh_v_fd_local = 0.0;

   add(mesh_coords_tv, delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local += out_bar * dg_state_tv;

   add(mesh_coords_tv, -2*delta, mesh_pert, mesh_coords_tv);
   mesh_coords.distributeSharedDofs(mesh_coords_tv); // update mesh nodes
   op.apply(state_tv, dg_state_tv);
   dout_dmesh_v_fd_local -= out_bar * dg_state_tv;

   dout_dmesh_v_fd_local /= 2*delta;
   double dout_dmesh_v_fd;
   MPI_Allreduce(&dout_dmesh_v_fd_local,
                 &dout_dmesh_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 state.space().GetComm());

   auto rank = state.space().GetMyRank();
   if (rank == 0)
   {
      std::cout << "dout_dmesh_v: " << dout_dmesh_v << "\n";
      std::cout << "dout_dmesh_v_fd: " << dout_dmesh_v_fd << "\n";
   }

   REQUIRE(dout_dmesh_v == Approx(dout_dmesh_v_fd).margin(1e-8));
}
