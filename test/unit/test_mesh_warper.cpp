#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "mesh_warper.hpp"

auto warp_options = R"(
{
   "space-dis": {
      "basis-type": "h1",
      "degree": 1
   },
   "lin-solver": {
      "type": "pcg",
      "printlevel": -1,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "nonlin-solver": {
      "type": "newton",
      "printlevel": -1,
      "maxiter": 1,
      "reltol": 1e-10,
      "abstol": 1e-9
   },
   "adj-solver": {
      "type": "pcg",
      "printlevel": 1,
      "maxiter": 100,
      "abstol": 1e-14,
      "reltol": 1e-14
   },
   "lin-prec": {
      "printlevel": -1
   },
   "bcs": {
      "essential": "all"
   }
})"_json;

TEST_CASE("MeshWarper::solveForState")
{
   auto comm = MPI_COMM_WORLD;

   int nxyz = 2;
   auto smesh = std::make_unique<mfem::Mesh>(
      mfem::Mesh::MakeCartesian3D(nxyz, nxyz, nxyz,
                                  mfem::Element::TETRAHEDRON));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector init_vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(init_vol_coords);
   mfem::Vector vol_coords = init_vol_coords;

   for (int i = 0; i < surf_mesh_size; i += 3)
   {
      surf_coords(i + 0) += 1.0; 
      surf_coords(i + 1) += 1.0; 
      surf_coords(i + 2) += 0.0; 
   }

   mfem::Array<int> surf_indices;
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }

   warper.solveForState(vol_coords);
   // vol_coords.Print(mfem::out, 3);

   for (int i = 0; i < vol_coords.Size(); i += 3)
   {
      REQUIRE(init_vol_coords[i + 0] + 1.0 == Approx(vol_coords[i + 0]));
      REQUIRE(init_vol_coords[i + 1] + 1.0 == Approx(vol_coords[i + 1]));
      REQUIRE(init_vol_coords[i + 2] + 0.0 == Approx(vol_coords[i + 2]));
   }

}

TEST_CASE("MeshWarper::vectorJacobianProduct wrt state")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   auto comm = MPI_COMM_WORLD;

   int nx = 3;
   auto smesh = std::make_unique<mfem::Mesh>(
      // mfem::Mesh::MakeCartesian1D(nx));
      mfem::Mesh::MakeCartesian2D(nx, nx, mfem::Element::QUADRILATERAL));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(vol_coords);

   mfem::Array<int> surf_indices;   
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }

   warper.solveForState(vol_coords);

   mach::MachInputs inputs{
      {"surf_mesh_coords", surf_coords},
      {"state", vol_coords}
   };
   warper.linearize(inputs);

   // mfem::DenseMatrix jac_rev(vol_mesh_size);
   // for (int i = 0; i < vol_mesh_size; ++i)
   // {
   //    mfem::Vector res_bar(vol_mesh_size);
   //    res_bar = 0.0;
   //    res_bar(i) = 1.0;

   //    mfem::Vector state_bar(vol_mesh_size);
   //    state_bar = 0.0;
   //    warper.vectorJacobianProduct(res_bar, "state", state_bar);

   //    for (int j = 0; j < vol_mesh_size; ++j)
   //    {
   //       jac_rev(i, j) = state_bar(j);
   //    }
   // }
   // std::cout << "Jac REV:\n";
   // jac_rev.Print(mfem::out, vol_mesh_size);

   // mfem::DenseMatrix jac_fwd(vol_mesh_size);
   // for (int i = 0; i < vol_mesh_size; ++i)
   // {
   //    mfem::Vector state_dot(vol_mesh_size);
   //    state_dot = 0.0;
   //    state_dot(i) = 1.0;

   //    mfem::Vector res_dot(vol_mesh_size);
   //    res_dot = 0.0;
   //    warper.jacobianVectorProduct(state_dot, "state", res_dot);

   //    for (int j = 0; j < vol_mesh_size; ++j)
   //    {
   //       jac_fwd(j, i) = res_dot(j);
   //    }
   // }
   // std::cout << "Jac FWD:\n";
   // jac_fwd.Print(mfem::out, vol_mesh_size);

   // mfem::DenseMatrix jac_fd(vol_mesh_size);
   // for (int i = 0; i < vol_mesh_size; ++i)
   // {
   //    mfem::Vector res(vol_mesh_size);
   //    warper.calcResidual(inputs, res);

   //    double delta = 1e-6;
   //    vol_coords(i) += delta;

   //    mfem::Vector res_plus(vol_mesh_size);
   //    warper.calcResidual(inputs, res_plus);
   //    vol_coords(i) -= delta; // reset

   //    for (int j = 0; j < vol_mesh_size; ++j)
   //    {
   //       jac_fd(j, i) = (res_plus(j) - res(j)) / delta;
   //    }
   // }
   // std::cout << "Jac FD:\n";
   // jac_fd.Print(mfem::out, vol_mesh_size);

   mfem::Vector res_bar(vol_mesh_size);
   for (int i = 0; i < res_bar.Size(); ++i)
   {
      res_bar(i) = uniform_rand(gen);
   }

   mfem::Vector state_bar(vol_mesh_size);
   state_bar = 0.0;
   warper.vectorJacobianProduct(res_bar, "state", state_bar);

   // initialize the vector that we use to perturb the state
   mfem::Vector v_tv(vol_mesh_size);
   for (int i = 0; i < v_tv.Size(); ++i)
   {
      v_tv(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = state_bar * v_tv;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector res_vec(vol_mesh_size);

   add(vol_coords, delta, v_tv, vol_coords);

   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local += res_bar * res_vec;

   add(vol_coords, -2*delta, v_tv, vol_coords);
   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local -= res_bar * res_vec;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

TEST_CASE("MeshWarper::jacobianVectorProduct wrt state")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   auto comm = MPI_COMM_WORLD;

   int nx = 3;
   auto smesh = std::make_unique<mfem::Mesh>(
      // mfem::Mesh::MakeCartesian1D(nx));
      mfem::Mesh::MakeCartesian2D(nx, nx, mfem::Element::QUADRILATERAL));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(vol_coords);

   mfem::Array<int> surf_indices;   
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }

   warper.solveForState(vol_coords);

   mach::MachInputs inputs{
      {"surf_mesh_coords", surf_coords},
      {"state", vol_coords}
   };
   warper.linearize(inputs);

   mfem::Vector state_dot(vol_mesh_size);
   for (int i = 0; i < state_dot.Size(); ++i)
   {
      state_dot(i) = uniform_rand(gen);
   }

   mfem::Vector res_dot(vol_mesh_size);
   res_dot = 0.0;
   warper.jacobianVectorProduct(state_dot, "state", res_dot);

   // initialize the vector that we use to perturb the state
   mfem::Vector v_tv(vol_mesh_size);
   for (int i = 0; i < v_tv.Size(); ++i)
   {
      v_tv(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = res_dot * v_tv;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector res_vec(vol_mesh_size);

   add(vol_coords, delta, state_dot, vol_coords);

   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local += v_tv * res_vec;

   add(vol_coords, -2*delta, state_dot, vol_coords);
   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local -= v_tv * res_vec;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

TEST_CASE("MeshWarper::vectorJacobianProduct wrt surf_mesh_coords")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   auto comm = MPI_COMM_WORLD;

   int nx = 3;
   auto smesh = std::make_unique<mfem::Mesh>(
      // mfem::Mesh::MakeCartesian1D(nx));
      mfem::Mesh::MakeCartesian2D(nx, nx, mfem::Element::QUADRILATERAL));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(vol_coords);

   mfem::Array<int> surf_indices;   
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }

   warper.solveForState(vol_coords);

   mach::MachInputs inputs{
      {"surf_mesh_coords", surf_coords},
      {"state", vol_coords}
   };
   warper.linearize(inputs);

   mfem::Vector res_bar(vol_mesh_size);
   for (int i = 0; i < res_bar.Size(); ++i)
   {
      res_bar(i) = uniform_rand(gen);
   }

   mfem::Vector wrt_bar(surf_mesh_size);
   wrt_bar = 0.0;
   warper.vectorJacobianProduct(res_bar, "surf_mesh_coords", wrt_bar);

   // initialize the vector that we use to perturb the state
   mfem::Vector v_tv(surf_mesh_size);
   for (int i = 0; i < v_tv.Size(); ++i)
   {
      v_tv(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = wrt_bar * v_tv;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector res_vec(vol_mesh_size);

   add(surf_coords, delta, v_tv, surf_coords);

   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local += res_bar * res_vec;

   add(surf_coords, -2*delta, v_tv, surf_coords);
   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local -= res_bar * res_vec;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

TEST_CASE("MeshWarper::jacobianVectorProduct wrt surf_mesh_coords")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   auto comm = MPI_COMM_WORLD;

   int nx = 3;
   auto smesh = std::make_unique<mfem::Mesh>(
      // mfem::Mesh::MakeCartesian1D(nx));
      mfem::Mesh::MakeCartesian2D(nx, nx, mfem::Element::QUADRILATERAL));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(vol_coords);

   mfem::Array<int> surf_indices;   
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }

   warper.solveForState(vol_coords);

   mach::MachInputs inputs{
      {"surf_mesh_coords", surf_coords},
      {"state", vol_coords}
   };
   warper.linearize(inputs);

   mfem::Vector wrt_dot(surf_mesh_size);
   for (int i = 0; i < wrt_dot.Size(); ++i)
   {
      wrt_dot(i) = uniform_rand(gen);
   }

   mfem::Vector res_dot(vol_mesh_size);
   res_dot = 0.0;
   warper.jacobianVectorProduct(wrt_dot, "surf_mesh_coords", res_dot);

   // initialize the vector that we use to perturb the surf coords
   mfem::Vector v_tv(vol_mesh_size);
   for (int i = 0; i < v_tv.Size(); ++i)
   {
      v_tv(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = res_dot * v_tv;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   // now compute the finite-difference approximation...
   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector res_vec(vol_mesh_size);

   add(surf_coords, delta, wrt_dot, surf_coords);

   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local += v_tv * res_vec;

   add(surf_coords, -2*delta, wrt_dot, surf_coords);
   res_vec = 0.0;
   warper.calcResidual(inputs, res_vec);
   dJdx_v_fd_local -= v_tv * res_vec;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

TEST_CASE("MeshWarper total derivative of vol_coords wrt surf_coords")
{
   static std::default_random_engine gen;
   static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

   auto comm = MPI_COMM_WORLD;

   int nx = 4;
   auto smesh = std::make_unique<mfem::Mesh>(
      // mfem::Mesh::MakeCartesian1D(nx));
      mfem::Mesh::MakeCartesian2D(nx, nx, mfem::Element::QUADRILATERAL));

   mach::MeshWarper warper(comm, warp_options, std::move(smesh));

   auto surf_mesh_size = warper.getSurfaceCoordsSize();
   mfem::Vector surf_coords(surf_mesh_size);
   warper.getInitialSurfaceCoords(surf_coords);

   auto vol_mesh_size = warper.getVolumeCoordsSize();
   mfem::Vector vol_coords(vol_mesh_size);
   warper.getInitialVolumeCoords(vol_coords);

   mfem::Array<int> surf_indices;   
   warper.getSurfCoordIndices(surf_indices);

   for (int i = 0; i < surf_mesh_size; ++i)
   {
      vol_coords(surf_indices[i]) = surf_coords(i);
   }
   warper.solveForState(vol_coords);

   mach::MachInputs inputs{
      {"surf_mesh_coords", surf_coords},
      {"state", vol_coords}
   };
   warper.linearize(inputs);

   // mfem::DenseMatrix jac_rev(vol_mesh_size, surf_mesh_size);
   // for (int i = 0; i < vol_mesh_size; ++i)
   // {
   //    mfem::Vector state_bar(vol_mesh_size);
   //    state_bar = 0.0;
   //    state_bar(i) = -1.0;

   //    mfem::Vector adjoint(vol_mesh_size);
   //    adjoint = 0.0;
   //    warper.solveForAdjoint(vol_coords, state_bar, adjoint);

   //    std::cout << "adjoint:\n";
   //    adjoint.Print(mfem::out, vol_mesh_size);
   //    std::cout << "\n";

   //    mfem::Vector surf_mesh_bar(surf_mesh_size);
   //    surf_mesh_bar = 0.0;
   //    warper.vectorJacobianProduct(adjoint, "surf_mesh_coords", surf_mesh_bar);

   //    for (int j = 0; j < surf_mesh_size; ++j)
   //    {
   //       jac_rev(i, j) = surf_mesh_bar(j);
   //    }
   // }
   // std::cout << "Jac REV:\n";
   // jac_rev.Print(mfem::out, vol_mesh_size);

   // mfem::DenseMatrix jac_fd(vol_mesh_size, surf_mesh_size);
   // for (int i = 0; i < surf_mesh_size; ++i)
   // {
   //    mfem::Vector vol_coords_plus(vol_mesh_size);
   //    vol_coords_plus = vol_coords;

   //    double delta = 1e-6;
   //    surf_coords(i) += delta;

   //    for (int i = 0; i < surf_mesh_size; ++i)
   //    {
   //       vol_coords_plus(surf_indices[i]) = surf_coords(i);
   //    }
   //    warper.solveForState(vol_coords_plus);
   //    surf_coords(i) -= delta; // reset

   //    for (int j = 0; j < vol_mesh_size; ++j)
   //    {
   //       jac_fd(j, i) = (vol_coords_plus(j) - vol_coords(j)) / delta;
   //    }
   // }
   // std::cout << "Jac FD:\n";
   // jac_fd.Print(mfem::out, vol_mesh_size);

   mfem::Vector state_bar(vol_mesh_size);
   for (int i = 0; i < state_bar.Size(); ++i)
   {
      state_bar(i) = uniform_rand(gen);
   }

   mfem::Vector adjoint(vol_mesh_size);
   adjoint = 0.0;
   warper.solveForAdjoint(vol_coords, state_bar, adjoint);
   adjoint *= -1.0;

   mfem::Vector surf_mesh_bar(surf_mesh_size);
   surf_mesh_bar = 0.0;
   warper.vectorJacobianProduct(adjoint, "surf_mesh_coords", surf_mesh_bar);

   // initialize the vector that we use to perturb the state
   mfem::Vector v(surf_mesh_bar.Size());
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = uniform_rand(gen);
   }

   auto dJdx_v_local = surf_mesh_bar * v;
   double dJdx_v;
   MPI_Allreduce(&dJdx_v_local,
                 &dJdx_v,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   // now compute the finite-difference approximation...
   mfem::Vector vol_coords_plus(vol_coords);
   mfem::Vector vol_coords_minus(vol_coords);

   auto delta = 1e-5;
   double dJdx_v_fd_local = 0.0;
   mfem::Vector res_vec(vol_mesh_size);

   add(surf_coords, delta, v, surf_coords);
   for (int i = 0; i < surf_coords.Size(); ++i)
   {
      vol_coords_plus(surf_indices[i]) = surf_coords(i);
   }
   warper.solveForState(vol_coords_plus);
   dJdx_v_fd_local += state_bar * vol_coords_plus;

   add(surf_coords, -2*delta, v, surf_coords);
   for (int i = 0; i < surf_coords.Size(); ++i)
   {
      vol_coords_minus(surf_indices[i]) = surf_coords(i);
   }
   warper.solveForState(vol_coords_minus);
   dJdx_v_fd_local -= state_bar * vol_coords_minus;

   dJdx_v_fd_local /= 2*delta;

   double dJdx_v_fd;
   MPI_Allreduce(&dJdx_v_fd_local,
                 &dJdx_v_fd,
                 1,
                 MPI_DOUBLE,
                 MPI_SUM,
                 comm);

   int rank;
   MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      std::cout << "dJdx_v: " << dJdx_v << "\n";
      std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";
   }

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}