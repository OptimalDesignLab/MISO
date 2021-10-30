#include <iostream>
#include <random>

#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "mach_load.hpp"
#include "magnetic_load.hpp"

using namespace mach;
using namespace mfem;

void northMagnetizationSource(const Vector &x,
                              Vector &M);

void southMagnetizationSource(const Vector &x,
                              Vector &M);

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
Mesh buildMesh(int nxy, int nz);

static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

TEST_CASE("LegacyMagneticLoad Value Test")
{

   Mesh smesh = buildMesh(10, 10);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;
   const auto dim = mesh.Dimension();

   // get the finite-element space for the state
   ND_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   // create mag_coeff coefficient
   VectorMeshDependentCoefficient mag_coeff(dim);
   {
      std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
            new VectorFunctionCoefficient(dim,
                                          northMagnetizationSource));
      mag_coeff.addCoefficient(1, move(temp_coeff));
   }
   {
      std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
            new VectorFunctionCoefficient(dim,
                                          southMagnetizationSource));
      mag_coeff.addCoefficient(2, move(temp_coeff));
   }

   // create nu coeff
   ConstantCoefficient nu(1.0);///(M_PI*4e-7));

   LegacyMagneticLoad load(fes, mag_coeff, nu);
   MachLoad ml(load);

   HypreParVector tv(&fes);

   MachInputs inputs;
   setInputs(ml, inputs);
   tv = 0.0;
   addLoad(ml, tv);

   auto norm = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   // std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(1.7467088168).margin(1e-10));

   setInputs(ml, inputs);
   tv = 0.0;
   addLoad(ml, tv);

   norm = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   // std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(1.7467088168).margin(1e-10));
}

TEST_CASE("MagneticLoad Value Test")
{
   Mesh smesh = buildMesh(3, 3);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;
   const auto dim = mesh.Dimension();

   // get the finite-element space for the state
   ND_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   // create mag_coeff coefficient
   VectorMeshDependentCoefficient mag_coeff(dim);
   {
      std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
            new VectorFunctionCoefficient(dim,
                                          northMagnetizationSource));
      mag_coeff.addCoefficient(1, move(temp_coeff));
   }
   {
      std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
            new VectorFunctionCoefficient(dim,
                                          southMagnetizationSource));
      mag_coeff.addCoefficient(2, move(temp_coeff));
   }

   // create nu coeff
   ConstantCoefficient nu(1.0);///(M_PI*4e-7));

   MagneticLoad loadLF(fes, mag_coeff, nu);
   MachLoad mlLF(loadLF);

   HypreParVector tv(&fes);

   MachInputs inputs;

   setInputs(mlLF, inputs);
   tv = 0.0;
   addLoad(mlLF, tv);

   auto normLF = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   // std::cout << "normLF: " << normLF << "\n";

   REQUIRE(normLF == Approx(1.8280057201).margin(1e-10));

   setInputs(mlLF, inputs);
   tv = 0.0;
   addLoad(mlLF, tv);

   normLF = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   // std::cout << "normLF: " << normLF << "\n";

   REQUIRE(normLF == Approx(1.8280057201).margin(1e-10));
}

TEST_CASE("MagneticLoad vectorJacobianProduct wrt mesh_coords")
{
   Mesh smesh = buildMesh(10, 10);
   ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   auto p = 2;
   const auto dim = mesh.Dimension();

   // get the finite-element space for the state
   ND_FECollection fec(p, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   // create mag_coeff coefficient
   // VectorFunctionCoefficient mag_coeff(dim, northMagnetizationSource);
   Vector mag_const(3); mag_const = 1.0;
   VectorConstantCoefficient mag_coeff(mag_const);

   // create nu coeff
   ConstantCoefficient nu(1.0);

   MagneticLoad load(fes, mag_coeff, nu);
   MachLoad ml(load);

   // extract mesh nodes and get their finite-element space
   auto &x_nodes = *dynamic_cast<mfem::ParGridFunction*>(mesh.GetNodes());
   auto &mesh_fes = *x_nodes.ParFESpace();

   auto inputs = MachInputs({
      {"mesh_coords", &x_nodes}
   });
   setInputs(ml, inputs);

   HypreParVector load_bar(&fes);
   {
      for (int i = 0; i < load_bar.Size(); ++i)
      {
         load_bar(i) = uniform_rand(gen);
      }
   }

   HypreParVector wrt_bar(&mesh_fes); wrt_bar = 0.0;
   vectorJacobianProduct(ml, load_bar, "mesh_coords", wrt_bar);

   // initialize the vector that we use to perturb the mesh nodes
   ParGridFunction v(&mesh_fes);
   VectorFunctionCoefficient pert(3, [](const mfem::Vector &x, mfem::Vector &u)
   {
      for (int i = 0; i < u.Size(); ++i)
         u(i) = uniform_rand(gen);
   });
   v.ProjectCoefficient(pert);
   HypreParVector v_tv(&mesh_fes);
   v.ParallelAssemble(v_tv);

   double dJdx_v = wrt_bar * v_tv;

   // now compute the finite-difference approximation...
   auto delta = 1e-5;

   HypreParVector load_vec(&fes);
   ParGridFunction x_pert(x_nodes);
   x_pert.Add(delta, v);
   mesh.SetNodes(x_pert);
   fes.Update();
   inputs.at("mesh_coords") = &x_pert;
   setInputs(ml, inputs);
   load_vec = 0.0;
   addLoad(ml, load_vec);
   double dJdx_v_fd = load_bar * load_vec;

   x_pert.Add(-2 * delta, v);
   mesh.SetNodes(x_pert);
   fes.Update();
   inputs.at("mesh_coords") = &x_pert;
   setInputs(ml, inputs);
   load_vec = 0.0;
   addLoad(ml, load_vec);
   dJdx_v_fd -= load_bar * load_vec;
   dJdx_v_fd /= 2*delta;

   mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
   fes.Update();

   std::cout << "dJdx_v: " << dJdx_v << "\n";
   std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

Mesh buildMesh(int nxy, int nz)
{
   /// generate a simple hex mesh
   auto mesh = Mesh::MakeCartesian3D(nxy, nxy, nz,
                                     Element::TETRAHEDRON, 1.0,
                                     1.0, (double)nz / (double)nxy, true);

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      Element *elem = mesh.GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh.GetVertex(verts[i]);
         if (vtx[1] <= 0.5)
         {
            below = below & true;
         }
         else
         {
            below = below & false;
         }
      }
      if (below)
      {
         elem->SetAttribute(1);
      }
      else
      {
         elem->SetAttribute(2);
      }
   }
   return mesh;
}

template <typename xdouble = double>
void north_magnetization(const xdouble& remnant_flux,
                         const xdouble *x,
                         xdouble *M)
{
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0]*r[0] + r[1]*r[1]);
   M[0] = r[0] * remnant_flux / norm_r;
   M[1] = r[1] * remnant_flux / norm_r;
   M[2] = 0.0;
}

void northMagnetizationSource(const Vector &x,
                              Vector &M)
{
   constexpr auto remnant_flux = 1.0;
   north_magnetization(remnant_flux, x.GetData(), M.GetData());
}

template <typename xdouble = double>
void south_magnetization(const xdouble& remnant_flux,
                         const xdouble *x,
                         xdouble *M)
{
   xdouble r[] = {0.0, 0.0, 0.0};
   r[0] = x[0];
   r[1] = x[1];
   xdouble norm_r = sqrt(r[0]*r[0] + r[1]*r[1]);
   M[0] = -r[0] * remnant_flux / norm_r;
   M[1] = -r[1] * remnant_flux / norm_r;
   M[2] = 0.0;
}

void southMagnetizationSource(const Vector &x,
                              Vector &M)
{
   constexpr auto remnant_flux = 1.0;
   south_magnetization(remnant_flux, x.GetData(), M.GetData());
}
