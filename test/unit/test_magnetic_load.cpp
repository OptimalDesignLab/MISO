#include <iostream>

#include "catch.hpp"
#include "json.hpp"
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
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("MagneticLoad Test",
          "[MagneticLoad]")
{

   std::unique_ptr<Mesh> smesh = buildMesh(10, 10);
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));
   mesh->ReorientTetMesh();

   auto p = 2;
   const auto dim = mesh->Dimension();

   // get the finite-element space for the state
   ND_FECollection fec(p, dim);
   ParFiniteElementSpace fes(mesh.get(), &fec);

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

   MagneticLoad load(fes, mag_coeff, nu);
   MachLoad ml(load);

   std::unique_ptr<HypreParVector> tv(fes.NewTrueDofVector());

   MachInputs inputs;
   setInputs(ml, inputs);
   assemble(ml, *tv);

   auto norm = ParNormlp(*tv, 2.0, MPI_COMM_WORLD);
   std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(1.3163956075).margin(1e-10));

   setInputs(ml, inputs);
   *tv = 0.0;
   assemble(ml, *tv);

   norm = ParNormlp(*tv, 2.0, MPI_COMM_WORLD);
   std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(1.3163956075).margin(1e-10));

}

std::unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(new Mesh(nxy, nxy, nz,
                              Element::HEXAHEDRON, true, 1.0,
                              1.0, (double)nz / (double)nxy, true));

   mesh->ReorientTetMesh();

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh->GetNE(); ++i)
   {
      Element *elem = mesh->GetElement(i);

      Array<int> verts;
      elem->GetVertices(verts);

      bool below = true;
      for (int i = 0; i < 4; ++i)
      {
         auto vtx = mesh->GetVertex(verts[i]);
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
