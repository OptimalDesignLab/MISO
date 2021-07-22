#include <iostream>

#include "catch.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "coefficient.hpp"
#include "mach_load.hpp"
#include "current_load.hpp"

template <typename xdouble = double>
void box1_current(const xdouble *x,
                  xdouble *J);

void box1CurrentSource(const mfem::Vector &x,
                       mfem::Vector &J);

void box1CurrentSourceRevDiff(const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar);

template <typename xdouble = double>
void box2_current(const xdouble *x,
                  xdouble *J);

void box2CurrentSource(const mfem::Vector &x,
                       mfem::Vector &J);

void box2CurrentSourceRevDiff(const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar);

using namespace mach;
using namespace mfem;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("CurrentLoad Test",
          "[CurrentLoad]")
{

   std::unique_ptr<Mesh> smesh = buildMesh(4, 4);
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));
   mesh->ReorientTetMesh();

   auto p = 2;
   const auto dim = mesh->Dimension();

   // get the finite-element space for the state
   ND_FECollection fec(p, dim);
   ParFiniteElementSpace fes(mesh.get(), &fec);

   // create current_coeff coefficient
   VectorMeshDependentCoefficient current_coeff;
   {
      std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
         new VectorFunctionCoefficient(dim,
                                       box1CurrentSource,
                                       box1CurrentSourceRevDiff));
      current_coeff.addCoefficient(1, move(temp_coeff));
   }
   {
      std::unique_ptr<mfem::VectorCoefficient> temp_coeff(
         new VectorFunctionCoefficient(dim,
                                       box2CurrentSource,
                                       box2CurrentSourceRevDiff));
      current_coeff.addCoefficient(2, move(temp_coeff));
   }

   CurrentLoad load(fes, current_coeff);

   MachLoad ml(load);

   auto inputs = MachInputs({
      {"current_density", 1.0}
   });

   std::unique_ptr<HypreParVector> tv(fes.NewTrueDofVector());

   setInputs(ml, inputs);

   *tv = 0.0;
   addLoad(ml, *tv);

   auto norm = ParNormlp(*tv, 2.0, MPI_COMM_WORLD);
   std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(0.3186887196).margin(1e-10));

   inputs.at("current_density") = 2.0;
   setInputs(ml, inputs);

   *tv = 0.0;
   addLoad(ml, *tv);

   norm = ParNormlp(*tv, 2.0, MPI_COMM_WORLD);
   std::cout << "norm: " << norm << "\n";

   REQUIRE(norm == Approx(0.6373774392).margin(1e-10));

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

template <typename xdouble>
void box1_current(const xdouble *x,
                  xdouble *J)
{
   for (int i = 0; i < 3; ++i)
   {
      J[i] = 0.0;
   }
	xdouble y = x[1] - .5;
   J[2] = -6*y;
}

void box1CurrentSource(const mfem::Vector &x,
                       mfem::Vector &J)
{
   box1_current(x.GetData(), J.GetData());
}

void box1CurrentSourceRevDiff(const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar)
{
   // mfem::DenseMatrix source_jac(3);
   // // declare vectors of active input variables
   // std::vector<adouble> x_a(x.Size());
   // // copy data from mfem::Vector
   // adept::set_values(x_a.data(), x.Size(), x.GetData());
   // // start recording
   // diff_stack.new_recording();
   // // the depedent variable must be declared after the recording
   // std::vector<adouble> J_a(x.Size());
   // box1_current<adouble>(x_a.data(), J_a.data());
   // // set the independent and dependent variable
   // diff_stack.independent(x_a.data(), x.Size());
   // diff_stack.dependent(J_a.data(), x.Size());
   // // calculate the jacobian w.r.t state vaiables
   // diff_stack.jacobian(source_jac.GetData());
   // source_jac.MultTranspose(V_bar, x_bar);
}

template <typename xdouble>
void box2_current(const xdouble *x,
                  xdouble *J)
{
   for (int i = 0; i < 3; ++i)
   {
      J[i] = 0.0;
   }
	xdouble y = x[1] - .5;
   J[2] = 6*y;
}

void box2CurrentSource(const mfem::Vector &x,
                       mfem::Vector &J)
{
   box2_current(x.GetData(), J.GetData());
}

void box2CurrentSourceRevDiff(const mfem::Vector &x,
                              const mfem::Vector &V_bar,
                              mfem::Vector &x_bar)
{
   // mfem::DenseMatrix source_jac(3);
   // // declare vectors of active input variables
   // std::vector<adouble> x_a(x.Size());
   // // copy data from mfem::Vector
   // adept::set_values(x_a.data(), x.Size(), x.GetData());
   // // start recording
   // diff_stack.new_recording();
   // // the depedent variable must be declared after the recording
   // std::vector<adouble> J_a(x.Size());
   // box2_current<adouble>(x_a.data(), J_a.data());
   // // set the independent and dependent variable
   // diff_stack.independent(x_a.data(), x.Size());
   // diff_stack.dependent(J_a.data(), x.Size());
   // // calculate the jacobian w.r.t state vaiables
   // diff_stack.jacobian(source_jac.GetData());
   // source_jac.MultTranspose(V_bar, x_bar);
}
