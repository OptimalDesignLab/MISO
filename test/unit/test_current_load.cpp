#include <iostream>

// #include "adept.h"
#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "coefficient.hpp"
#include "mach_load.hpp"
#include "current_load.hpp"

void simpleCurrent(const mfem::Vector &x,
                   mfem::Vector &J);

void simpleCurrentRevDiff(const mfem::Vector &x,
                          const mfem::Vector &J_bar,
                          mfem::Vector &x_bar);

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

static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

using namespace mach;
using namespace mfem;
// using adept::adouble;
// static adept::Stack diff_stack;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
std::unique_ptr<Mesh> buildMesh(int nxy,
                                int nz);

TEST_CASE("CurrentLoad setInputs")
{

   std::unique_ptr<Mesh> smesh = buildMesh(4, 4);
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));
   mesh->EnsureNodes();

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

   nlohmann::json options;
   CurrentLoad load(fes, options, current_coeff);

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

TEST_CASE("CurrentLoad vectorJacobianProduct wrt current_density")
{
   std::unique_ptr<Mesh> smesh = buildMesh(1, 1);
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));
   
   mesh->EnsureNodes();

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

   nlohmann::json options;
   CurrentLoad load(fes, options, current_coeff);

   MachLoad ml(load);

   auto current_density = 1e6;
   auto inputs = MachInputs({
      {"current_density", current_density}
   });
   setInputs(ml, inputs);

   HypreParVector load_bar(&fes);
   {
      // std::default_random_engine gen;
      // std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);
      for (int i = 0; i < load_bar.Size(); ++i)
      {
         load_bar(i) = uniform_rand(gen);
      }
   }
   double wrt_bar = vectorJacobianProduct(ml, load_bar, "current_density");

   /// somewhat large step size since the magnitude of current density is large
   auto delta = 1e-2;
   HypreParVector tv(&fes);
   inputs.at("current_density") = current_density + delta;
   setInputs(ml, inputs);
   tv = 0.0;
   addLoad(ml, tv);
   double wrt_bar_fd = load_bar * tv;

   inputs.at("current_density") = current_density - delta;
   setInputs(ml, inputs);
   tv = 0.0;
   addLoad(ml, tv);
   wrt_bar_fd -= load_bar * tv;
   wrt_bar_fd /= 2*delta;

   // std::cout << "wrt_bar: " << wrt_bar << "\n";
   // std::cout << "wrt_bar_fd: " << wrt_bar_fd << "\n";
   REQUIRE(wrt_bar == Approx(wrt_bar_fd));
}

TEST_CASE("CurrentLoad vectorJacobianProduct wrt mesh_coords")
{
   std::unique_ptr<Mesh> smesh = buildMesh(1, 1);
   std::unique_ptr<ParMesh> mesh(new ParMesh(MPI_COMM_WORLD, *smesh));
   
   mesh->EnsureNodes();

   auto p = 2;
   const auto dim = mesh->Dimension();

   // get the finite-element space for the state
   ND_FECollection fec(p, dim);
   ParFiniteElementSpace fes(mesh.get(), &fec);

   VectorFunctionCoefficient current_coeff(dim,
                                           simpleCurrent,
                                           simpleCurrentRevDiff);

   nlohmann::json options;
   CurrentLoad load(fes, options, current_coeff);
   MachLoad ml(load);

   // extract mesh nodes and get their finite-element space
   auto &x_nodes = *dynamic_cast<mfem::ParGridFunction*>(mesh->GetNodes());
   auto &mesh_fes = *x_nodes.ParFESpace();

   auto current_density = 1e6;
   auto inputs = MachInputs({
      {"current_density", current_density},
      {"mesh_coords", x_nodes}
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
   mesh->SetNodes(x_pert);
   fes.Update();
   inputs.at("mesh_coords") = x_pert;
   setInputs(ml, inputs);
   load_vec = 0.0;
   addLoad(ml, load_vec);
   double dJdx_v_fd = load_bar * load_vec;

   x_pert.Add(-2 * delta, v);
   mesh->SetNodes(x_pert);
   fes.Update();
   inputs.at("mesh_coords") = x_pert;
   setInputs(ml, inputs);
   load_vec = 0.0;
   addLoad(ml, load_vec);
   dJdx_v_fd -= load_bar * load_vec;
   dJdx_v_fd /= 2*delta;

   mesh->SetNodes(x_nodes); // remember to reset the mesh nodes
   fes.Update();

   std::cout << "dJdx_v: " << dJdx_v << "\n";
   std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";

   REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
}

std::unique_ptr<Mesh> buildMesh(int nxy, int nz)
{
   // generate a simple tet mesh
   std::unique_ptr<Mesh> mesh(
      new Mesh(Mesh::MakeCartesian3D(nxy, nxy, nz,
                                     Element::HEXAHEDRON,
                                     1.0, 1.0, (double)nz / (double)nxy, true)));

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

void simpleCurrent(const mfem::Vector &x,
                   mfem::Vector &J)
{
   J = 0.0;
   J[2] = 1.0;
}

void simpleCurrentRevDiff(const mfem::Vector &x,
                          const mfem::Vector &J_bar,
                          mfem::Vector &x_bar)
{

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
