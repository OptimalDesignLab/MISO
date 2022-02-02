#include <iostream>
#include <random>

#include "adept.h"
#include "catch.hpp"
#include "nlohmann/json.hpp"
#include "mfem.hpp"

#include "utils.hpp"
#include "mach_load.hpp"
#include "magnetostatic.hpp"
#include "material_library.hpp"

using namespace mach;

/// Generate mesh 
/// \param[in] nxy - number of nodes in the x and y directions
/// \param[in] nz - number of nodes in the z direction
mfem::Mesh buildMesh(int nxy, int nz);

static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

auto options = R"(
{
   "space-dis": {
      "degree": 2,
      "basis-type": "ND"
   },
   "current": {
      "group1": {
         "ring": [1]
      }
   },
   "magnets": {
      "Nd2Fe14B": {
         "ccw": [1]
      }
   }
})"_json;

TEST_CASE("MagnetostaticLoad Value Test")
{
   mfem::Mesh smesh = buildMesh(3, 3);
   mfem::ParMesh mesh(MPI_COMM_WORLD, smesh);
   mesh.EnsureNodes();

   const auto dim = mesh.Dimension();

   // create nu coeff
   mfem::ConstantCoefficient nu(1.0);

   adept::Stack diff_stack;
   std::map<std::string, FiniteElementState> fields;
   fields.emplace(std::piecewise_construct,
                  std::forward_as_tuple("state"),
                  std::forward_as_tuple(mesh, options["space-dis"]));

   auto &mesh_gf = *dynamic_cast<mfem::ParGridFunction *>(mesh.GetNodes());
   auto *mesh_fespace = mesh_gf.ParFESpace();
   /// create new state vector copying the mesh's fe space
   fields.emplace(
         std::piecewise_construct,
         std::forward_as_tuple("mesh_coords"),
         std::forward_as_tuple(mesh, *mesh_fespace, "mesh_coords"));
   FiniteElementState &mesh_coords = fields.at("mesh_coords");
   /// set the values of the new GF to those of the mesh's old nodes
   mesh_coords.gridFunc() = mesh_gf;
   /// tell the mesh to use this GF for its Nodes
   /// (and that it doesn't own it)
   mesh.NewNodes(mesh_coords.gridFunc(), false);

   nlohmann::json materials(material_library);

   auto &fes = fields.at("state").space();
   MagnetostaticLoad load(diff_stack, fes, fields, options, materials, nu);

   mfem::HypreParVector tv(&fes);

   auto inputs = MachInputs({
      {"current_density:group1", 1.0}
   });

   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   auto norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "\nnorm: " << norm << "\n\n";

   // REQUIRE(norm == Approx(1.8696677851).margin(1e-10));

   inputs.at("current_density:group1") = 2.0;
   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   norm = mfem::ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "\nnorm: " << norm << "\n\n";

   // REQUIRE(norm == Approx(1.9505429368).margin(1e-10));

   inputs.at("current_density:group1") = 0.0;
   setInputs(load, inputs);
   tv = 0.0;
   addLoad(load, tv);

   norm = ParNormlp(tv, 2.0, MPI_COMM_WORLD);
   std::cout << "\nnorm: " << norm << "\n\n";

   // REQUIRE(norm == Approx(1.8280057201).margin(1e-10));

}

// TEST_CASE("MagnetostaticLoad vectorJacobianProduct wrt current_density")
// {
//    Mesh smesh = buildMesh(3, 3);
//    ParMesh mesh(MPI_COMM_WORLD, smesh);
//    mesh.EnsureNodes();

//    auto p = 2;
//    const auto dim = mesh.Dimension();

//    // get the finite-element space for the state
//    ND_FECollection fec(p, dim);
//    ParFiniteElementSpace fes(&mesh, &fec);

//    // create current_coeff coefficient
//    VectorFunctionCoefficient current_coeff(dim,
//                                            simpleCurrent,
//                                            simpleCurrentRevDiff);

//    // create mag_coeff coefficient
//    Vector mag_const(3); mag_const = 1.0;
//    VectorConstantCoefficient mag_coeff(mag_const);

//    // create nu coeff
//    ConstantCoefficient nu(1.0);

//    MagnetostaticLoad load(fes, current_coeff, mag_coeff, nu);
//    MachLoad ml(load);

//    auto current_density = 1e6;
//    auto inputs = MachInputs({
//       {"current_density", current_density}
//    });
//    setInputs(ml, inputs);

//    HypreParVector load_bar(&fes);
//    {
//       // std::default_random_engine gen;
//       // std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);
//       for (int i = 0; i < load_bar.Size(); ++i)
//       {
//          load_bar(i) = uniform_rand(gen);
//       }
//    }
//    double wrt_bar = vectorJacobianProduct(ml, load_bar, "current_density");

//    /// somewhat large step size since the magnitude of current density is large
//    auto delta = 1e-2;
//    HypreParVector tv(&fes);
//    inputs.at("current_density") = current_density + delta;
//    setInputs(ml, inputs);
//    tv = 0.0;
//    addLoad(ml, tv);
//    double wrt_bar_fd = load_bar * tv;

//    inputs.at("current_density") = current_density - delta;
//    setInputs(ml, inputs);
//    tv = 0.0;
//    addLoad(ml, tv);
//    wrt_bar_fd -= load_bar * tv;
//    wrt_bar_fd /= 2*delta;

//    // std::cout << "wrt_bar: " << wrt_bar << "\n";
//    // std::cout << "wrt_bar_fd: " << wrt_bar_fd << "\n";
//    REQUIRE(wrt_bar == Approx(wrt_bar_fd));
// }

// TEST_CASE("MagnetostaticLoad vectorJacobianProduct wrt mesh_coords")
// {
//    Mesh smesh = buildMesh(3, 3);
//    ParMesh mesh(MPI_COMM_WORLD, smesh);
//    mesh.EnsureNodes();

//    auto p = 2;
//    const auto dim = mesh.Dimension();

//    // get the finite-element space for the state
//    ND_FECollection fec(p, dim);
//    ParFiniteElementSpace fes(&mesh, &fec);

//    // create current_coeff coefficient
//    VectorFunctionCoefficient current_coeff(dim,
//                                            simpleCurrent,
//                                            simpleCurrentRevDiff);

//    // create mag_coeff coefficient
//    // VectorFunctionCoefficient mag_coeff(dim, northMagnetizationSource);
//    Vector mag_const(3); mag_const = 1.0;
//    VectorConstantCoefficient mag_coeff(mag_const);

//    // create nu coeff
//    ConstantCoefficient nu(1.0);

//    MagnetostaticLoad load(fes, current_coeff, mag_coeff, nu);
//    MachLoad ml(load);

//    // extract mesh nodes and get their finite-element space
//    auto &x_nodes = *dynamic_cast<mfem::ParGridFunction*>(mesh.GetNodes());
//    auto &mesh_fes = *x_nodes.ParFESpace();

//    auto inputs = MachInputs({
//       {"mesh_coords", x_nodes}
//    });
//    setInputs(ml, inputs);

//    HypreParVector load_bar(&fes);
//    {
//       for (int i = 0; i < load_bar.Size(); ++i)
//       {
//          load_bar(i) = uniform_rand(gen);
//       }
//    }

//    HypreParVector wrt_bar(&mesh_fes); wrt_bar = 0.0;
//    vectorJacobianProduct(ml, load_bar, "mesh_coords", wrt_bar);

//    // initialize the vector that we use to perturb the mesh nodes
//    ParGridFunction v(&mesh_fes);
//    VectorFunctionCoefficient pert(3, [](const mfem::Vector &x, mfem::Vector &u)
//    {
//       for (int i = 0; i < u.Size(); ++i)
//          u(i) = uniform_rand(gen);
//    });
//    v.ProjectCoefficient(pert);
//    HypreParVector v_tv(&mesh_fes);
//    v.ParallelAssemble(v_tv);

//    double dJdx_v = wrt_bar * v_tv;

//    // now compute the finite-difference approximation...
//    auto delta = 1e-5;

//    HypreParVector load_vec(&fes);
//    ParGridFunction x_pert(x_nodes);
//    x_pert.Add(delta, v);
//    mesh.SetNodes(x_pert);
//    fes.Update();
//    inputs.at("mesh_coords") = x_pert;
//    setInputs(ml, inputs);
//    load_vec = 0.0;
//    addLoad(ml, load_vec);
//    double dJdx_v_fd = load_bar * load_vec;

//    x_pert.Add(-2 * delta, v);
//    mesh.SetNodes(x_pert);
//    fes.Update();
//    inputs.at("mesh_coords") = x_pert;
//    setInputs(ml, inputs);
//    load_vec = 0.0;
//    addLoad(ml, load_vec);
//    dJdx_v_fd -= load_bar * load_vec;
//    dJdx_v_fd /= 2*delta;

//    mesh.SetNodes(x_nodes); // remember to reset the mesh nodes
//    fes.Update();

//    std::cout << "dJdx_v: " << dJdx_v << "\n";
//    std::cout << "dJdx_v_fd: " << dJdx_v_fd << "\n";

//    REQUIRE(dJdx_v == Approx(dJdx_v_fd).margin(1e-8));
// }

mfem::Mesh buildMesh(int nxy, int nz)
{
   /// generate a simple tet mesh
   auto mesh = mfem::Mesh::MakeCartesian3D(nxy, nxy, nz,
                                     mfem::Element::TETRAHEDRON, 1.0,
                                     1.0, (double)nz / (double)nxy, true);

   // assign attributes to top and bottom sides
   for (int i = 0; i < mesh.GetNE(); ++i)
   {
      auto *elem = mesh.GetElement(i);
      elem->SetAttribute(1);

      // Array<int> verts;
      // elem->GetVertices(verts);

      // bool below = true;
      // for (int i = 0; i < 4; ++i)
      // {
      //    auto vtx = mesh.GetVertex(verts[i]);
      //    if (vtx[1] <= 0.5)
      //    {
      //       below = below & true;
      //    }
      //    else
      //    {
      //       below = below & false;
      //    }
      // }
      // if (below)
      // {
      //    elem->SetAttribute(1);
      // }
      // else
      // {
      //    elem->SetAttribute(2);
      // }
   }
   return mesh;
}
