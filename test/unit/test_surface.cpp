#include <fstream>
#include <iostream>

#include "catch.hpp"
#include "mfem.hpp"
#include "surface.hpp"

/// This projects nodes onto the hypersphere
void SnapNodes(mfem::Mesh &mesh)
{   
   using namespace mfem;
   GridFunction &nodes = *mesh.GetNodes();
   Vector node(mesh.SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }
      node /= node.Norml2();
      for (int d = 0; d < mesh.SpaceDimension(); d++)
      {
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
      }
   }
}

TEST_CASE( "calc/solveDistance applied to linear segment", "[surface]")
{
   using namespace mfem;

   int order = 1;
   int Nvert = 2;
   int Nelem = 1;
   Mesh *mesh = new Mesh(1, Nvert, Nelem, 0, 2);
   const double seg_v[2][2] =
   {
      {1, 0}, {0, 1}
   };
   const int seg_e[1][2] =
   {
      {0, 1}
   };
   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(seg_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddSegment(seg_e[j], attribute);
   }
   mesh->FinalizeMesh();

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   
   // Snap nodes to the circle 
   SnapNodes(*mesh);

   // Construct the surface object
   mach::Surface<2> surf(*mesh);

   // Get the relevant element transformation
   ElementTransformation *trans = mesh->GetElementTransformation(0);

   // Test 1: Closest point on the interior of reference domain
   Vector x(2);
   x(0) = 3.0;
   x(1) = 2.0;
   double dist = surf.solveDistance(*trans, x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(2*sqrt(2)).margin(1e-12) );
   REQUIRE( surf.calcDistance(x) == Approx(dist).margin(1e-14) );

   // Test 2: Closest point on the edge of the reference domain
   x(0) = 0.0;
   x(1) = 2.0;
   dist = surf.solveDistance(*trans, x);
   //std::cout << "distance = " << dist << std::endl;      
   REQUIRE( dist == Approx(1.0).margin(1e-12) );
   REQUIRE( surf.calcDistance(x) == Approx(dist).margin(1e-14) );

   delete mesh;
}

TEST_CASE( "calc/solveDistance applied to quadratic segment", "[surface]")
{
   using namespace mfem;

   int order = 2;
   int Nvert = 2;
   int Nelem = 1;
   Mesh *mesh = new Mesh(1, Nvert, Nelem, 0, 2);
   const double seg_v[2][2] =
   {
      {1, 0}, {0, 1}
   };
   const int seg_e[1][2] =
   {
      {0, 1}
   };
   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(seg_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddSegment(seg_e[j], attribute);
   }
   mesh->FinalizeMesh();

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   
   // Snap nodes to the circle 
   SnapNodes(*mesh);

   // TEMP: visualize mesh 
   //std::ofstream sol_ofs("surface.vtk");
   //sol_ofs.precision(14);
   //mesh->PrintVTK(sol_ofs, 20);
   //sol_ofs.close();

   // Construct the surface object
   mach::Surface<2> surf(*mesh);

   // Get the relevant element transformation
   ElementTransformation *trans = mesh->GetElementTransformation(0);

   // Test 1: Closest point on the interior of reference domain
   Vector x(2);
   x(0) = 2.0;
   x(1) = 1.0;
   double dist = surf.solveDistance(*trans, x);
   //std::cout << std::setprecision(15);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(1.24261668385551).margin(1e-12) );
   REQUIRE( surf.calcDistance(x) == Approx(dist).margin(1e-14) );

   // Test 2: Closest point on the edge of the reference domain
   x(1) = -1.0;
   dist = surf.solveDistance(*trans, x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(sqrt(2.0)).margin(1e-12) );
   REQUIRE( surf.calcDistance(x) == Approx(dist).margin(1e-14) );

   delete mesh;
}

TEST_CASE( "calc/solveDistance applied to cubic segment", "[surface]")
{
   using namespace mfem;

   int order = 3;
   int Nvert = 2;
   int Nelem = 1;
   Mesh *mesh = new Mesh(1, Nvert, Nelem, 0, 2);
   const double seg_v[2][2] =
   {
      {-1, 0}, {1, 0}
   };
   const int seg_e[1][2] =
   {
      {0, 1}
   };
   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(seg_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddSegment(seg_e[j], attribute);
   }
   mesh->FinalizeMesh();

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   
   // Snap nodes to the circle 
   //SnapNodes(*mesh);

   // Snap the nodes to a oscillatory cubic function
   GridFunction &nodes = *mesh->GetNodes();
   Vector node(2);
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); i++)
   {
      for (int d = 0; d < 2; d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }
      node(1) = (node(0) + 0.9)*(node(0) - 1)*(node(0) + 1);
      for (int d = 0; d < 2; d++)
      {
         nodes(nodes.FESpace()->DofToVDof(i, d)) = node(d);
      }
   }

   // TEMP: visualize mesh 
   //std::ofstream sol_ofs("surface.vtk");
   //sol_ofs.precision(14);
   //mesh->PrintVTK(sol_ofs, 20);
   //sol_ofs.close();

   // Construct the surface object
   mach::Surface<2> surf(*mesh);

   // Get the relevant element transformation
   ElementTransformation *trans = mesh->GetElementTransformation(0);

   // Test 1: test that local maximizers can be avoided
   Vector x(2);
   x(0) = 0.0;
   x(1) = 1.0;
   double dist = surf.solveDistance(*trans, x);
   //std::cout << std::setprecision(15);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(1.30292258016208).margin(1e-12) );
   REQUIRE( surf.calcDistance(x) == Approx(dist).margin(1e-14) );

#if 0
   // Test 2: Closest point on the edge of the reference domain
   // This test currently fails, because we converge to a local minimizer
   // rather than the global minimizer. 
   x(0) = 0.5;
   x(1) = 1.0;
   dist = surf.solveDistance(*trans, x);
   std::cout << "distance = " << dist << std::endl;
   REQUIRE(dist == Approx(0.5*sqrt(5.0)).margin(1e-12));
#endif

   delete mesh;
}

TEST_CASE( "calcDistance applied to linear mesh", "[surface]")
{
   using namespace mfem;

   int order = 1;
   int Nvert = 4;
   int Nelem = 4;
   Mesh *mesh = new Mesh(1, Nvert, Nelem, 0, 2);
   const double seg_v[4][2] =
   {
      {1, 0}, {0, 1}, {-1, 0}, {0, -1}
   };
   const int seg_e[4][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}
   };
   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(seg_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddSegment(seg_e[j], attribute);
   }
   mesh->FinalizeMesh();

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   
   // Refine the mesh while snapping nodes to the sphere.
   const int ref_levels = 4;
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
      {
         mesh->UniformRefinement();
      }
      SnapNodes(*mesh);
   }

   // TEMP: visualize mesh 
   //std::ofstream sol_ofs("surface.vtk");
   //sol_ofs.precision(14);
   //mesh->PrintVTK(sol_ofs, 20);
   //sol_ofs.close();

   // Construct the surface object
   mach::Surface<2> surf(*mesh);

   // Test 1: Closest point on a vertex
   Vector x(2);
   x(0) = 2.0;
   x(1) = 2.0;
   double dist = surf.calcDistance(x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( surf.calcDistance(x) == Approx(sqrt(8.0) - 1).margin(1e-12) );

   // Test 2: Closest point inside a element
   x(0) = 1.9975909124103448;
   x(1) = 0.09813534865483603;
   dist = surf.calcDistance(x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(1.00120454379483).margin(1e-12) );

   delete mesh;
}

TEST_CASE( "calcDistance applied to quadratic mesh", "[surface]")
{
   using namespace mfem;

   int order = 2;
   int Nvert = 4;
   int Nelem = 4;
   Mesh *mesh = new Mesh(1, Nvert, Nelem, 0, 2);
   const double seg_v[4][2] =
   {
      {1, 0}, {0, 1}, {-1, 0}, {0, -1}
   };
   const int seg_e[4][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}
   };
   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(seg_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddSegment(seg_e[j], attribute);
   }
   mesh->FinalizeMesh();

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   
   // Refine the mesh while snapping nodes to the sphere.
   const int ref_levels = 2;
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
      {
         mesh->UniformRefinement();
      }
      SnapNodes(*mesh);
   }

   // TEMP: visualize mesh 
   //std::ofstream sol_ofs("surface.vtk");
   //sol_ofs.precision(14);
   //mesh->PrintVTK(sol_ofs, 20);
   //sol_ofs.close();

   // Construct the surface object
   mach::Surface<2> surf(*mesh);

   // Test 1: closest point inside an element
   Vector x(2);
   x(0) = -2.0;
   x(1) = 1.0;
   double dist = surf.calcDistance(x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( surf.calcDistance(x) == Approx(1.23626562072248).margin(1e-12) );

   // Test 2: Closest point at a vertex
   x(0) = 0.0;
   x(1) = -2.0;
   dist = surf.calcDistance(x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(1.0).margin(1e-12) );

   delete mesh;
}

TEST_CASE( "calcDistance applied to cubic mesh", "[surface]")
{
   using namespace mfem;

   int order = 3;
   int Nvert = 4;
   int Nelem = 4;
   Mesh *mesh = new Mesh(1, Nvert, Nelem, 0, 2);
   const double seg_v[4][2] =
   {
      {1, 0}, {0, 1}, {-1, 0}, {0, -1}
   };
   const int seg_e[4][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0}
   };
   for (int j = 0; j < Nvert; j++)
   {
      mesh->AddVertex(seg_v[j]);
   }
   for (int j = 0; j < Nelem; j++)
   {
      int attribute = j + 1;
      mesh->AddSegment(seg_e[j], attribute);
   }
   mesh->FinalizeMesh();

   // Set the space for the high-order mesh nodes.
   H1_FECollection fec(order, mesh->Dimension());
   FiniteElementSpace nodal_fes(mesh, &fec, mesh->SpaceDimension());
   mesh->SetNodalFESpace(&nodal_fes);
   
   // Refine the mesh while snapping nodes to the sphere.
   const int ref_levels = 2;
   for (int l = 0; l <= ref_levels; l++)
   {
      if (l > 0) // for l == 0 just perform snapping
      {
         mesh->UniformRefinement();
      }
      SnapNodes(*mesh);
   }

   // TEMP: visualize mesh 
   //std::ofstream sol_ofs("surface.vtk");
   //sol_ofs.precision(14);
   //mesh->PrintVTK(sol_ofs, 20);
   //sol_ofs.close();

   // Construct the surface object
   mach::Surface<2> surf(*mesh);

   // Test 1: closest point inside an element
   Vector x(2);
   x(0) = -2.0;
   x(1) = 1.0;
   double dist = surf.calcDistance(x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( surf.calcDistance(x) == Approx(1.23606070227677).margin(1e-12) );

   // Test 2: Closest point at a vertex; have to go inside the "circle" because
   // the snapped cubic mesh is slightly concave at the vertices.
   x(0) = 0.0;
   x(1) = -0.5;
   dist = surf.calcDistance(x);
   //std::cout << "distance = " << dist << std::endl;
   REQUIRE( dist == Approx(0.5).margin(1e-12) );

   delete mesh;
}
