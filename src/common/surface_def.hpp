#ifndef MACH_SURFACE_DEF
#define MACH_SURFACE_DEF

#include "mfem.hpp"

#include "kdtree.hpp"
#include "utils.hpp"
#include "surface.hpp"

namespace mach
{
template <int dim>
Surface<dim>::Surface(mfem::Mesh &ext_mesh)
{
   using namespace mfem;
   if (dim != 2)
   {
      throw MachException("Surface class not tested for dim != 2.");
   }
   // check that this is a valid surface mesh, in terms of dimensions
   MFEM_ASSERT(dim == ext_mesh.Dimension() + 1, "Invalid surface mesh.");
   MFEM_ASSERT(dim == ext_mesh.SpaceDimension(), "Invalid surface mesh.");
   mesh = new Mesh(ext_mesh);
   // build out the kd-tree data structure
   buildKDTree();
}

template <int dim>
Surface<dim>::Surface(mfem::Mesh &vol_mesh, mfem::Array<int> &bdr_attr_marker)
{
   using namespace mfem;
   if (dim != 2)
   {
      throw MachException("Surface class not tested for dim != 2.");
   }
   // check that this is a valid volume mesh, in terms of dimensions
   MFEM_ASSERT(dim == vol_mesh.Dimension(), "Invalid volume mesh.");
   MFEM_ASSERT(dim == vol_mesh.SpaceDimension(), "Invalid volume mesh.");

   // Determine mapping from vertex to boundary vertex
   Array<int> v2v(vol_mesh.GetNV());
   v2v = -1;
   int num_belem = 0;
   for (int i = 0; i < vol_mesh.GetNBE(); i++)
   {
      Element *el = vol_mesh.GetBdrElement(i);
      int bdr_attr = el->GetAttribute();
      if (bdr_attr_marker[bdr_attr - 1] == 0)
      {
         continue;
      }
      num_belem++;
      int *v = el->GetVertices();
      int nv = el->GetNVertices();
      for (int j = 0; j < nv; j++)
      {
         v2v[v[j]] = 0;
      }
   }
   // Count the number of unique boundary vertices and index them
   int nbvt = 0;
   for (int i = 0; i < v2v.Size(); i++)
   {
      if (v2v[i] == 0)
      {
         v2v[i] = nbvt++;
      }
   }

   // Create the new boundary mesh
   mesh = new Mesh(
       vol_mesh.Dimension() - 1, nbvt, num_belem, 0, vol_mesh.SpaceDimension());

   // Copy vertices to the boundary mesh
   nbvt = 0;
   for (int i = 0; i < v2v.Size(); i++)
   {
      if (v2v[i] >= 0)
      {
         double *c = vol_mesh.GetVertex(i);
         mesh->AddVertex(c);
         nbvt++;
      }
   }

   // Copy elements to the boundary mesh
   int bv[4];
   for (int i = 0; i < vol_mesh.GetNBE(); i++)
   {
      Element *el = vol_mesh.GetBdrElement(i);
      int bdr_attr = el->GetAttribute();
      if (bdr_attr_marker[bdr_attr - 1] == 0)
      {
         continue;
      }

      int *v = el->GetVertices();
      int nv = el->GetNVertices();

      for (int j = 0; j < nv; j++)
      {
         bv[j] = v2v[v[j]];
      }

      switch (el->GetGeometryType())
      {
      case Geometry::SEGMENT:
         mesh->AddSegment(bv, el->GetAttribute());
         break;
      case Geometry::TRIANGLE:
         mesh->AddTriangle(bv, el->GetAttribute());
         break;
      case Geometry::SQUARE:
         mesh->AddQuad(bv, el->GetAttribute());
         break;
      default:
         break;  /// This should not happen
      }
   }
   mesh->FinalizeTopology();

   // Copy GridFunction describing nodes if present
   if (vol_mesh.GetNodes() != nullptr)
   {
      FiniteElementSpace *fes = vol_mesh.GetNodes()->FESpace();
      const FiniteElementCollection *fec = fes->FEColl();
      if (dynamic_cast<const H1_FECollection *>(fec) != nullptr)
      {
         FiniteElementCollection *fec_copy =
             FiniteElementCollection::New(fec->Name());
         auto *fes_copy = new FiniteElementSpace(*fes, mesh, fec_copy);
         auto *bdr_nodes = new GridFunction(fes_copy);
         bdr_nodes->MakeOwner(fec_copy);

         mesh->NewNodes(*bdr_nodes, true);

         Array<int> vdofs;
         Array<int> bvdofs;
         Vector v;
         for (int i = 0; i < vol_mesh.GetNBE(); i++)
         {
            int bdr_attr = vol_mesh.GetBdrAttribute(i);
            if (bdr_attr_marker[bdr_attr - 1] == 0)
            {
               continue;
            }

            fes->GetBdrElementVDofs(i, vdofs);
            vol_mesh.GetNodes()->GetSubVector(vdofs, v);

            fes_copy->GetElementVDofs(i, bvdofs);
            bdr_nodes->SetSubVector(bvdofs, v);
         }
      }
      else
      {
         throw MachException("Discontinuous nodes not yet supported");
      }
   }
   // build out the kd-tree data structure
   buildKDTree();
}

template <int dim>
void Surface<dim>::buildKDTree()
{
   using namespace mfem;
   if (mesh == nullptr)
   {
      throw MachException("Cannot call buildKDTree with empty mesh.");
   }
   mesh->EnsureNodes();
   mesh->GetNodes()->FESpace()->BuildDofToArrays();
   // loop over the mesh nodes, and add to the tree
   GridFunction &nodes = *mesh->GetNodes();
   tree.set_size(nodes.FESpace()->GetNDofs());
   Vector node(mesh->SpaceDimension());
   for (int i = 0; i < nodes.FESpace()->GetNDofs(); ++i)
   {
      for (int d = 0; d < dim; d++)
      {
         node(d) = nodes(nodes.FESpace()->DofToVDof(i, d));
      }
      tree.add_node(node, i);
   }
   tree.finalize();
}

template <int dim>
double Surface<dim>::calcDistance(const mfem::Vector &x)
{
   using namespace mfem;
   const FiniteElementSpace *fes = mesh->GetNodalFESpace();
   // search for the element whose center is closest to x
   int i = tree.nearest(x);
   int e = fes->GetElementForDof(i);

   // Apply Newton's method to find the closest point in ref space
   ElementTransformation *trans = fes->GetElementTransformation(e);
   double dist = solveDistance(*trans, x);
   return dist;
}

template <int dim>
double Surface<dim>::solveDistance(mfem::ElementTransformation &trans,
                                   const mfem::Vector &x)
{
   using namespace mfem;

#ifdef MFEM_THREAD_SAFE
   Vector res(dim), gradient(dim - 1), step(dim - 1),
       hess_vec(dim * (dim - 1) / 2);
   Vector xi(dim - 1);
   IntegrationPoint ip, ip_new;
   DenseMatrix Hess(dim - 1);
#endif
   res.SetSize(dim);
   gradient.SetSize(dim - 1);
   step.SetSize(dim - 1);
   hess_vec.SetSize(dim * (dim - 1) / 2);
   xi.SetSize(dim - 1);
   Hess.SetSize(dim - 1);

   // use centroid for initial guess for reference coordinate
   ip = Geometries.GetCenter(trans.GetGeometryType());
   trans.SetIntPoint(&ip);

   // evaluate the gradient of the least-squares objective
   const DenseMatrix &Jac = trans.Jacobian();
   trans.Transform(ip, res);
   res -= x;
   Jac.MultTranspose(res, gradient);

#ifdef MFEM_DEBUG
   std::cout << std::endl;
   std::cout << "Surface::solveDistance starting Newton iterations"
             << std::endl;
#endif

   // Loop over Newton iterations
   const int max_iter = 30;
   const double tol = 1e-13;
   bool hit_boundary = false;
   for (int n = 0; n < max_iter; ++n)
   {
#ifdef MFEM_DEBUG
      std::cout << "\titeration " << n << ": optimality = " << gradient.Norml2()
                << ": distance = " << res.Norml2() << std::endl;
#endif
      // check for convergence
      if (gradient.Norml2() < tol)
      {
         return res.Norml2();
      }
      // compute the Hessian for the Newton step
      MultAtB(Jac, Jac, Hess);
      const DenseMatrix &d2Fdx2 = trans.Hessian();
      d2Fdx2.MultTranspose(res, hess_vec);
      int idx = 0;
      for (int i = 0; i < dim - 1; ++i)
      {
         Hess(i, i) += hess_vec(idx);
         ++idx;
         for (int j = i + 1; j < dim - 1; ++j)
         {
            Hess(i, j) += hess_vec(idx);
            Hess(j, i) += hess_vec(idx);
            ++idx;
         }
      }

      // compute the Newton step
      step.Set(-1.0, gradient);
      bool success = LinearSolve(Hess, step.GetData());
      if (!success)
      {
         throw MachException("LinearSolve failed in Surface::solveDistance!");
      }

      // check that this is a descent direction; if not, use scaled negative
      // gradient
      if (InnerProduct(step, gradient) > 0.0)
      {
#ifdef MFEM_DEBUG
         std::cout << "\tNewton step was not a descent direction; "
                   << "switching to steepest descent." << std::endl;
#endif
         step.Set(-0.1 / gradient.Norml2(), gradient);
      }
      ip.Get(xi.GetData(), dim - 1);
      xi += step;
      ip_new.Set(xi.GetData(), dim - 1);
      // check that point is not outside element geometry
      if (!mfem::Geometry::ProjectPoint(trans.GetGeometryType(), ip, ip_new))
      {
         // If we get here, ip_new was outside the element and had to be
         // projected to boundary
#ifdef MFEM_DEBUG
         std::cout << "\tNewton step was projected onto feasible space."
                   << std::endl;
#endif
         if (hit_boundary)
         {
            return res.Norml2();
            // if this happens twice, we should move to constrained
            // optimization for 3D (ie. surface embedded in 3D)
            // solveConstrained();
         }
         hit_boundary = true;
      }

      // prepare for the next iteration
      ip_new.Get(xi.GetData(), dim - 1);
      ip.Set(xi.GetData(), dim - 1);
      trans.SetIntPoint(&ip);
      trans.Jacobian();
      trans.Transform(ip, res);
      res -= x;
      Jac.MultTranspose(res, gradient);
   }
   // If we get here, we exceeded the maximum number of Newton iterations
   throw MachException("Newton solve failed in Surface::solveDistance!");
}

}  // namespace mach

#endif
