#ifndef MISO_SURFACE
#define MISO_SURFACE

#include "mfem.hpp"

#include "kdtree.hpp"
#include "utils.hpp"

namespace miso
{
/// A geometry representation for distance calculations
/// \tparam dim - number of spatial dimensions (only `dim=2` is tested)
/// \warning The class has only been tested thoroughly for spatial degree
/// `dim=2` and degree p=1 and p=2 meshes. Higher degree meshes pose changes
/// because the number of local minimizers increases (i.e. more than one point
/// on the surface satisfies the first-order optimality conditions).  For
/// dimension `dim=3`, the code is mostly ready, but we need to handle cases
/// where the closest point is on an edge and the optimization becomes
/// constrained.
/// \note When we need to differentiation this code, we will need to think
/// about how to account for the sensitivities with respect to the mesh nodes
/// (the sensitivity with respect to the points we provide to `calcDistance` is
/// straightforward).  One option is to expand this class to act as the
/// underlying geometry parameterization.
template <int dim>
class Surface
{
public:
   /// Construct a Surface object using a copy of an existing surface mesh
   /// \param[in] ext_mesh - external mesh that defines surface directly
   /// For this constructor, the spatial dimension must be one dimension
   /// greater than the parametric dimension.
   Surface(mfem::Mesh &ext_mesh);

   /// Construct a Surface object by extracting surface mesh from volume mesh
   /// \param[in] vol_mesh - volume mesh to extract surface mesh from
   /// \param[in] bdr_attr_marker - extract boundary elements with nonzero attr.
   /// \note The definition was adapted from the mesh-explorer.cpp miniapp
   Surface(mfem::Mesh &vol_mesh, mfem::Array<int> &bdr_attr_marker);

   /// Destructor, which deletes the surface mesh object
   ~Surface() { delete mesh; }

   /// Find the distance from `x` to the surface
   /// \param[in] x - the point whose distance to the surface we want
   /// \returns the distance
   double calcDistance(const mfem::Vector &x);

   /// Find the distance from `x` to element defined by `trans`.
   /// \param[in] trans - coordinate transformation for an element
   /// \param[in] x - the point whose distance to the surface we want
   /// \returns the distance
   double solveDistance(mfem::ElementTransformation &trans,
                        const mfem::Vector &x);

private:
   // Some aliases to high template parameters
   using Point = point<double, dim>;
   using KDTree = kdtree<double, dim>;

   /// The surface is defined using an mfem Mesh object.
   mfem::Mesh *mesh;
   /// kd-tree used to narrow search of candidate elements
   KDTree tree;

#ifndef MFEM_THREAD_SAFE
   /// reference space integration point used in Newton's method
   mfem::IntegrationPoint ip;
   /// reference space integration point used in Newton's method
   mfem::IntegrationPoint ip_new;
   /// The difference between the surface point and the target point
   mfem::Vector res;
   /// The gradient of the least-squares objective
   mfem::Vector gradient;
   /// The Newton step to update the parametric coordinates
   mfem::Vector step;
   /// Hessian vector product; here, Hessian refers to the coordinate transform
   mfem::Vector hess_vec;
   /// Storage for the reference coordinates
   mfem::Vector xi;
   /// Storage for the Hessian
   mfem::DenseMatrix Hess;
#endif

   /// Constructs the kd-tree data structure based on stored *mesh object
   void buildKDTree();

   double solveConstrained()
   {
      throw MISOException("solveConstrained is not implemented yet!");
   }
};

}  // namespace miso

#include "surface_def.hpp"

#endif
