#include "mfem.hpp"

#include "mesh_move_integ.hpp"

using namespace mfem;

namespace mach
{
void ElasticityPositionIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &trans,
    const Vector &elfun,
    Vector &elvect)
{
   DenseMatrix elmat;
   AssembleElementMatrix(el, trans, elmat);

   Vector disp(elfun);

   auto &iso_trans = *dynamic_cast<IsoparametricTransformation *>(&trans);
   auto point_mat = iso_trans.GetPointMat();
   point_mat.Transpose();
   Vector point_vec(point_mat.GetData(),
                    point_mat.Height() * point_mat.Width());

   disp -= point_vec;

   elvect.SetSize(elmat.Height());
   elmat.Mult(disp, elvect);
}

}  // namespace mach
