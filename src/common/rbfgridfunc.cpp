#include <iostream>
#include "rbfgridfunc.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

RBFGridFunction::RBFGridFunction(FiniteElementSpace *f, Array<Vector *> &center,
                                 function<void(const Vector &, Vector &)> F)
{
   basisCenter = center;
   Function = F;
   SetSize(f->GetVDim() * center.Size());
   fes = f;
   fec = NULL;
   UseDevice(true);
}

// void CentGridFunction::ProjectCoefficient(VectorCoefficient &coeff)
// {
//    int vdim = fes->GetVDim();
//    Array<int> vdofs(vdim);
//    Vector vals;

//    int geom = fes->GetMesh()->GetElement(0)->GetGeometryType();
//    const IntegrationPoint &cent = Geometries.GetCenter(geom);
//    const FiniteElement *fe;
//    ElementTransformation *eltransf;
//    for (int i = 0; i < fes->GetNE(); i++)
//    {
//       fe = fes->GetFE(i);
//       // Get the indices of dofs
//       for (int j = 0; j < vdim; j ++)
//       {
//          vdofs[j] = i * vdim +j;
//       }

//       eltransf = fes->GetElementTransformation(i);
//       eltransf->SetIntPoint(&cent);
//       vals.SetSize(vdofs.Size());
//       coeff.Eval(vals, *eltransf, cent);

//       if (fe->GetMapType() == 1 )
//       {
//          vals(i) *= eltransf->Weight();
//       }
//       SetSubVector(vdofs, vals);
//    }
// }

void RBFGridFunction::ProjectCoefficient(std::function<void(const Vector &, Vector &)> F)
{
   int vdim = fes->GetVDim();
   Array<int> vdofs(vdim);
   Vector vals(vdim);

   for (int i = 0; i < basisCenter.Size(); i++)
   {
      for (int j = 0; j < vdim; j++)
      {
         vdofs[j] = i * vdim + j;
      }
      F(*basisCenter[i], vals);
      SetSubVector(vdofs, vals);
   }
}

RBFGridFunction & RBFGridFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetVDim()*basisCenter.Size(),
               "vector size is not equal to number of basis center");
   Vector::operator=(v);
   return *this;
}

RBFGridFunction & RBFGridFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}

} // end of namespace mfem