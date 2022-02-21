#include <iostream>
#include "centgridfunc.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

CentGridFunction::CentGridFunction(FiniteElementSpace *f)
{
   center = dynamic_cast<RBFSpace*>(f)->GetBasisCenter();
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

void CentGridFunction::ProjectCoefficient(VectorCoefficient &coeff)
{
   int vdim = fes->GetVDim();
   Array<int> vdofs(vdim);
   Vector vals(vdim);
   std::function<void(const mfem::Vector &, mfem::Vector &)> Function =
      dynamic_cast<VectorFunctionCoefficient*>(&coeff)->GetFunc();
   for (int i = 0; i < center.Size(); i++)
   {
      for (int j = 0; j < vdim; j++)
      {
         vdofs[j] = i * vdim + j;
      }
      Function(*center[i], vals);
      SetSubVector(vdofs, vals);
   }
}

CentGridFunction & CentGridFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetTrueVSize(), "");
   Vector::operator=(v);
   return *this;
}

CentGridFunction & CentGridFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}

} // end of namespace mfem