#include <iostream>
#include "centgridfunc.hpp"

using namespace std;
using namespace mfem;

namespace mfem
{

CentGridFunction::CentGridFunction(FiniteElementSpace *f)
{
   dim = dynamic_cast<DGDSpace*>(f)->GetMesh()->Dimension();
   basisCenter = dynamic_cast<DGDSpace*>(f)->GetBasisCenter();
   numBasis = basisCenter.Size()/dim;
   SetSize(f->GetVDim() * numBasis);
   fes = f;
   fec = NULL;
   UseDevice(true);
}

CentGridFunction::CentGridFunction(FiniteElementSpace *f, Vector center)
{
   dim = dynamic_cast<DGDSpace*>(f)->GetMesh()->Dimension();
   basisCenter = center;
   numBasis = basisCenter.Size()/dim;
   SetSize(f->GetVDim() * numBasis);
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
   int i,j,k;
   Array<int> vdofs(vdim);
   Vector vals(vdim);
   Vector loc(dim);
   std::function<void(const mfem::Vector &, mfem::Vector &)> F = 
      dynamic_cast<VectorFunctionCoefficient*>(&coeff)->GetFunc();
   for (i = 0; i < numBasis; i++)
   {
      dynamic_cast<DGDSpace*>(FESpace())->GetBasisCenter(i,loc);
      for (j = 0; j < vdim; j++)
      {
         vdofs[j] = i * vdim + j;
      }
      F(loc, vals);
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