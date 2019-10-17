#include "forceIntegrator.hpp"
#include "sbp_fe.hpp"

using namespace mfem;
using namespace std;

namespace mach
{


double ForceIntegrator::GetFaceEnergy(
	const FiniteElement &el_bnd,
   const FiniteElement &el_unused,
   FaceElementTransformations &trans,
   const Vector &elfun)
{
   cout.flush();
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   //DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   double functional;
   const FiniteElement *sbp_face;
   switch (dim)
   {
      case 1: sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
              break;
      case 2: sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
              break;
      default: throw mach::MachException(
         "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
         "\tcannot handle given dimension");
   }
   IntegrationPoint el_ip;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(face_ip, el_ip);
      trans.Elem1->Transform(el_ip, x);
      int j = sbp.getIntegrationPointIndex(el_ip);
      u.GetRow(j, u_face);

      // get the normal vector and the flux on the face
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      //cout << "face node " << face_ip.x << ": nrm = " << nrm[0] << ", " << nrm[1] << endl;
      bnd_fun(x.GetData(), nrm.GetData(), u_face.GetData(),flux_face.GetData());
      
      // Todo: angle of attack and orther dimension.
      switch(dim)
      {
         case 1:
         case 2: functional += flux_face[1]*dir[2] + flux_face[2]*dir[3];
                  break;
         default: ;
      }
   }
   return functional;
}


} // namespace mach