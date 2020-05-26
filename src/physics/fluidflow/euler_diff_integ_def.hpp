#include "euler_diff_integ.hpp"

template <int dim, bool entvar>
double FarFieldBCDiff<dim, entvar>::GetFaceEnergy(const mfem::FiniteElement &el_bnd,
                                            const mfem::FiniteElement &el_unused,
                                            mfem::FaceElementTransformations &trans,
                                            const mfem::Vector &elfun)
{
//    using namespace mfem;
//    const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
//    const int num_nodes = el_bnd.GetDof();
//    //const int dim = sbp.GetDim();
// #ifdef MFEM_THREAD_SAFE
//    Vector u_face, x, nrm, flux_face;
// #endif
//    num_states = qfs.Size();
//    u_face.SetSize(num_states);
//    x.SetSize(dim);
//    nrm.SetSize(dim);
//    flux_face.SetSize(num_states);
//    //Vector elvect(num_states*num_nodes);
//    //elvect = 0.0;
//     double energy = 0.0;
//    // get element adjoint vector
//    Vector eladj; Array<int> vdofs;
//     adjoint->FESpace()->GetElementVDofs(trans.Elem1no, vdofs);
//     adjoint->GetSubVector(vdofs, eladj);

//    DenseMatrix u(elfun.GetData(), num_nodes, num_states);
//    //DenseMatrix res(elvect.GetData(), num_nodes, num_states);

//    const FiniteElement *sbp_face;
//    switch (dim)
//    {
//       case 1: sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
//               break;
//       case 2: sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
//               break;
//       default: throw mach::MachException(
//          "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
//          "\tcannot handle given dimension");
//    }
//    IntegrationPoint el_ip;
//    for (int i = 0; i < sbp_face->GetDof(); ++i)
//    {
//       const IntegrationPoint &face_ip = sbp_face->GetNodes().IntPoint(i);
//       trans.Loc1.Transform(face_ip, el_ip);
//       trans.Elem1->Transform(el_ip, x);
//       int j = sbp.getIntegrationPointIndex(el_ip);
//       u.GetRow(j, u_face);

//       // get the normal vector and the flux on the face
//       trans.Face->SetIntPoint(&face_ip);
//       CalcOrtho(trans.Face->Jacobian(), nrm);
//       flux(x, nrm, u_face, flux_face); //this function needs to return the derivatives w.r.t mach number
//       flux_face *= face_ip.weight;

//       // multiply by test function, contract with adjoint, and sum contributions
//       for (int n = 0; n < num_states; ++n)
//       {
//          energy += eladj(i*num_states + n)*alpha*flux_face(n);
//       }
//    }

//    return energy;
   mfem::Vector eladj; mfem::Array<int> vdofs;
   adjoint->FESpace()->GetElementVDofs(trans.Elem1No, vdofs);
   adjoint->GetSubVector(vdofs, eladj);
   mfem::Vector elvect(elfun.Size());
   this->AssembleFaceVector(el_bnd, el_unused, trans, elfun, elvect);

   return eladj*elvect;
}

template <int dim, bool entvar>
void FarFieldBCDiff<dim, entvar>::calcFlux(const mfem::Vector &x,
                                          const mfem::Vector &dir,
                                          const mfem::Vector &q,
                                          mfem::Vector &flux_vec)
{
   calcFarFieldMachDiff<double, dim, entvar>(dir.GetData(), qfs.GetData(),
                                         q.GetData(), work_vec.GetData(),
                                         flux_vec.GetData(), mach_fs, aoa_fs);
}

