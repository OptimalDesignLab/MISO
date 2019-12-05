#include "inviscid_integ.hpp"
#include "sbp_fe.hpp"

using namespace mfem;
using namespace std;

namespace mach
{
// template <typename Derived>
// void InviscidIntegrator<Derived>::flux(const mfem::Vector &dir, const mfem::Vector &u,
//              mfem::Vector &flux_vec)
// {
//    static_cast<Derived*>(this)->calcFlux(dir, u, flux_vec);
// }
// void DyadicFluxIntegrator::AssembleElementGrad(
//    const FiniteElement &el,
//    ElementTransformation &Trans,
//    const Vector &elfun, DenseMatrix &elmat)
// {
//    // This should be in a try/catch, but that creates other issues
//    SBPFiniteElement &sbp = dynamic_cast<SBPFiniteElement&>(el);


// 	elmat = 0.0;
//    for (int i = 0; i < num_nodes; ++i)
//    {
//       Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
//       CalcAdjugate(Trans.Jacobian(), adjJ_i);
//       u.GetColumnReference(i, ui);
//       res.GetColumnReference(i, resi);

// 		for (int j = i+1; j < num_nodes; ++j)
// 		{
//          Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
//          CalcAdjugate(Trans.Jacobian(), adjJ_j);
//          u.GetColumnReference(j, uj);
//          res.GetColumnReference(j, resj);

// 			for (int di = 0; di < dim; ++di)
// 			{
//             stack.new_recording();
//             adept::set_values(ui_ad, num_states, ui.GetData());
//             adept::set_values(uj_ad, num_states, uj.GetData());
// 				flux_fun<adouble>(di, ui_ad, uj_ad, flux_ad);
//             double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
//             // The following is for if we start using Adept arrays
//             // double Sij = sbp.getSkewEntry(di, i, j, dxidx(i,__,__),
//             //                              dxidx(j,__,__));
//             Sij*flux_jac
//             resi.Add(Sij, flux);
//             resj.Add(-Sij, flux);
// 			} // di loop
//       } // j node loop
//    } // i node loop

// }

// template <typename Derived>
// void InviscidBoundaryIntegrator<Derived>::AssembleFaceVector(
//    const FiniteElement &el_bnd,
//    const FiniteElement &el_unused,
//    FaceElementTransformations &trans,
//    const Vector &elfun,
//    Vector &elvect)
// {
//    //cout << "bnd_marker = " << bnd_marker << endl;
//    cout.flush();
//    const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el_bnd);
//    const int num_nodes = el_bnd.GetDof();
//    const int dim = sbp.GetDim();
// #ifdef MFEM_THREAD_SAFE
//    Vector u_face, x, nrm, flux_face;
// #endif
// 	elvect.SetSize(num_states*num_nodes);
//    u_face.SetSize(num_states);
//    x.SetSize(dim);
//    nrm.SetSize(dim);
//    flux_face.SetSize(num_states);
//    elvect.SetSize(num_states*num_nodes);
//    elvect = 0.0;

//    DenseMatrix u(elfun.GetData(), num_nodes, num_states);
//    DenseMatrix res(elvect.GetData(), num_nodes, num_states);

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
//       //cout << "face node " << face_ip.x << ": nrm = " << nrm[0] << ", " << nrm[1] << endl;
//       bnd_flux(x.GetData(), nrm.GetData(), u_face.GetData(),
//                flux_face.GetData());

//       // cout << "face node " << face_ip.x << ": flux = ";
//       // for (int n = 0; n < num_states; ++n)
//       // {
//       //    cout << flux_face[n] << ", ";
//       // }
//       // cout << endl;


//       flux_face *= face_ip.weight;


//       // multiply by test function
//       for (int n = 0; n < num_states; ++n)
//       {
//          res(j, n) += alpha*flux_face(n);
//       }
//    }
//}

} // namespace mach