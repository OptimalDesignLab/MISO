#ifndef MACH_INVISCID_INTEG_DEF
#define MACH_INVISCID_INTEG_DEF

#include "mfem.hpp"

#include "utils.hpp"
#include "sbp_fe.hpp"
#include "inviscid_integ.hpp"

namespace mach
{
template <typename Derived>
double InviscidIntegrator<Derived>::GetElementEnergy(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();  // not used at present
#ifdef MFEM_THREAD_SAFE
   Vector x_i, ui;
#endif
   x_i.SetSize(dim);
   ui.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);

   double fun = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      trans.Transform(el.GetNodes().IntPoint(i), x_i);
      u.GetRow(i, ui);
      // get node contribution; might need to include mapping Jacobian/adjugate
      fun += volFun(x_i, ui) * trans.Weight() * sbp.getDiagNormEntry(i);
   }
   return fun * alpha;
}

template <typename Derived>
void InviscidIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, fluxi, dxidx;
   DenseMatrix adjJ_i, elflux, elres;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   adjJ_i.SetSize(dim);
   dxidx.SetSize(dim);
   elflux.SetSize(num_states, num_nodes);
   elres.SetSize(num_states, num_nodes);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   elres = 0.0;
   for (int di = 0; di < dim; ++di)
   {
      // get the flux at all the nodes
      for (int i = 0; i < num_nodes; ++i)
      {
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
         CalcAdjugate(Trans.Jacobian(), adjJ_i);
         adjJ_i.GetRow(di, dxidx);
         u.GetRow(i, ui);
         elflux.GetColumnReference(i, fluxi);
         flux(dxidx, ui, fluxi);
      }
      sbp.multWeakOperator(di, elflux, elres, true);
   }
   // This is necessary because data in elvect is expected to be ordered
   // `byNODES`
   res.Transpose(elres);
   res *= alpha;
}

template <typename Derived>
void InviscidIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, dxidx;
   DenseMatrix adjJ_i, flux_jaci;
#endif
   elmat.SetSize(num_states * num_nodes);
   elmat = 0.0;
   ui.SetSize(num_states);
   adjJ_i.SetSize(dim);
   dxidx.SetSize(dim);
   flux_jaci.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   for (int di = 0; di < dim; ++di)
   {
      for (int i = 0; i < num_nodes; ++i)
      {
         // get the flux Jacobian at node i
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
         CalcAdjugate(Trans.Jacobian(), adjJ_i);
         adjJ_i.GetRow(di, dxidx);
         u.GetRow(i, ui);
         fluxJacState(dxidx, ui, flux_jaci);

         // loop over rows j for contribution (Q^T)_{i,j} * Jac_i
         for (int j = 0; j < num_nodes; ++j)
         {
            // get the entry of (Q^T)_{j,i} = Q_{i,j}
            double Q = alpha * sbp.getQ(di, i, j);
            for (int n = 0; n < dim + 2; ++n)
            {
               for (int m = 0; m < dim + 2; ++m)
               {
                  elmat(m * num_nodes + j, n * num_nodes + i) -=
                      Q * flux_jaci(m, n);
               }
            }
         }
      }
   }
}

template <typename Derived>
void DyadicFluxIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uj, fluxij;
   DenseMatrix adjJ_i, adjJ_j;
#endif
   elvect.SetSize(num_states * num_nodes);
   fluxij.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      adjJ_i = Trans.AdjugateJacobian();
      u.GetRow(i, ui);
      for (int j = i + 1; j < num_nodes; ++j)
      {
         Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
         adjJ_j = Trans.AdjugateJacobian();
         u.GetRow(j, uj);
         for (int di = 0; di < dim; ++di)
         {
            // TODO: we should add state_offset to ui and uj, and eqn_offset to
            // fluxij because the flux function may not know about other states
            // and equations.
            flux(di, ui, uj, fluxij);
            double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            Sij *= alpha;
            for (int n = 0; n < num_states; ++n)
            {
               res(i, n) += Sij * fluxij(n);
               res(j, n) -= Sij * fluxij(n);
            }
         }  // di loop
      }     // j node loop
   }        // i node loop
}

template <typename Derived>
void DyadicFluxIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   // This should be in a try/catch, but that creates other issues
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uj;
   DenseMatrix adjJ_i, adjJ_j, flux_jaci, flux_jacj;
#endif
   elmat.SetSize(num_states * num_nodes);
   elmat = 0.0;
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   flux_jaci.SetSize(num_states);
   flux_jacj.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   for (int di = 0; di < dim; ++di)
   {
      for (int i = 0; i < num_nodes; ++i)
      {
         // get the flux Jacobian at node i
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
         adjJ_i = Trans.AdjugateJacobian();
         u.GetRow(i, ui);
         // loop over rows j for contribution (Q^T)_{i,j} * Jac_i
         for (int j = i + 1; j < num_nodes; ++j)
         {
            // get the flux Jacobian at node i
            Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
            adjJ_j = Trans.AdjugateJacobian();
            u.GetRow(j, uj);
            fluxJacStates(di, ui, uj, flux_jaci, flux_jacj);
            double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            Sij *= alpha;
            for (int n = 0; n < num_states; ++n)
            {
               for (int m = 0; m < num_states; ++m)
               {
                  // res(i,n) += Sij*fluxij(n);
                  elmat(n * num_nodes + i, m * num_nodes + i) +=
                      Sij * flux_jaci(n, m);
                  elmat(n * num_nodes + i, m * num_nodes + j) +=
                      Sij * flux_jacj(n, m);
                  // res(j,n) -= Sij*fluxij(n);
                  elmat(n * num_nodes + j, m * num_nodes + i) -=
                      Sij * flux_jaci(n, m);
                  elmat(n * num_nodes + j, m * num_nodes + j) -=
                      Sij * flux_jacj(n, m);
               }
            }
         }
      }
   }
}

template <typename Derived>
void LPSIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui;
   DenseMatrix adjJt, w, Pw;
#endif
   elvect.SetSize(num_states * num_nodes);
   ui.SetSize(num_states);
   adjJt.SetSize(dim);
   w.SetSize(num_states, num_nodes);
   Pw.SetSize(num_states, num_nodes);
   Vector wi;
   Vector Pwi;
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   // Step 1: convert from working variables (this may be the identity)
   for (int i = 0; i < num_nodes; ++i)
   {
      u.GetRow(i, ui);
      w.GetColumnReference(i, wi);
      convert(ui, wi);
   }
   // Step 2: apply the projection operator to w
   sbp.multProjOperator(w, Pw, false);
   // Step 3: apply scaling matrix at each node and diagonal norm
   for (int i = 0; i < num_nodes; ++i)
   {
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      // CalcAdjugateTranspose(Trans.Jacobian(), adjJt);
      CalcAdjugate(Trans.Jacobian(), adjJt);
      u.GetRow(i, ui);
      Pw.GetColumnReference(i, Pwi);
      w.GetColumnReference(i, wi);
      scale(adjJt, ui, Pwi, wi);
      wi *= lps_coeff;
   }
   sbp.multNormMatrix(w, w);
   // Step 4: apply the transposed projection operator to H*A*P*w
   sbp.multProjOperator(w, Pw, true);
   // This is necessary because data in elvect is expected to be ordered
   // `byNODES`
   res.Transpose(Pw);
   res *= alpha;
}

template <typename Derived>
void LPSIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &Trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui;
   DenseMatrix adjJt, w, Pw, jac_term, jac_node, Lij;
#endif
   Vector wi;
   Vector Pwi;
   elmat.SetSize(num_states * num_nodes);
   elmat = 0.0;
   ui.SetSize(num_states);
   adjJt.SetSize(dim);
   w.SetSize(num_states, num_nodes);
   Pw.SetSize(num_states, num_nodes);
   jac_term.SetSize(num_states);
   jac_node.SetSize(num_states);
   Lij.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);

   // convert from working variables (this may be the identity)
   for (int i = 0; i < num_nodes; ++i)
   {
      u.GetRow(i, ui);
      w.GetColumnReference(i, wi);
      convert(ui, wi);
   }
   // apply the projection operator to w
   sbp.multProjOperator(w, Pw, false);

   for (int i = 0; i < num_nodes; ++i)
   {
      // get contribution to Jacobian due to scaling operation
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      CalcAdjugate(Trans.Jacobian(), adjJt);
      u.GetRow(i, ui);
      Pw.GetColumnReference(i, Pwi);
      scaleJacState(adjJt, ui, Pwi, jac_term);
      for (int j = 0; j < num_nodes; ++j)
      {
         double coeff =
             sbp.getDiagNormEntry(i) * sbp.getProjOperatorEntry(i, j);
         for (int n = 0; n < num_states; ++n)
         {
            for (int m = 0; m < num_states; ++m)
            {
               elmat(n * num_nodes + j, m * num_nodes + i) +=
                   coeff * jac_term(n, m);
            }
         }
      }

      // get contribution to Jacobian assuming scaling is constant
      for (int j = i; j < num_nodes; ++j)
      {
         // find matrix entry assuming scaling is constant;
         // Lij = sum_{k=1}^{n} (P_ki H_k A(u_k) P_kj)
         Lij = 0.0;
         for (int k = 0; k < num_nodes; ++k)
         {
            Trans.SetIntPoint(&el.GetNodes().IntPoint(k));
            CalcAdjugate(Trans.Jacobian(), adjJt);
            u.GetRow(k, ui);
            scaleJacV(adjJt, ui, jac_term);
            double coeff = sbp.getProjOperatorEntry(k, i) *
                           sbp.getDiagNormEntry(k) *
                           sbp.getProjOperatorEntry(k, j);
            Lij.Add(coeff, jac_term);
         }
         // insert node-level Jacobian (i,j) into element matrix
         u.GetRow(j, ui);
         convertJacState(ui, jac_term);
         Mult(Lij, jac_term, jac_node);
         for (int n = 0; n < num_states; ++n)
         {
            for (int m = 0; m < num_states; ++m)
            {
               elmat(n * num_nodes + i, m * num_nodes + j) += jac_node(n, m);
            }
         }
         if (i == j)
         {
            continue;  // don't double count the diagonal terms
         }
         // insert node-level Jacobian (j,i) into element matrix
         u.GetRow(i, ui);
         convertJacState(ui, jac_term);
         Mult(Lij, jac_term, jac_node);
         for (int n = 0; n < num_states; ++n)
         {
            for (int m = 0; m < num_states; ++m)
            {
               elmat(n * num_nodes + j, m * num_nodes + i) += jac_node(n, m);
            }
         }
      }
   }
   elmat *= alpha;
}

template <typename Derived>
double InviscidBoundaryIntegrator<Derived>::GetFaceEnergy(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   double fun = 0.0;  // initialize the functional value
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   case 3:
      sbp_face = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
      break;
   default:
      throw mach::MachException(
          "InviscidBoundaryIntegrator::GetFaceEnergy())\n"
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

      // get the normal vector, and then add contribution to function
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      fun += bndryFun(x, nrm, u_face) * face_ip.weight * alpha;
   }
   return fun;
}

template <typename Derived>
void InviscidBoundaryIntegrator<Derived>::AssembleFaceVector(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm, flux_face;
#endif
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   elvect.SetSize(num_states * num_nodes);
   elvect = 0.0;

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   case 3:
      sbp_face = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
      break;
   default:
      throw mach::MachException(
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
      flux(x, nrm, u_face, flux_face);
      flux_face *= face_ip.weight;

      // multiply by test function
      for (int n = 0; n < num_states; ++n)
      {
         res(j, n) += alpha * flux_face(n);
      }
   }
}

template <typename Derived>
void InviscidBoundaryIntegrator<Derived>::AssembleFaceGrad(
    const mfem::FiniteElement &el_bnd,
    const mfem::FiniteElement &el_unused,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_bnd);
   const int num_nodes = el_bnd.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face, x, nrm;  // flux_face;
   DenseMatrix flux_jac_face;
#endif
   // elvect.SetSize(num_states*num_nodes);
   u_face.SetSize(num_states);
   x.SetSize(dim);
   nrm.SetSize(dim);
   // flux_face.SetSize(num_states);
   flux_jac_face.SetSize(num_states);
   elmat.SetSize(num_states * num_nodes);
   elmat = 0.0;

   DenseMatrix u(elfun.GetData(), num_nodes, num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   case 3:
      sbp_face = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
      break;
   default:
      throw mach::MachException(
          "InviscidBoundaryIntegrator::AssembleFaceGrad())\n"
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

      // get the normal vector and the flux Jacobian on the face
      trans.Face->SetIntPoint(&face_ip);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      // flux(x, nrm, u_face, flux_face);
      fluxJacState(x, nrm, u_face, flux_jac_face);

      // flux_face *= face_ip.weight;
      flux_jac_face *= face_ip.weight;

      // multiply by test function
      for (int n = 0; n < num_states; ++n)
      {
         for (int m = 0; m < num_states; ++m)
         {
            // res(j, n) += alpha*flux_face(n);
            elmat(m * num_nodes + j, n * num_nodes + j) +=
                alpha * flux_jac_face(m, n);
         }
      }
   }
}

template <typename Derived>
double InviscidFaceIntegrator<Derived>::GetFaceEnergy(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_left);
   const int num_nodes_left = el_left.GetDof();
   const int num_nodes_right = el_right.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face_left, u_face_right, nrm;
#endif
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   nrm.SetSize(dim);
   DenseMatrix u_left(elfun.GetData(), num_nodes_left, num_states);
   DenseMatrix u_right(elfun.GetData() + num_nodes_left * num_states,
                       num_nodes_right,
                       num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   case 3:
      sbp_face = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
      break;
   default:
      throw mach::MachException(
          "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
          "\tcannot handle given dimension");
   }
   IntegrationPoint ip_left;
   IntegrationPoint ip_right;
   double fun = 0.0;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      const IntegrationPoint &ip_face = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(ip_face, ip_left);
      trans.Loc2.Transform(ip_face, ip_right);

      int i_left = sbp.getIntegrationPointIndex(ip_left);
      u_left.GetRow(i_left, u_face_left);
      int i_right = sbp.getIntegrationPointIndex(ip_right);
      u_right.GetRow(i_right, u_face_right);

      // get the contribution to the function on the face
      trans.Face->SetIntPoint(&ip_face);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      nrm *= ip_face.weight;
      fun += iFaceFun(nrm, u_face_left, u_face_right);
   }
   return fun;
}

template <typename Derived>
void InviscidFaceIntegrator<Derived>::AssembleFaceVector(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_left);
   const int num_nodes_left = el_left.GetDof();
   const int num_nodes_right = el_right.GetDof();
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face_left, u_face_right, nrm, flux_face;
#endif
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   nrm.SetSize(dim);
   flux_face.SetSize(num_states);
   elvect.SetSize(num_states * (num_nodes_left + num_nodes_right));
   elvect = 0.0;

   DenseMatrix u_left(elfun.GetData(), num_nodes_left, num_states);
   DenseMatrix u_right(elfun.GetData() + num_nodes_left * num_states,
                       num_nodes_right,
                       num_states);
   DenseMatrix res_left(elvect.GetData(), num_nodes_left, num_states);
   DenseMatrix res_right(elvect.GetData() + num_nodes_left * num_states,
                         num_nodes_right,
                         num_states);

   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   case 3:
      sbp_face = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
   default:
      throw mach::MachException(
          "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
          "\tcannot handle given dimension");
   }
   IntegrationPoint ip_left;
   IntegrationPoint ip_right;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      const IntegrationPoint &ip_face = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(ip_face, ip_left);
      trans.Loc2.Transform(ip_face, ip_right);

      int i_left = sbp.getIntegrationPointIndex(ip_left);
      u_left.GetRow(i_left, u_face_left);
      int i_right = sbp.getIntegrationPointIndex(ip_right);
      u_right.GetRow(i_right, u_face_right);

      // get the normal vector and the flux on the face
      trans.Face->SetIntPoint(&ip_face);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      nrm *= ip_face.weight;
      flux(nrm, u_face_left, u_face_right, flux_face);

      // multiply by test functions from left and right elements
      for (int n = 0; n < num_states; ++n)
      {
         res_left(i_left, n) += alpha * flux_face(n);
         res_right(i_right, n) -= alpha * flux_face(n);
      }
   }
}

template <typename Derived>
void InviscidFaceIntegrator<Derived>::AssembleFaceGrad(
    const mfem::FiniteElement &el_left,
    const mfem::FiniteElement &el_right,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el_left);
   const int num_nodes_left = el_left.GetDof();
   const int num_nodes_right = el_right.GetDof();
   const int num_rows = num_states * (num_nodes_left + num_nodes_right);
   const int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_face_left, u_face_right, nrm;
   DenseMatrix flux_jac_left, flux_jac_right;
#endif
   elmat.SetSize(num_rows);
   elmat = 0.0;
   u_face_left.SetSize(num_states);
   u_face_right.SetSize(num_states);
   nrm.SetSize(dim);
   flux_jac_left.SetSize(num_states);
   flux_jac_right.SetSize(num_states);

   DenseMatrix u_left(elfun.GetData(), num_nodes_left, num_states);
   DenseMatrix u_right(elfun.GetData() + num_nodes_left * num_states,
                       num_nodes_right,
                       num_states);
   const FiniteElement *sbp_face = nullptr;
   switch (dim)
   {
   case 1:
      sbp_face = fec->FiniteElementForGeometry(Geometry::POINT);
      break;
   case 2:
      sbp_face = fec->FiniteElementForGeometry(Geometry::SEGMENT);
      break;
   case 3:
      sbp_face = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
   default:
      throw mach::MachException(
          "InviscidBoundaryIntegrator::AssembleFaceVector())\n"
          "\tcannot handle given dimension");
   }
   IntegrationPoint ip_left;
   IntegrationPoint ip_right;
   for (int i = 0; i < sbp_face->GetDof(); ++i)
   {
      const IntegrationPoint &ip_face = sbp_face->GetNodes().IntPoint(i);
      trans.Loc1.Transform(ip_face, ip_left);
      trans.Loc2.Transform(ip_face, ip_right);

      int i_left = sbp.getIntegrationPointIndex(ip_left);
      u_left.GetRow(i_left, u_face_left);
      int i_right = sbp.getIntegrationPointIndex(ip_right);
      u_right.GetRow(i_right, u_face_right);

      // get the normal vector and the flux Jacobians on the face
      trans.Face->SetIntPoint(&ip_face);
      CalcOrtho(trans.Face->Jacobian(), nrm);
      nrm *= alpha * ip_face.weight;
      // flux(nrm, u_face_left, u_face_right, flux_face);
      fluxJacStates(
          nrm, u_face_left, u_face_right, flux_jac_left, flux_jac_right);

      // insert flux Jacobians into element stiffness matrices
      const int offset = num_states * num_nodes_left;
      for (int n = 0; n < num_states; ++n)
      {
         for (int m = 0; m < num_states; ++m)
         {
            // res_left(i_left, n) += alpha*flux_face(n);
            elmat(n * num_nodes_left + i_left, m * num_nodes_left + i_left) +=
                flux_jac_left(n, m);
            elmat(n * num_nodes_left + i_left,
                  offset + m * num_nodes_right + i_right) +=
                flux_jac_right(n, m);
            // res_right(i_right, n) -= alpha*flux_face(n);
            elmat(offset + n * num_nodes_right + i_right,
                  m * num_nodes_left + i_left) -= flux_jac_left(n, m);
            elmat(offset + n * num_nodes_right + i_right,
                  offset + m * num_nodes_right + i_right) -=
                flux_jac_right(n, m);
         }
      }
   }
}

template <typename Derived>
void NonlinearMassIntegrator<Derived>::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   // const IntegrationRule &ir = sbp.GetNodes();
   int num_nodes = sbp.GetDof();
   // int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_i, q_i;
#endif
   elvect.SetSize(num_states * num_nodes);
   u_i.SetSize(num_states);
   q_i.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      const IntegrationPoint &ip = el.GetNodes().IntPoint(i);
      trans.SetIntPoint(&ip);
      double weight = trans.Weight() * ip.weight;
      u.GetRow(i, u_i);
      convert(u_i, q_i);
      for (int n = 0; n < num_states; ++n)
      {
         res(i, n) += weight * q_i(n);
      }
   }
   res *= alpha;
}

template <typename Derived>
void NonlinearMassIntegrator<Derived>::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   using namespace mfem;
   const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
   // const IntegrationRule &ir = sbp.GetNodes();
   int num_nodes = sbp.GetDof();
   // int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector u_i;
   DenseMatrix A_i;
#endif
   elmat.SetSize(num_states * num_nodes);
   u_i.SetSize(num_states);
   A_i.SetSize(num_states);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   elmat = 0.0;
   // loop over the SBP nodes/integration points
   for (int i = 0; i < num_nodes; ++i)
   {
      const IntegrationPoint &ip = el.GetNodes().IntPoint(i);
      trans.SetIntPoint(&ip);
      double weight = trans.Weight() * ip.weight;
      u.GetRow(i, u_i);
      convertJacState(u_i, A_i);
      for (int n = 0; n < num_states; ++n)
      {
         for (int m = 0; m < num_states; ++m)
         {
            elmat(n * num_nodes + i, m * num_nodes + i) += weight * A_i(n, m);
         }
      }
   }
   elmat *= alpha;
}

}  // namespace mach

#endif
