#include "mfem.hpp"

#include "mach_input.hpp"

#include "thermal_integ.hpp"
#include "utils.hpp"

namespace mach
{
void TestBCIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);
   elvect.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = el1.GetOrder() + el2.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double trans_weight = trans.Weight();
      const double w = ip.weight * trans_weight;

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      const double val = elfun * shape * w;

      add(elvect, val, shape, elvect);
   }
}

void TestBCIntegratorMeshRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &mesh_trans,
    mfem::Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   auto &trans =
       *state.FESpace()->GetMesh()->GetBdrFaceTransformations(element);
   const int state_elem_num = trans.Elem1->ElementNo;
   const auto &el1 = *state.FESpace()->GetFE(state_elem_num);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el1.GetDof();
   const int space_dim = trans.GetSpaceDim();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector psi;
#endif
   auto *dof_tr = state.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix shape;
   DenseMatrix shape_bar;
   DenseMatrix PointMat_bar;
#else
   auto &shape = integ.shape;
#endif

   shape.SetSize(ndof);
   shape_bar.SetSize(ndof);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   const mfem::IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el1.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      // const double trans_weight = trans.Weight();
      // const double w = ip.weight * trans_weight;

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      double elfun_shape = elfun * shape;
      double psi_shape = psi * shape;

      // add(elvect, elfun_shape * w, shape, elvect);

      /// dummy functional
      // fun += psi_shape * elfun_shape * w;

      double fun_bar = 1.0;

      /// fun += psi_shape * elfun_shape * w;
      // double psi_shape_bar = fun_bar * elfun_shape * w;
      // double elfun_shape_bar = fun_bar * psi_shape * w;
      double w_bar = fun_bar * psi_shape * elfun_shape;

      /// const double w = ip.weight * trans_weight;
      double trans_weight_bar = w_bar * ip.weight;

      /// const double trans_weight = trans.Weight();
      PointMat_bar = 0.0;
      trans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

void L2ProjectionIntegrator::AssembleElementVector(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);
   }

   elvect.SetSize(ndof);
   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      double w = alpha * ip.weight * trans_weight;

      el.CalcShape(ip, shape);

      double state = elfun * shape;
      double desired = g.Eval(trans, ip);

      elvect.Add(w * (state - desired), shape);
   }
}

void L2ProjectionIntegrator::AssembleElementGrad(
    const mfem::FiniteElement &el,
    mfem::ElementTransformation &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   int ndof = el.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);
   }

   elmat.SetSize(ndof);
   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      double w = alpha * ip.weight * trans_weight;

      el.CalcShape(ip, shape);

      AddMult_a_VVt(w, shape, elmat);
   }
}

void L2ProjectionIntegratorMeshRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &trans,
    mfem::Vector &mesh_coords_bar)
{
   const auto &el = *state.FESpace()->GetFE(trans.ElementNo);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el.GetDof();
   const int space_dim = trans.GetSpaceDim();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector psi;
#endif

   auto *dof_tr = state.FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.ElementNo, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
   mfem::DenseMatrix PointMat_bar;
#else
   auto &shape = integ.shape;
#endif
   shape.SetSize(ndof);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   // cast the ElementTransformation
   auto &isotrans = dynamic_cast<mfem::IsoparametricTransformation &>(trans);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(el.GetGeomType(), order);
   }

   auto &alpha = integ.alpha;
   auto &g = integ.g;

   mesh_coords_bar.SetSize(space_dim * mesh_ndof);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      double trans_weight = trans.Weight();

      double w = alpha * ip.weight * trans_weight;

      el.CalcShape(ip, shape);

      double state = elfun * shape;
      double desired = g.Eval(trans, ip);

      double psi_shape = psi * shape;

      // elvect.Add(w * (state - desired), shape);
      // double fun += w * (state - desired) * psi_shape;

      double fun_bar = 1.0;

      /// fun += w * (state - desired) * psi_shape;
      double w_bar = fun_bar * (state - desired) * psi_shape;
      // double state_bar = fun_bar * w * psi_shape;
      double desired_bar = -fun_bar * w * psi_shape;
      // double psi_shape_bar = fun_bar * w * (state - desired);

      /// double psi_shape = psi * shape;

      /// double desired = g.Eval(trans, ip);
      PointMat_bar = 0.0;
      g.EvalRevDiff(desired_bar, trans, ip, PointMat_bar);

      /// double state = elfun * shape;
      /// el.CalcShape(ip, shape);

      /// double w = alpha * ip.weight * trans_weight;
      double trans_weight_bar = w_bar * alpha * ip.weight;

      /// double trans_weight = trans.Weight();
      isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

void setInputs(ThermalContactResistanceIntegrator &integ,
               const MachInputs &inputs)
{
   if (!integ.name.empty())
   {
      setValueFromInputs(inputs, "h_c:" + integ.name, integ.h);
   }
   else
   {
      setValueFromInputs(inputs, "h_c", integ.h);
   }
}

void ThermalContactResistanceIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof1 = el1.GetDof();
   int ndof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;
#endif

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   mfem::Vector elfun1(elfun.GetData(), ndof1);
   mfem::Vector elfun2(elfun.GetData() + ndof1, ndof2);

   elvect.SetSize(ndof1 + ndof2);
   mfem::Vector elvect1(elvect.GetData(), ndof1);
   mfem::Vector elvect2(elvect.GetData() + ndof1, ndof2);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      const double temp1 = elfun1 * shape1;
      const double temp2 = elfun2 * shape2;

      const double val = h * (temp1 - temp2);

      elvect1.Add(w * val, shape1);
      elvect2.Add(-w * val, shape2);
   }
}

void ThermalContactResistanceIntegrator::AssembleFaceGrad(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   int ndof1 = el1.GetDof();
   int ndof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;

   mfem::DenseMatrix elmat11;
   mfem::DenseMatrix elmat12;
   mfem::DenseMatrix elmat22;
#endif

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   mfem::Vector elfun1(elfun.GetData(), ndof1);
   mfem::Vector elfun2(elfun.GetData() + ndof1, ndof2);

   elmat.SetSize(ndof1 + ndof2);

   elmat11.SetSize(ndof1);
   elmat12.SetSize(ndof1, ndof2);
   elmat22.SetSize(ndof2);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      // const double temp1 = elfun1 * shape1;
      // const double temp2 = elfun2 * shape2;

      // const double val = h * (temp1 - temp2);

      /// elvect1.Add(w * val, shape1);
      /// elvect2.Add(-w * val, shape2);

      MultVVt(shape1, elmat11);
      elmat11 *= w * h;
      MultVVt(shape2, elmat22);
      elmat22 *= w * h;

      // elmat12;
      MultVWt(shape1, shape2, elmat12);
      elmat12 *= w * h;

      for (int j = 0; j < ndof1; ++j)
      {
         for (int k = 0; k < ndof1; ++k)
         {
            elmat(j, k) += elmat11(j, k);
         }
      }

      for (int j = 0; j < ndof1; ++j)
      {
         for (int k = 0; k < ndof2; ++k)
         {
            elmat(j, k + ndof1) += -elmat12(j, k);
            elmat(k + ndof1, j) += -elmat12(j, k);
         }
      }

      for (int j = 0; j < ndof2; ++j)
      {
         for (int k = 0; k < ndof2; ++k)
         {
            elmat(j + ndof1, k + ndof1) += elmat22(j, k);
         }
      }
   }
}

void ThermalContactResistanceIntegratorMeshRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el1,
    const mfem::FiniteElement &mesh_el2,
    mfem::FaceElementTransformations &trans,
    mfem::Vector &mesh_coords_bar)
{
   const auto &el1 = *state.FESpace()->GetFE(trans.Elem1->ElementNo);
   const auto &el2 = *state.FESpace()->GetFE(trans.Elem2->ElementNo);

   const int mesh_ndof1 = mesh_el1.GetDof();
   const int mesh_ndof2 = mesh_el2.GetDof();
   const int mesh_face_ndof = trans.GetFE()->GetDof();

   const int ndof1 = el1.GetDof();
   const int ndof2 = el2.GetDof();

   const int space_dim = trans.GetSpaceDim();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs1;
   mfem::Array<int> vdofs2;
   mfem::Vector elfun1;
   mfem::Vector elfun2;
   mfem::Vector psi1;
   mfem::Vector psi2;
#endif
   auto *dof_tr =
       state.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   state.GetSubVector(vdofs1, elfun1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun1);
   }
   dof_tr = state.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   state.GetSubVector(vdofs2, elfun2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun2);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   adjoint.GetSubVector(vdofs1, psi1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi1);
   }
   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   adjoint.GetSubVector(vdofs2, psi2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi2);
   }

   dof_tr = mesh_fes.GetElementVDofs(trans.Elem1->ElementNo, vdofs1);

   if (trans.ElementType == mfem::ElementTransformation::FACE)
   {
      mesh_fes.GetFaceVDofs(trans.ElementNo, vdofs2);
   }
   else
   {
      mesh_fes.GetBdrElementVDofs(trans.ElementNo, vdofs2);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;

   // mfem::Vector sipg_elvect;
   mfem::DenseMatrix PointMatFace_bar;
   mfem::Vector mesh_coords_face_bar;
#else
   auto &shape1 = integ.shape1;
   auto &shape2 = integ.shape2;
   // auto &sipg_elvect = integ.shape2;
#endif
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   // sipg_elvect.SetSize(ndof1 + ndof2);

   PointMatFace_bar.SetSize(space_dim, mesh_face_ndof);
   mesh_coords_face_bar.SetSize(space_dim * mesh_face_ndof);

   const auto *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   auto &h = integ.h;

   mesh_coords_bar.SetSize(space_dim * (mesh_ndof1 + mesh_ndof2));
   mfem::Vector mesh_coords_bar1(mesh_coords_bar.GetData(),
                                 space_dim * mesh_ndof1);
   mfem::Vector mesh_coords_bar2(
       mesh_coords_bar.GetData() + space_dim * mesh_ndof1,
       space_dim * mesh_ndof2);
   mesh_coords_bar = 0.0;
   mesh_coords_face_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      // const double face_weight = trans.Weight();

      // double w = alpha * ip.weight * face_weight;

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      double psi1_shape1 = psi1 * shape1;
      double psi2_shape2 = psi2 * shape2;

      const double temp1 = elfun1 * shape1;
      const double temp2 = elfun2 * shape2;

      const double val = h * (temp1 - temp2);

      // elvect1.Add(w * val, shape1);
      // double term1 = w * val * psi1_shape1;

      // elvect2.Add(-w * val, shape2);
      // double term2 = -w * val * psi2_shape2;

      /// dummy functional for adjoint-weighted residual
      /// fun += term1 + term2;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += term1 + term2;
      const double term1_bar = fun_bar;
      const double term2_bar = fun_bar;

      /// double term2 = -w * val * psi2_shape2;
      double w_bar = term2_bar * -val * psi2_shape2;

      /// double term1 = w * val * psi1_shape1;
      w_bar += term1_bar * val * psi1_shape1;

      /// double w = alpha * ip.weight * face_weight;
      double face_weight_bar = w_bar * alpha * ip.weight;

      /// const double face_weight = trans.Weight();
      PointMatFace_bar = 0.0;
      trans.WeightRevDiff(face_weight_bar, PointMatFace_bar);

      // code to insert PointMatFace_bar into mesh_coords_face_bar
      for (int j = 0; j < mesh_face_ndof; ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            mesh_coords_face_bar(k * mesh_face_ndof + j) +=
                PointMatFace_bar(k, j);
         }
      }
   }

   // code to insert mesh_coords_face_bar into mesh_coords_bar
   for (int j = 0; j < vdofs2.Size(); ++j)
   {
      auto idx = vdofs1.Find(vdofs2[j]);
      if (idx == -1)
      {
         continue;
      }
      else
      {
         mesh_coords_bar1(idx) += mesh_coords_face_bar(j);
      }
   }
}

double ThermalContactResistanceIntegratorHRevSens::GetFaceEnergy(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   const int ndof1 = el1.GetDof();
   const int ndof2 = el2.GetDof();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs1;
   mfem::Array<int> vdofs2;
   mfem::Vector elfun1;
   mfem::Vector elfun2;
   mfem::Vector psi1;
   mfem::Vector psi2;
#endif
   auto *dof_tr =
       state.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   state.GetSubVector(vdofs1, elfun1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun1);
   }
   dof_tr = state.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   state.GetSubVector(vdofs2, elfun2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun2);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   adjoint.GetSubVector(vdofs1, psi1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi1);
   }
   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   adjoint.GetSubVector(vdofs2, psi2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi2);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;
#else
   auto &shape1 = integ.shape1;
   auto &shape2 = integ.shape2;
#endif
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const auto *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder()) + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   // auto &h = integ.h;
   // auto &theta_f = integ.theta_f;

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      const double psi1_shape1 = psi1 * shape1;
      const double psi2_shape2 = psi2 * shape2;

      const double temp1 = elfun1 * shape1;
      const double temp2 = elfun2 * shape2;

      // const double val = h * (temp1 - temp2);

      // elvect1.Add(w * val, shape1);
      // double term1 = w * val * psi1_shape1;

      // elvect2.Add(-w * val, shape2);
      // double term2 = -w * val * psi2_shape2;

      /// dummy functional for adjoint-weighted residual
      /// fun += term1 + term2;

      /// start reverse pass
      const double fun_bar = 1.0;

      /// fun += term1 + term2;
      const double term1_bar = fun_bar;
      const double term2_bar = fun_bar;

      /// double term2 = -w * val * psi2_shape2;
      double val_bar = term2_bar * -w * psi2_shape2;

      /// double term1 = w * val * psi1_shape1;
      val_bar += term1_bar * w * psi1_shape1;

      /// const double val = h * (temp1 - temp2);
      const double h_bar = val_bar * (temp1 - temp2);

      fun += h_bar;
   }
   return fun;
}

void setInputs(InternalConvectionInterfaceIntegrator &integ,
               const MachInputs &inputs)
{
   if (!integ.name.empty())
   {
      setValueFromInputs(inputs, "h_c:" + integ.name, integ.h);
      setValueFromInputs(inputs, "fluid_temp:" + integ.name, integ.theta_f);
   }
   else
   {
      setValueFromInputs(inputs, "h_c", integ.h);
      setValueFromInputs(inputs, "fluid_temp", integ.theta_f);
   }
}

void InternalConvectionInterfaceIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof1 = el1.GetDof();
   int ndof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;
#endif

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   mfem::Vector elfun1(elfun.GetData(), ndof1);
   mfem::Vector elfun2(elfun.GetData() + ndof1, ndof2);

   elvect.SetSize(ndof1 + ndof2);
   mfem::Vector elvect1(elvect.GetData(), ndof1);
   mfem::Vector elvect2(elvect.GetData() + ndof1, ndof2);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder()) + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      const double temp1 = elfun1 * shape1;
      const double temp2 = elfun2 * shape2;

      const double val1 = h * (temp1 - theta_f);
      const double val2 = h * (temp2 - theta_f);

      elvect1.Add(w * val1, shape1);
      elvect2.Add(w * val2, shape2);
   }
}

void InternalConvectionInterfaceIntegrator::AssembleFaceGrad(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   int ndof1 = el1.GetDof();
   int ndof2 = el2.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;

   mfem::DenseMatrix elmat11;
   mfem::DenseMatrix elmat22;
#endif

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   mfem::Vector elfun1(elfun.GetData(), ndof1);
   mfem::Vector elfun2(elfun.GetData() + ndof1, ndof2);

   elmat.SetSize(ndof1 + ndof2);

   elmat11.SetSize(ndof1);
   elmat22.SetSize(ndof2);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      // const double temp1 = elfun1 * shape1;
      // const double temp2 = elfun2 * shape2;

      // const double val1 = h * (temp1 - theta_f);
      // const double val2 = h * (temp2 - theta_f);

      // elvect1.Add(w * val1, shape1);
      // elvect2.Add(w * val2, shape2);

      MultVVt(shape1, elmat11);
      elmat11 *= w * h;

      MultVVt(shape2, elmat22);
      elmat22 *= w * h;

      for (int j = 0; j < ndof1; ++j)
      {
         for (int k = 0; k < ndof1; ++k)
         {
            elmat(j, k) += elmat11(j, k);
         }
      }

      for (int j = 0; j < ndof2; ++j)
      {
         for (int k = 0; k < ndof2; ++k)
         {
            elmat(j + ndof1, k + ndof1) += elmat22(j, k);
         }
      }
   }
}

void InternalConvectionInterfaceIntegratorMeshRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el1,
    const mfem::FiniteElement &mesh_el2,
    mfem::FaceElementTransformations &trans,
    mfem::Vector &mesh_coords_bar)
{
   const auto &el1 = *state.FESpace()->GetFE(trans.Elem1->ElementNo);
   const auto &el2 = *state.FESpace()->GetFE(trans.Elem2->ElementNo);

   const int mesh_ndof1 = mesh_el1.GetDof();
   const int mesh_ndof2 = mesh_el2.GetDof();
   const int mesh_face_ndof = trans.GetFE()->GetDof();

   const int ndof1 = el1.GetDof();
   const int ndof2 = el2.GetDof();

   const int space_dim = trans.GetSpaceDim();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs1;
   mfem::Array<int> vdofs2;
   mfem::Vector elfun1;
   mfem::Vector elfun2;
   mfem::Vector psi1;
   mfem::Vector psi2;
#endif
   auto *dof_tr =
       state.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   state.GetSubVector(vdofs1, elfun1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun1);
   }
   dof_tr = state.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   state.GetSubVector(vdofs2, elfun2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun2);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   adjoint.GetSubVector(vdofs1, psi1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi1);
   }
   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   adjoint.GetSubVector(vdofs2, psi2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi2);
   }

   dof_tr = mesh_fes.GetElementVDofs(trans.Elem1->ElementNo, vdofs1);

   if (trans.ElementType == mfem::ElementTransformation::FACE)
   {
      mesh_fes.GetFaceVDofs(trans.ElementNo, vdofs2);
   }
   else
   {
      mesh_fes.GetBdrElementVDofs(trans.ElementNo, vdofs2);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;

   mfem::DenseMatrix PointMatFace_bar;
   mfem::Vector mesh_coords_face_bar;
#else
   auto &shape1 = integ.shape1;
   auto &shape2 = integ.shape2;
#endif
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   PointMatFace_bar.SetSize(space_dim, mesh_face_ndof);
   mesh_coords_face_bar.SetSize(space_dim * mesh_face_ndof);

   const auto *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder());
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   auto &h = integ.h;
   auto &theta_f = integ.theta_f;

   mesh_coords_bar.SetSize(space_dim * (mesh_ndof1 + mesh_ndof2));
   mfem::Vector mesh_coords_bar1(mesh_coords_bar.GetData(),
                                 space_dim * mesh_ndof1);
   mfem::Vector mesh_coords_bar2(
       mesh_coords_bar.GetData() + space_dim * mesh_ndof1,
       space_dim * mesh_ndof2);
   mesh_coords_bar = 0.0;
   mesh_coords_face_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      // const double face_weight = trans.Weight();

      // double w = alpha * ip.weight * face_weight;

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      double psi1_shape1 = psi1 * shape1;
      double psi2_shape2 = psi2 * shape2;

      const double temp1 = elfun1 * shape1;
      const double temp2 = elfun2 * shape2;

      const double val1 = h * (temp1 - theta_f);
      const double val2 = h * (temp2 - theta_f);

      // elvect1.Add(w * val1, shape1);
      // double term1 = w * val1 * psi1_shape1;

      // elvect2.Add(w * val2, shape2);
      // double term2 = w * val2 * psi2_shape2;

      /// dummy functional for adjoint-weighted residual
      /// fun += term1 + term2;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += term1 + term2;
      const double term1_bar = fun_bar;
      const double term2_bar = fun_bar;

      /// double term2 = w * val2 * psi2_shape2;
      double w_bar = term2_bar * val2 * psi2_shape2;

      /// double term1 = w * val1 * psi1_shape1;
      w_bar += term1_bar * val1 * psi1_shape1;

      /// double w = alpha * ip.weight * face_weight;
      double face_weight_bar = w_bar * alpha * ip.weight;

      /// const double face_weight = trans.Weight();
      PointMatFace_bar = 0.0;
      trans.WeightRevDiff(face_weight_bar, PointMatFace_bar);

      // code to insert PointMatFace_bar into mesh_coords_face_bar
      for (int j = 0; j < mesh_face_ndof; ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            mesh_coords_face_bar(k * mesh_face_ndof + j) +=
                PointMatFace_bar(k, j);
         }
      }
   }

   // code to insert mesh_coords_face_bar into mesh_coords_bar
   for (int j = 0; j < vdofs2.Size(); ++j)
   {
      auto idx = vdofs1.Find(vdofs2[j]);
      if (idx == -1)
      {
         continue;
      }
      else
      {
         mesh_coords_bar1(idx) += mesh_coords_face_bar(j);
      }
   }
}

double InternalConvectionInterfaceIntegratorHRevSens::GetFaceEnergy(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   const int ndof1 = el1.GetDof();
   const int ndof2 = el2.GetDof();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs1;
   mfem::Array<int> vdofs2;
   mfem::Vector elfun1;
   mfem::Vector elfun2;
   mfem::Vector psi1;
   mfem::Vector psi2;
#endif
   auto *dof_tr =
       state.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   state.GetSubVector(vdofs1, elfun1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun1);
   }
   dof_tr = state.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   state.GetSubVector(vdofs2, elfun2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun2);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   adjoint.GetSubVector(vdofs1, psi1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi1);
   }
   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   adjoint.GetSubVector(vdofs2, psi2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi2);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;
#else
   auto &shape1 = integ.shape1;
   auto &shape2 = integ.shape2;
#endif
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const auto *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder()) + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   // auto &h = integ.h;
   auto &theta_f = integ.theta_f;

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      const double psi1_shape1 = psi1 * shape1;
      const double psi2_shape2 = psi2 * shape2;

      const double temp1 = elfun1 * shape1;
      const double temp2 = elfun2 * shape2;

      // const double val1 = h * (temp1 - theta_f);
      // const double val2 = h * (temp2 - theta_f);

      // elvect1.Add(w * val1, shape1);
      // double term1 = w * va1 * psi1_shape1;

      // elvect2.Add(w * val2, shape2);
      // double term2 = w * val2 * psi2_shape2;

      /// dummy functional for adjoint-weighted residual
      /// fun += term1 + term2;

      /// start reverse pass
      const double fun_bar = 1.0;

      /// fun += term1 + term2;
      const double term1_bar = fun_bar;
      const double term2_bar = fun_bar;

      /// double term2 = w * val2 * psi2_shape2;
      const double val2_bar = term2_bar * w * psi2_shape2;

      /// double term1 = w * val1 * psi1_shape1;
      const double val1_bar = term1_bar * w * psi1_shape1;

      /// const double val2 = h * (temp2 - theta_f);
      double h_bar = val2_bar * (temp2 - theta_f);

      /// const double val1 = h * (temp1 - theta_f);
      h_bar += val1_bar * (temp1 - theta_f);

      fun += h_bar;
   }
   return fun;
}

double InternalConvectionInterfaceIntegratorFluidTempRevSens::GetFaceEnergy(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   const int ndof1 = el1.GetDof();
   const int ndof2 = el2.GetDof();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs1;
   mfem::Array<int> vdofs2;
   mfem::Vector psi1;
   mfem::Vector psi2;
#endif

   auto *dof_tr =
       adjoint.FESpace()->GetElementVDofs(trans.Elem1->ElementNo, vdofs1);
   adjoint.GetSubVector(vdofs1, psi1);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi1);
   }
   dof_tr = adjoint.FESpace()->GetElementVDofs(trans.Elem2->ElementNo, vdofs2);
   adjoint.GetSubVector(vdofs2, psi2);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi2);
   }

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape1;
   mfem::Vector shape2;
#else
   auto &shape1 = integ.shape1;
   auto &shape2 = integ.shape2;
#endif
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const auto *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * std::max(el1.GetOrder(), el2.GetOrder()) + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   auto &h = integ.h;
   // auto &theta_f = integ.theta_f;

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Weight();

      // Access the neighboring element's integration point
      const auto &eip1 = trans.GetElement1IntPoint();
      const auto &eip2 = trans.GetElement2IntPoint();

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      const double psi1_shape1 = psi1 * shape1;
      const double psi2_shape2 = psi2 * shape2;

      // const double temp1 = elfun1 * shape1;
      // const double temp2 = elfun2 * shape2;

      // const double val1 = h * (temp1 - theta_f);
      // const double val2 = h * (temp2 - theta_f);

      // elvect1.Add(w * val1, shape1);
      // double term1 = w * va1 * psi1_shape1;

      // elvect2.Add(w * val2, shape2);
      // double term2 = w * val2 * psi2_shape2;

      /// dummy functional for adjoint-weighted residual
      /// fun += term1 + term2;

      /// start reverse pass
      const double fun_bar = 1.0;

      /// fun += term1 + term2;
      const double term1_bar = fun_bar;
      const double term2_bar = fun_bar;

      /// double term2 = w * val2 * psi2_shape2;
      const double val2_bar = term2_bar * w * psi2_shape2;

      /// double term1 = w * val1 * psi1_shape1;
      const double val1_bar = term1_bar * w * psi1_shape1;

      /// const double val2 = h * (temp2 - theta_f);
      double theta_f_bar = -val2_bar * h;

      /// const double val1 = h * (temp1 - theta_f);
      theta_f_bar -= val1_bar * h;

      fun += theta_f_bar;
   }
   return fun;
}

void ConvectionBCIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);
   elvect.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = el1.GetOrder() + el2.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      const double val = h * ((elfun * shape) - theta_f);

      add(elvect, w * val, shape, elvect);
   }
}

void ConvectionBCIntegrator::AssembleFaceGrad(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   int ndof = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);
   elmat.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = el1.GetOrder() + el2.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      AddMult_a_VVt(w * h, shape, elmat);
   }
}

void ConvectionBCIntegratorMeshRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &mesh_trans,
    mfem::Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   auto &trans =
       *state.FESpace()->GetMesh()->GetBdrFaceTransformations(element);
   const int state_elem_num = trans.Elem1->ElementNo;
   const auto &el1 = *state.FESpace()->GetFE(state_elem_num);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el1.GetDof();
   const int space_dim = trans.GetSpaceDim();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector psi;
#endif
   auto *dof_tr = state.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix shape;
   DenseMatrix PointMat_bar;
#else
   auto &shape = integ.shape;
#endif

   shape.SetSize(ndof);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   const mfem::IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el1.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   auto &h = integ.h;
   auto &theta_f = integ.theta_f;

   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      // const double trans_weight = trans.Weight();
      // const double w = alpha * ip.weight * trans_weight;

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      const double flux = h * ((elfun * shape) - theta_f);
      const double adjoint = psi * shape;

      /// dummy functional for adjoint-weighted residual
      // fun += adjoint * flux * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += adjoint * flux * w;
      // double adjoint_bar = fun_bar * flux * w;
      // double flux_bar = fun_bar * adjoint * w;
      double w_bar = fun_bar * adjoint * flux;

      /// const double adjoint = psi * shape;
      /// const double flux = h * ((elfun * shape) - theta_f);

      /// const double w = alpha * ip.weight * trans_weight;
      const double trans_weight_bar = w_bar * alpha * ip.weight;

      PointMat_bar = 0.0;
      trans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

double ConvectionBCIntegratorHRevSens::GetFaceEnergy(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   const int state_elem_num = trans.Elem1->ElementNo;
   const int ndof = el1.GetDof();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
   auto *dof_tr = adjoint.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix shape;
#else
   auto &shape = integ.shape;
#endif
   shape.SetSize(ndof);

   const mfem::IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el1.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   // auto &h = integ.h;
   auto &theta_f = integ.theta_f;

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      const double trans_weight = trans.Weight();

      const double w = alpha * ip.weight * trans_weight;

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      // const double flux = h * ((elfun * shape) - theta_f);
      const double adjoint = psi * shape;

      /// dummy functional for adjoint-weighted residual
      // fun += adjoint * flux * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += adjoint * flux * w;
      // double adjoint_bar = fun_bar * flux * w;
      double flux_bar = fun_bar * adjoint * w;
      // double w_bar = fun_bar * adjoint * flux;

      /// const double adjoint = psi * shape;

      /// const double flux = h * ((elfun * shape) - theta_f);
      const double h_bar = flux_bar * ((elfun * shape) - theta_f);

      /// const double w = alpha * ip.weight * trans_weight;

      fun += h_bar;
   }
   return fun;
}

double ConvectionBCIntegratorFluidTempRevSens::GetFaceEnergy(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun)
{
   const int state_elem_num = trans.Elem1->ElementNo;
   const int ndof = el1.GetDof();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
   auto *dof_tr = adjoint.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix shape;
#else
   auto &shape = integ.shape;
#endif
   shape.SetSize(ndof);

   const mfem::IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el1.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   auto &h = integ.h;
   // auto &theta_f = integ.theta_f;

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      const double trans_weight = trans.Weight();

      const double w = alpha * ip.weight * trans_weight;

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      // const double flux = h * ((elfun * shape) - theta_f);
      const double adjoint = psi * shape;

      /// dummy functional for adjoint-weighted residual
      // fun += adjoint * flux * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += adjoint * flux * w;
      // double adjoint_bar = fun_bar * flux * w;
      double flux_bar = fun_bar * adjoint * w;
      // double w_bar = fun_bar * adjoint * flux;

      /// const double adjoint = psi * shape;

      /// const double flux = h * ((elfun * shape) - theta_f);
      const double theta_f_bar = -flux_bar * h;

      /// const double w = alpha * ip.weight * trans_weight;

      fun += theta_f_bar;
   }
   return fun;
}

void OutfluxBCIntegrator::AssembleFaceVector(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::Vector &elvect)
{
   int ndof = el1.GetDof();

#ifdef MFEM_THREAD_SAFE
   mfem::Vector shape;
#endif
   shape.SetSize(ndof);
   elvect.SetSize(ndof);

   const auto *ir = IntRule;
   if (ir == nullptr)
   {
      int order = el1.GetOrder() + el2.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const auto &ip = ir->IntPoint(i);

      // Set the integration point in the face and the neighboring element
      trans.SetAllIntPoints(&ip);

      const double w = alpha * ip.weight * trans.Face->Weight();

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      add(elvect, w * flux, shape, elvect);
   }
}

void OutfluxBCIntegrator::AssembleFaceGrad(
    const mfem::FiniteElement &el1,
    const mfem::FiniteElement &el2,
    mfem::FaceElementTransformations &trans,
    const mfem::Vector &elfun,
    mfem::DenseMatrix &elmat)
{
   int ndof = el1.GetDof();
   elmat.SetSize(ndof);
   elmat = 0.0;
}

void OutfluxBCIntegratorMeshRevSens::AssembleRHSElementVect(
    const mfem::FiniteElement &mesh_el,
    mfem::ElementTransformation &mesh_trans,
    mfem::Vector &mesh_coords_bar)
{
   const int element = mesh_trans.ElementNo;
   auto &trans =
       *state.FESpace()->GetMesh()->GetBdrFaceTransformations(element);
   const int state_elem_num = trans.Elem1->ElementNo;
   const auto &el1 = *state.FESpace()->GetFE(state_elem_num);

   const int mesh_ndof = mesh_el.GetDof();
   const int ndof = el1.GetDof();
   const int space_dim = trans.GetSpaceDim();

   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun;
   mfem::Vector psi;
#endif
   auto *dof_tr = state.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   state.GetSubVector(vdofs, elfun);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(elfun);
   }

   dof_tr = adjoint.FESpace()->GetElementVDofs(state_elem_num, vdofs);
   adjoint.GetSubVector(vdofs, psi);
   if (dof_tr != nullptr)
   {
      dof_tr->InvTransformPrimal(psi);
   }

#ifdef MFEM_THREAD_SAFE
   DenseMatrix shape;
   DenseMatrix PointMat_bar;
#else
   auto &shape = integ.shape;
#endif

   shape.SetSize(ndof);
   PointMat_bar.SetSize(space_dim, mesh_ndof);

   const mfem::IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order = 2 * el1.GetOrder() + trans.OrderW();
      ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
   }

   auto &alpha = integ.alpha;
   auto &flux = integ.flux;

   mesh_coords_bar.SetSize(mesh_ndof * space_dim);
   mesh_coords_bar = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      // Set the integration point in the face and the neighboring element
      const auto &ip = ir->IntPoint(i);
      trans.SetIntPoint(&ip);

      // const double trans_weight = trans.Weight();
      // const double w = alpha * ip.weight * trans_weight;

      // Access the neighboring element's integration point
      const auto &eip = trans.GetElement1IntPoint();
      el1.CalcShape(eip, shape);

      const double adjoint = psi * shape;

      /// dummy functional for adjoint-weighted residual
      // fun += adjoint * flux * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += adjoint * flux * w;
      // double adjoint_bar = fun_bar * flux * w;
      // double flux_bar = fun_bar * adjoint * w;
      double w_bar = fun_bar * adjoint * flux;

      /// const double adjoint = psi * shape;

      /// const double w = alpha * ip.weight * trans_weight;
      const double trans_weight_bar = w_bar * alpha * ip.weight;

      PointMat_bar = 0.0;
      trans.WeightRevDiff(trans_weight_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < mesh_ndof; ++j)
      {
         for (int k = 0; k < space_dim; ++k)
         {
            mesh_coords_bar(k * mesh_ndof + j) += PointMat_bar(k, j);
         }
      }
   }
}

// void DiffusionIntegrator::AssembleElementMatrix(const FiniteElement &el,
//                                                 ElementTransformation &Trans,
//                                                 DenseMatrix &elmat)
// {
//    int nd = el.GetDof();
//    dim = el.GetDim();
//    int spaceDim = Trans.GetSpaceDim();
//    bool square = (dim == spaceDim);
//    double w;

//    if (VQ)
//    {
//       MFEM_VERIFY(VQ->GetVDim() == spaceDim,
//                   "Unexpected dimension for VectorCoefficient");
//    }
//    if (MQ)
//    {
//       MFEM_VERIFY(MQ->GetWidth() == spaceDim,
//                   "Unexpected width for MatrixCoefficient");
//       MFEM_VERIFY(MQ->GetHeight() == spaceDim,
//                   "Unexpected height for MatrixCoefficient");
//    }

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix dshape(nd, dim), dshapedxt(nd, spaceDim);
//    DenseMatrix dshapedxt_m(nd, MQ ? spaceDim : 0);
//    DenseMatrix M(MQ ? spaceDim : 0);
//    Vector D(VQ ? VQ->GetVDim() : 0);
// #else
//    dshape.SetSize(nd, dim);
//    dshapedxt.SetSize(nd, spaceDim);
//    dshapedxt_m.SetSize(nd, MQ ? spaceDim : 0);
//    M.SetSize(MQ ? spaceDim : 0);
//    D.SetSize(VQ ? VQ->GetVDim() : 0);
// #endif
//    elmat.SetSize(nd);

//    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);

//    elmat = 0.0;
//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       el.CalcDShape(ip, dshape);

//       Trans.SetIntPoint(&ip);
//       w = Trans.Weight();
//       w = ip.weight / (square ? w : w * w * w);
//       // AdjugateJacobian = / adj(J),         if J is square
//       //                    \ adj(J^t.J).J^t, otherwise
//       Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
//       if (MQ)
//       {
//          MQ->Eval(M, Trans, ip);
//          M *= w;
//          Mult(dshapedxt, M, dshapedxt_m);
//          AddMultABt(dshapedxt_m, dshapedxt, elmat);
//       }
//       else if (VQ)
//       {
//          VQ->Eval(D, Trans, ip);
//          D *= w;
//          AddMultADAt(dshapedxt, D, elmat);
//       }
//       else
//       {
//          if (Q)
//          {
//             w *= Q->Eval(Trans, ip);
//          }
//          AddMult_a_AAt(w, dshapedxt, elmat);
//       }
//    }
// }

// void DiffusionIntegrator::AssembleElementMatrix2(const FiniteElement
// &trial_fe,
//                                                  const FiniteElement
//                                                  &test_fe,
//                                                  ElementTransformation
//                                                  &Trans, DenseMatrix &elmat)
// {
//    int tr_nd = trial_fe.GetDof();
//    int te_nd = test_fe.GetDof();
//    dim = trial_fe.GetDim();
//    int spaceDim = Trans.GetSpaceDim();
//    bool square = (dim == spaceDim);
//    double w;

//    if (VQ)
//    {
//       MFEM_VERIFY(VQ->GetVDim() == spaceDim,
//                   "Unexpected dimension for VectorCoefficient");
//    }
//    if (MQ)
//    {
//       MFEM_VERIFY(MQ->GetWidth() == spaceDim,
//                   "Unexpected width for MatrixCoefficient");
//       MFEM_VERIFY(MQ->GetHeight() == spaceDim,
//                   "Unexpected height for MatrixCoefficient");
//    }

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix dshape(tr_nd, dim), dshapedxt(tr_nd, spaceDim);
//    DenseMatrix te_dshape(te_nd, dim), te_dshapedxt(te_nd, spaceDim);
//    DenseMatrix invdfdx(dim, spaceDim);
//    DenseMatrix dshapedxt_m(te_nd, MQ ? spaceDim : 0);
//    DenseMatrix M(MQ ? spaceDim : 0);
//    Vector D(VQ ? VQ->GetVDim() : 0);
// #else
//    dshape.SetSize(tr_nd, dim);
//    dshapedxt.SetSize(tr_nd, spaceDim);
//    te_dshape.SetSize(te_nd, dim);
//    te_dshapedxt.SetSize(te_nd, spaceDim);
//    invdfdx.SetSize(dim, spaceDim);
//    dshapedxt_m.SetSize(te_nd, MQ ? spaceDim : 0);
//    M.SetSize(MQ ? spaceDim : 0);
//    D.SetSize(VQ ? VQ->GetVDim() : 0);
// #endif
//    elmat.SetSize(te_nd, tr_nd);

//    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(trial_fe,
//    test_fe);

//    elmat = 0.0;
//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       trial_fe.CalcDShape(ip, dshape);
//       test_fe.CalcDShape(ip, te_dshape);

//       Trans.SetIntPoint(&ip);
//       CalcAdjugate(Trans.Jacobian(), invdfdx);
//       w = Trans.Weight();
//       w = ip.weight / (square ? w : w * w * w);
//       Mult(dshape, invdfdx, dshapedxt);
//       Mult(te_dshape, invdfdx, te_dshapedxt);
//       // invdfdx, dshape, and te_dshape no longer needed
//       if (MQ)
//       {
//          MQ->Eval(M, Trans, ip);
//          M *= w;
//          Mult(te_dshapedxt, M, dshapedxt_m);
//          AddMultABt(dshapedxt_m, dshapedxt, elmat);
//       }
//       else if (VQ)
//       {
//          VQ->Eval(D, Trans, ip);
//          D *= w;
//          AddMultADAt(dshapedxt, D, elmat);
//       }
//       else
//       {
//          if (Q)
//          {
//             w *= Q->Eval(Trans, ip);
//          }
//          dshapedxt *= w;
//          AddMultABt(te_dshapedxt, dshapedxt, elmat);
//       }
//    }
// }

// void DiffusionIntegrator::AssembleElementVector(const FiniteElement &el,
//                                                 ElementTransformation &Tr,
//                                                 const Vector &elfun,
//                                                 Vector &elvect)
// {
//    int nd = el.GetDof();
//    dim = el.GetDim();
//    int spaceDim = Tr.GetSpaceDim();
//    double w;

//    if (VQ)
//    {
//       MFEM_VERIFY(VQ->GetVDim() == spaceDim,
//                   "Unexpected dimension for VectorCoefficient");
//    }
//    if (MQ)
//    {
//       MFEM_VERIFY(MQ->GetWidth() == spaceDim,
//                   "Unexpected width for MatrixCoefficient");
//       MFEM_VERIFY(MQ->GetHeight() == spaceDim,
//                   "Unexpected height for MatrixCoefficient");
//    }

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix dshape(nd, dim), invdfdx(dim, spaceDim), M(MQ ? spaceDim : 0);
//    Vector D(VQ ? VQ->GetVDim() : 0);
// #else
//    dshape.SetSize(nd, dim);
//    invdfdx.SetSize(dim, spaceDim);
//    M.SetSize(MQ ? spaceDim : 0);
//    D.SetSize(VQ ? VQ->GetVDim() : 0);
// #endif
//    vec.SetSize(dim);
//    vecdxt.SetSize((VQ || MQ) ? spaceDim : 0);
//    pointflux.SetSize(spaceDim);

//    elvect.SetSize(nd);

//    const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el);

//    elvect = 0.0;
//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       el.CalcDShape(ip, dshape);

//       Tr.SetIntPoint(&ip);
//       CalcAdjugate(Tr.Jacobian(), invdfdx);  // invdfdx = adj(J)
//       w = ip.weight / Tr.Weight();

//       if (!MQ && !VQ)
//       {
//          dshape.MultTranspose(elfun, vec);
//          invdfdx.MultTranspose(vec, pointflux);
//          if (Q)
//          {
//             w *= Q->Eval(Tr, ip);
//          }
//       }
//       else
//       {
//          dshape.MultTranspose(elfun, vec);
//          invdfdx.MultTranspose(vec, vecdxt);
//          if (MQ)
//          {
//             MQ->Eval(M, Tr, ip);
//             M.Mult(vecdxt, pointflux);
//          }
//          else
//          {
//             VQ->Eval(D, Tr, ip);
//             for (int j = 0; j < spaceDim; ++j)
//             {
//                pointflux[j] = D[j] * vecdxt[j];
//             }
//          }
//       }
//       pointflux *= w;
//       invdfdx.Mult(pointflux, vec);
//       dshape.AddMult(vec, elvect);
//    }
// }

// void DiffusionIntegrator::ComputeElementFlux(const FiniteElement &el,
//                                              ElementTransformation &Trans,
//                                              Vector &u,
//                                              const FiniteElement &fluxelem,
//                                              Vector &flux,
//                                              bool with_coef,
//                                              const IntegrationRule *ir)
// {
//    int nd, spaceDim, fnd;

//    nd = el.GetDof();
//    dim = el.GetDim();
//    spaceDim = Trans.GetSpaceDim();

//    if (VQ)
//    {
//       MFEM_VERIFY(VQ->GetVDim() == spaceDim,
//                   "Unexpected dimension for VectorCoefficient");
//    }
//    if (MQ)
//    {
//       MFEM_VERIFY(MQ->GetWidth() == spaceDim,
//                   "Unexpected width for MatrixCoefficient");
//       MFEM_VERIFY(MQ->GetHeight() == spaceDim,
//                   "Unexpected height for MatrixCoefficient");
//    }

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix dshape(nd, dim), invdfdx(dim, spaceDim);
//    DenseMatrix M(MQ ? spaceDim : 0);
//    Vector D(VQ ? VQ->GetVDim() : 0);
// #else
//    dshape.SetSize(nd, dim);
//    invdfdx.SetSize(dim, spaceDim);
//    M.SetSize(MQ ? spaceDim : 0);
//    D.SetSize(VQ ? VQ->GetVDim() : 0);
// #endif
//    vec.SetSize(dim);
//    vecdxt.SetSize(spaceDim);
//    pointflux.SetSize(MQ || VQ ? spaceDim : 0);

//    if (!ir)
//    {
//       ir = &fluxelem.GetNodes();
//    }
//    fnd = ir->GetNPoints();
//    flux.SetSize(fnd * spaceDim);

//    for (int i = 0; i < fnd; i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       el.CalcDShape(ip, dshape);
//       dshape.MultTranspose(u, vec);

//       Trans.SetIntPoint(&ip);
//       CalcInverse(Trans.Jacobian(), invdfdx);
//       invdfdx.MultTranspose(vec, vecdxt);

//       if (with_coef)
//       {
//          if (!MQ && !VQ)
//          {
//             if (Q)
//             {
//                vecdxt *= Q->Eval(Trans, ip);
//             }
//             for (int j = 0; j < spaceDim; j++)
//             {
//                flux(fnd * j + i) = vecdxt(j);
//             }
//          }
//          else
//          {
//             if (MQ)
//             {
//                MQ->Eval(M, Trans, ip);
//                M.Mult(vecdxt, pointflux);
//             }
//             else
//             {
//                VQ->Eval(D, Trans, ip);
//                for (int j = 0; j < spaceDim; ++j)
//                {
//                   pointflux[j] = D[j] * vecdxt[j];
//                }
//             }
//             for (int j = 0; j < spaceDim; j++)
//             {
//                flux(fnd * j + i) = pointflux(j);
//             }
//          }
//       }
//       else
//       {
//          for (int j = 0; j < spaceDim; j++)
//          {
//             flux(fnd * j + i) = vecdxt(j);
//          }
//       }
//    }
// }

// double DiffusionIntegrator::ComputeFluxEnergy(const FiniteElement &fluxelem,
//                                               ElementTransformation &Trans,
//                                               Vector &flux,
//                                               Vector *d_energy)
// {
//    int nd = fluxelem.GetDof();
//    dim = fluxelem.GetDim();
//    int spaceDim = Trans.GetSpaceDim();

// #ifdef MFEM_THREAD_SAFE
//    DenseMatrix M;
//    Vector D(VQ ? VQ->GetVDim() : 0);
// #else
//    D.SetSize(VQ ? VQ->GetVDim() : 0);
// #endif

//    shape.SetSize(nd);
//    pointflux.SetSize(spaceDim);
//    if (d_energy)
//    {
//       vec.SetSize(spaceDim);
//    }
//    if (MQ)
//    {
//       M.SetSize(spaceDim);
//    }

//    int order = 2 * fluxelem.GetOrder();  // <--
//    const IntegrationRule *ir = &IntRules.Get(fluxelem.GetGeomType(), order);

//    double energy = 0.0;
//    if (d_energy)
//    {
//       *d_energy = 0.0;
//    }

//    for (int i = 0; i < ir->GetNPoints(); i++)
//    {
//       const IntegrationPoint &ip = ir->IntPoint(i);
//       fluxelem.CalcShape(ip, shape);

//       pointflux = 0.0;
//       for (int k = 0; k < spaceDim; k++)
//       {
//          for (int j = 0; j < nd; j++)
//          {
//             pointflux(k) += flux(k * nd + j) * shape(j);
//          }
//       }

//       Trans.SetIntPoint(&ip);
//       double w = Trans.Weight() * ip.weight;

//       if (MQ)
//       {
//          MQ->Eval(M, Trans, ip);
//          energy += w * M.InnerProduct(pointflux, pointflux);
//       }
//       else if (VQ)
//       {
//          VQ->Eval(D, Trans, ip);
//          D *= pointflux;
//          energy += w * (D * pointflux);
//       }
//       else
//       {
//          double e = (pointflux * pointflux);
//          if (Q)
//          {
//             e *= Q->Eval(Trans, ip);
//          }
//          energy += w * e;
//       }

//       if (d_energy)
//       {
//          // transform pointflux to the ref. domain and integrate the
//          components Trans.Jacobian().MultTranspose(pointflux, vec); for (int
//          k = 0; k < dim; k++)
//          {
//             (*d_energy)[k] += w * vec[k] * vec[k];
//          }
//          // TODO: Q, VQ, MQ
//       }
//    }

//    return energy;
// }

// const IntegrationRule &DiffusionIntegrator::GetRule(
//     const FiniteElement &trial_fe,
//     const FiniteElement &test_fe)
// {
//    int order;
//    if (trial_fe.Space() == FunctionSpace::Pk)
//    {
//       order = trial_fe.GetOrder() + test_fe.GetOrder() - 2;
//    }
//    else
//    {
//       // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
//       order = trial_fe.GetOrder() + test_fe.GetOrder() + trial_fe.GetDim() -
//       1;
//    }

//    if (trial_fe.Space() == FunctionSpace::rQk)
//    {
//       return RefinedIntRules.Get(trial_fe.GetGeomType(), order);
//    }
//    return IntRules.Get(trial_fe.GetGeomType(), order);
// }

// void DiffusionIntegratorMeshRevSens::AssembleRHSElementVect(
//     const FiniteElement &mesh_el,
//     ElementTransformation &mesh_trans,
//     Vector &mesh_coords_bar)
// {
//    /// TODO: Comment out all of these mach diffusion integrators and just use
//    /// the EM one
//    /// TODO: Implement DiffusionIntegratorMeshRevSens::AssembleRHSElementVect
//    /// (using NonlinearDiffusionIntegratorMeshRevSens as a starting point)
// }

}  // namespace mach
