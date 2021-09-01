#include "mfem.hpp"

#include "mfem_common_integ.hpp"

using namespace mfem;

namespace mach
{

void DiffusionIntegratorMeshSens::AssembleRHSElementVect(
   const mfem::FiniteElement &mesh_el,
   mfem::ElementTransformation &mesh_trans,
   mfem::Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
   int element = mesh_trans.ElementNo;
   auto &el = *state->FESpace()->GetFE(element);
   auto &trans = *state->FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int spaceDim = trans.GetSpaceDim();
   const bool square = (dim == spaceDim);
   mesh_coords_bar.SetSize(ndof*dim);
   mesh_coords_bar = 0.0;

   state->FESpace()->GetElementVDofs(element, vdofs);
   state->GetSubVector(vdofs, elfun);
   adjoint->GetSubVector(vdofs, psi);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(el_ndof, dim),
   DenseMatrix dshapedxt(el_ndof, spaceDim);
   DenseMatrix dshapedxt_bar(el_ndof, dim);
   DenseMatrix PointMat_bar(dim, ndof);
#else
   dshape.SetSize(el_ndof, dim);
   dshapedxt.SetSize(el_ndof, spaceDim);
   dshapedxt_bar.SetSize(el_ndof, dim);
   PointMat_bar.SetSize(dim ,ndof);
#endif

   /// these vector's size is the spatial dimension we can stack allocate
   double DT_state_buffer[3];
   Vector DT_state(DT_state_buffer, dim);
   double DT_psi_buffer[3];
   Vector DT_psi(DT_psi_buffer, dim);

   double DT_state_bar_buffer[3];
   Vector DT_state_bar(DT_state_bar_buffer, dim);
   double DT_psi_bar_buffer[3];
   Vector DT_psi_bar(DT_psi_bar_buffer, dim);

   double adj_jac_bar_buffer[9];
   DenseMatrix adj_jac_bar(adj_jac_bar_buffer, dim, dim);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int order;
      if (el.Space() == FunctionSpace::Pk)
      {
         order = 2 * el.GetOrder() - 2;
      }
      else
      {
         // order = 2*el.GetOrder() - 2;  // <-- this seems to work fine too
         order = 2 * el.GetOrder() + el.GetDim() - 1;
      }
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);

      trans.SetIntPoint(&ip);
      double w;
      if (square)
         w = ip.weight / trans.Weight();
      else
         w = ip.weight / pow(trans.Weight(), 3);

      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, trans.AdjugateJacobian(), dshapedxt);

      // AddMult_a_AAt(w, dshapedxt, elmat);

      dshapedxt.MultTranspose(elfun, DT_state);
      dshapedxt.MultTranspose(psi, DT_psi);

      /// dummy functional for adjoint-weighted residual
      const double DT_state_dot_DT_psi = DT_state * DT_psi;
      // fun += DT_state_dot_DT_psi * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += DT_state_dot_DT_psi * w;
      double DT_state_dot_DT_psi_bar = fun_bar * w;
      double w_bar = fun_bar * DT_state_dot_DT_psi;

      /// const double DT_state_dot_DT_psi = DT_state * DT_psi;
      DT_state_bar = 0.0;
      DT_psi_bar = 0.0;
      add(DT_psi_bar, DT_state_dot_DT_psi_bar, DT_state, DT_psi_bar);
      add(DT_state_bar, DT_state_dot_DT_psi_bar, DT_psi, DT_state_bar);
      
      dshapedxt_bar = 0.0;
      /// dshapedxt.MultTranspose(psi, DT_psi);
      AddMultVWt(psi, DT_psi_bar, dshapedxt_bar);

      /// dshapedxt.MultTranspose(elfun, DT_state);
      AddMultVWt(elfun, DT_state_bar, dshapedxt_bar);

      /// Mult(dshape, trans.AdjugateJacobian(), dshapedxt);
      adj_jac_bar = 0.0;
      MultAtB(dshape, dshapedxt_bar, adj_jac_bar);

      /// w = ip.weight / trans.Weight();
      double trans_weight_bar = -w_bar * ip.weight / pow(trans.Weight(), 2);

      PointMat_bar = 0.0;
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= trans_weight_bar;

      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);
      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            mesh_coords_bar(d*ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void VectorFEWeakDivergenceIntegratorMeshSens::AssembleRHSElementVect(
   const mfem::FiniteElement &mesh_el,
   mfem::ElementTransformation &mesh_trans,
   mfem::Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
   int element = mesh_trans.ElementNo;
   auto &nd_el = *state->FESpace()->GetFE(element);
   auto &h1_el = *adjoint->FESpace()->GetFE(element);

   const int ndof = mesh_el.GetDof();
   const int dim = mesh_el.GetDim();
   const int nd_ndof = nd_el.GetDof();
   const int h1_ndof = h1_el.GetDof();
   mesh_coords_bar.SetSize(ndof*dim);
   mesh_coords_bar = 0.0;

   state->FESpace()->GetElementVDofs(element, vdofs);
   state->GetSubVector(vdofs, elfun);
   adjoint->FESpace()->GetElementDofs(element, vdofs);
   adjoint->GetSubVector(vdofs, psi);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix dshape(h1_ndof, dim);
   DenseMatrix dshapedxt(h1_ndof, dim);
   DenseMatrix vshape(nd_ndof, dim);
   DenseMatrix vshapedxt(nd_ndof, dim);
   DenseMatrix dshapedxt_bar(h1_ndof, dim);
   DenseMatrix vshapedxt_bar(nd_ndof, dim);
   DenseMatrix PointMat_bar(dim, ndof);
#else
   dshape.SetSize(h1_ndof, dim);
   dshapedxt.SetSize(h1_ndof, dim);
   vshape.SetSize(nd_ndof, dim);
   vshapedxt.SetSize(nd_ndof, dim);
   dshapedxt_bar.SetSize(h1_ndof, dim);
   vshapedxt_bar.SetSize(nd_ndof, dim);
   PointMat_bar.SetSize(dim, ndof);
#endif

   /// these vector's size is the spatial dimension we can stack allocate
   double VT_state_buffer[3];
   Vector VT_state(VT_state_buffer, dim);
   double DT_psi_buffer[3];
   Vector DT_psi(DT_psi_buffer, dim);

   double VT_state_bar_buffer[3];
   Vector VT_state_bar(VT_state_bar_buffer, dim);
   double DT_psi_bar_buffer[3];
   Vector DT_psi_bar(DT_psi_bar_buffer, dim);

   double adj_jac_bar_buffer[9];
   DenseMatrix adj_jac_bar(adj_jac_bar_buffer, dim, dim); 
   double adj_jac_bar_buffer_temp[9];
   DenseMatrix adj_jac_bar_temp(adj_jac_bar_buffer_temp, dim, dim);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(mesh_trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      int ir_order = (nd_el.Space() == FunctionSpace::Pk) ?
                     (nd_el.GetOrder() + h1_el.GetOrder() - 1) :
                     (nd_el.GetOrder() + h1_el.GetOrder() + 2*(dim-2));
      ir = &IntRules.Get(nd_el.GetGeomType(), ir_order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      h1_el.CalcDShape(ip, dshape);

      mesh_trans.SetIntPoint(&ip);
      double w = ip.weight / mesh_trans.Weight();

      Mult(dshape, mesh_trans.AdjugateJacobian(), dshapedxt);

      nd_el.CalcVShape(ip, vshape);
      Mult(vshape, mesh_trans.AdjugateJacobian(), vshapedxt);

      /// dummy functional for adjoint-weighted residual...
      /// AddMultABt(dshapedxt, vshape, elmat);
      vshapedxt.MultTranspose(elfun, VT_state);
      dshapedxt.MultTranspose(psi, DT_psi);
      const double VT_state_dot_DT_psi = VT_state * DT_psi;
      // fun -= VT_state_dot_DT_psi * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun -= VT_state_dot_DT_psi * w;
      double VT_state_dot_DT_psi_bar = -fun_bar * w;
      double w_bar = -fun_bar * VT_state_dot_DT_psi;

      /// const double VT_state_dot_DT_psi = VT_state * DT_psi;
      VT_state_bar = 0.0;
      DT_psi_bar = 0.0;
      add(VT_state_bar, VT_state_dot_DT_psi_bar, DT_psi, VT_state_bar);
      add(DT_psi_bar, VT_state_dot_DT_psi_bar, VT_state, DT_psi_bar);
      
      /// dshapedxt.MultTranspose(psi, DT_psi);
      dshapedxt_bar = 0.0;
      AddMultVWt(psi, DT_psi_bar, dshapedxt_bar);

      /// vshapedxt.MultTranspose(elfun, VT_state);
      vshapedxt_bar = 0.0;
      AddMultVWt(elfun, VT_state_bar, vshapedxt_bar);

      /// Mult(vshape, mesh_trans.AdjugateJacobian(), vshapedxt);
      adj_jac_bar = 0.0;
      MultAtB(vshape, vshapedxt_bar, adj_jac_bar);

      /// Mult(dshape, mesh_trans.AdjugateJacobian(), dshapedxt);
      adj_jac_bar_temp = 0.0;
      MultAtB(dshape, dshapedxt_bar, adj_jac_bar_temp);
      adj_jac_bar += adj_jac_bar_temp;

      /// w = ip.weight / mesh_trans.Weight();
      double trans_weight_bar = -w_bar * ip.weight / pow(mesh_trans.Weight(), 2);
      PointMat_bar = 0.0;
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= trans_weight_bar;

      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);
      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            mesh_coords_bar(d*ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

/** Not differentiated, not needed since we use linear form version of MagneticLoad
void VectorFECurlIntegratorMeshSens::AssembleRHSElementVect(
   const mfem::FiniteElement &mesh_el,
   mfem::ElementTransformation &mesh_trans,
   mfem::Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
   int element = mesh_trans.ElementNo;
   auto &state_el = *state->FESpace()->GetFE(element);
   auto &adjoint_el = *adjoint->FESpace()->GetFE(element);
   const FiniteElement *nd_el;
   const FiniteElement *vec_el;
   if (state_el.GetMapType() == mfem::FiniteElement::H_CURL)
   {
      nd_el = &state_el;
      vec_el = &adjoint_el;
   }
   else
   {
      nd_el = &adjoint_el;
      vec_el = &state_el;
   }

   const int ndof = mesh_el.GetDof();
   const int dim = mesh_el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;
   const int nd_ndof = nd_el->GetDof();
   const int vec_ndof = vec_el->GetDof();
   mesh_coords_bar.SetSize(ndof*dim);
   mesh_coords_bar = 0.0;

   state->FESpace()->GetElementVDofs(element, vdofs);
   state->GetSubVector(vdofs, elfun);
   adjoint->FESpace()->GetElementDofs(element, vdofs);
   adjoint->GetSubVector(vdofs, psi);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(nd_ndof, dimc);
   DenseMatrix curlshape_dFt(nd_ndof, dimc);
   DenseMatrix vshape(vec_ndof, dimc);
   DenseMatrix vshapedxt(vec_ndof, dimc);
   DenseMatrix curlshape_dFt_bar(dimc, nd_ndof);  // transposed dimensions of curlshape_dFt so I don't have to transpose J later
   // DenseMatrix vshape_bar(vec_ndof, dimc);
   DenseMatrix vshapedxt_bar(vec_ndof, dimc);
   DenseMatrix PointMat_bar(dim, ndofc);
#else
   curlshape.SetSize(nd_ndof, dimc);
   curlshape_dFt.SetSize(nd_ndof, dimc);
   vshape.SetSize(vec_ndof, dimc);
   vshapedxt.SetSize(vec_ndof, dimc);
   curlshape_dFt_bar.SetSize(dimc, nd_ndof);  // transposed dimensions of curlshape_dFt so I don't have to transpose J later
   // vshape_bar.SetSize(vec_ndof, dimc);
   vshapedxt_bar.SetSize(vec_ndof, dimc);
   PointMat_bar.SetSize(dim, ndof);
#endif
   Vector shape(vshape.GetData(), vec_ndof);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(mesh_trans);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = nd_el->GetOrder() + vec_el->GetOrder() - 1; // <--
      ir = &IntRules.Get(nd_el->GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      double w = ip.weight;
      if (dim == 3)
      {
         // vec_el->CalcVShape(ip, vshape);
         // vec_el->CalcVShape(isotrans, vshape);
         vec_el->CalcVShape(ip, vshape);
         MultABt(vshape, isotrans.Jacobian(), vshapedxt);
         w /= isotrans.Weight();

         // nd_el->CalcCurlShape(ip, curlshape);
         // MultABt(curlshape, isotrans.Jacobian(), curlshape_dFt);
         nd_el->CalcCurlShape(ip, curlshape_dFt);
      }
      else
      {
         vec_el->CalcShape(ip, shape);
         nd_el->CalcCurlShape(ip, curlshape_dFt);
      }

      // if (Q)
      // {
      //    w *= Q->Eval(isotrans, ip);
      // }

      // Note: shape points to the same data as vshape
      PointMat_bar = 0.0;
      double fun_bar = 1.0;
      double w_bar = 0.0;
      if (state_el.GetMapType() == mfem::FiniteElement::H_CURL)
      {
         double CT_state_buffer[3];
         Vector CT_state(CT_state_buffer, dimc);
         double VT_psi_buffer[3];
         Vector VT_psi(VT_psi_buffer, dimc);

         // AddMultABt(vshape, curlshape_dFt, elmat);
         curlshape_dFt.MultTranspose(elfun, CT_state);
         vshapedxt.MultTranspose(psi, VT_psi);
         const double VT_psi_dot_CT_state = VT_psi * CT_state;
         // fun += w * VT_psi_dot_CT_state;
         /// start reverse pass

         /// fun += w * VT_psi_dot_CT_state;
         double VT_psi_dot_CT_state_bar = fun_bar * w;
         w_bar += fun_bar * VT_psi_dot_CT_state;

         /// const double VT_psi_dot_CT_state = VT_psi * CT_state;
         double CT_state_bar_buffer[3];
         Vector CT_state_bar(CT_state_bar_buffer, dimc); CT_state_bar = 0.0;
         double VT_psi_bar_buffer[3];
         Vector VT_psi_bar(VT_psi_bar_buffer, dimc); VT_psi_bar = 0.0;
         add(CT_state_bar, VT_psi_dot_CT_state_bar, VT_psi, CT_state_bar);
         add(VT_psi_bar, VT_psi_dot_CT_state_bar, CT_state, VT_psi_bar);

         /// vshape.MultTranspose(psi, VT_psi);
         vshapedxt_bar = 0.0;
         AddMultVWt(psi, VT_psi_bar, vshapedxt_bar);

         /// curlshape_dFt.MultTranspose(elfun, CT_state);
         curlshape_dFt_bar = 0.0;
         AddMultVWt(CT_state_bar, elfun, curlshape_dFt_bar);
      }
      else
      {
         double CT_psi_buffer[3];
         Vector CT_psi(CT_psi_buffer, dimc);
         double VT_state_buffer[3];
         Vector VT_state(VT_state_buffer, dimc);
         // AddMultABt(curlshape_dFt, vshape, elmat);
         curlshape_dFt.MultTranspose(psi, CT_psi);
         vshapedxt.MultTranspose(elfun, VT_state);
         const double VT_state_dot_CT_psi = VT_state * CT_psi;
         // fun += w * VT_state_dot_CT_psi;

         double VT_state_dot_CT_psi_bar = fun_bar * w;
         w_bar += fun_bar * VT_state_dot_CT_psi;

         /// const double VT_state_dot_CT_psi = VT_state * CT_psi;
         double VT_state_bar_buffer[3];
         Vector VT_state_bar(VT_state_bar_buffer, dimc); VT_state_bar = 0.0;
         double CT_psi_bar_buffer[3];
         Vector CT_psi_bar(CT_psi_bar_buffer, dimc); CT_psi_bar = 0.0;
         add(VT_state_bar, VT_state_dot_CT_psi_bar, CT_psi, VT_state_bar);
         add(CT_psi_bar, VT_state_dot_CT_psi_bar, VT_state, CT_psi_bar);

         /// vshape.MultTranspose(elfun, VT_state);
         vshapedxt_bar = 0.0;
         AddMultVWt(elfun, VT_state_bar, vshapedxt_bar);

         /// curlshape_dFt.MultTranspose(psi, CT_psi);
         curlshape_dFt_bar = 0.0;
         AddMultVWt(CT_psi_bar, psi, curlshape_dFt_bar);
      }

      if (dim == 3)
      {
         // /// vec_el->CalcVShape(ip, vshape);
         /// vec_el->CalcVShape(isotrans, vshape);
         // vec_el->CalcVShapeRevDiff(isotrans, vshape_bar, PointMat_bar);


         /// w /= isotrans.Weight();
         double weight_bar = -w_bar * ip.weight / pow(isotrans.Weight(), 2);
         isotrans.WeightRevDiff(weight_bar, PointMat_bar);

         /// MultAtB(vshape, isotrans.Jacobian(), vshapedxt);
         double jac_bar_buffer[9];
         DenseMatrix jac_bar(jac_bar_buffer, dimc, dimc); jac_bar = 0.0;
         MultAtB(vshapedxt_bar, vshape, jac_bar);
         isotrans.JacobianRevDiff(jac_bar, PointMat_bar);



         /// nd_el->CalcCurlShape(ip, curlshape_dFt);
         // /// MultABt(curlshape, isotrans.Jacobian(), curlshape_dFt);
         // double jac_bar_buffer[9];
         // DenseMatrix jac_bar(jac_bar_buffer, dimc, dimc); jac_bar = 0.0;
         // AddMult(curlshape_dFt_bar, curlshape, jac_bar);
         // isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

         // /// nd_el->CalcCurlShape(ip, curlshape);
      }
      else
      {
         /// vec_el->CalcShape(ip, shape);
         /// nd_el->CalcCurlShape(ip, curlshape_dFt);
      }

      // if (Q)
      // {
      //    /// double w = ip.weight * Q->Eval(isotrans, ip);
      //    double Q_bar = w_bar * ip.weight;
      //    Q->EvalRevDiff(Q_bar, isotrans, ip, PointMat_bar);
      // }
      // else
      // {
      //    /// double w = ip.weight;
      // }

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dimc; ++d)
         {
            mesh_coords_bar(d*ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}
*/

void VectorFEMassIntegratorMeshSens::AssembleRHSElementVect(
   const mfem::FiniteElement &mesh_el,
   mfem::ElementTransformation &mesh_trans,
   mfem::Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and state vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector elfun, psi;
#endif
   int element = mesh_trans.ElementNo;
   auto &el = *state->FESpace()->GetFE(element);
   auto &trans = *state->FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int spaceDim = trans.GetSpaceDim();
   mesh_coords_bar.SetSize(ndof*dim);
   mesh_coords_bar = 0.0;

   state->FESpace()->GetElementVDofs(element, vdofs);
   state->GetSubVector(vdofs, elfun);
   adjoint->GetSubVector(vdofs, psi);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(el_ndof, dim),
   DenseMatrix vshapedxt(el_ndof, spaceDim);
   DenseMatrix vshapedxt_bar(el_ndof, dim);
   DenseMatrix PointMat_bar(dim, ndof);
#else
   vshape.SetSize(el_ndof, dim);
   vshapedxt.SetSize(el_ndof, spaceDim);
   vshapedxt_bar.SetSize(el_ndof, dim);
   PointMat_bar.SetSize(dim ,ndof);
#endif

   /// these vector's size is the spatial dimension we can stack allocate
   double VT_state_buffer[3];
   Vector VT_state(VT_state_buffer, dim);
   double VT_psi_buffer[3];
   Vector VT_psi(VT_psi_buffer, dim);

   double VT_state_bar_buffer[3];
   Vector VT_state_bar(VT_state_bar_buffer, dim);
   double VT_psi_bar_buffer[3];
   Vector VT_psi_bar(VT_psi_bar_buffer, dim);

   double adj_jac_bar_buffer[9];
   DenseMatrix adj_jac_bar(adj_jac_bar_buffer, dim, dim);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      // int order = 2 * el.GetOrder();
      int order = mesh_trans.OrderW() + 2 * el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);
      double w = alpha * ip.weight / trans.Weight();

      el.CalcVShape(ip, vshape);
      Mult(vshape, mesh_trans.AdjugateJacobian(), vshapedxt);

      // AddMult_a_AAt(w, vshapedxt, elmat);

      vshapedxt.MultTranspose(elfun, VT_state);
      vshapedxt.MultTranspose(psi, VT_psi);

      /// dummy functional for adjoint-weighted residual
      const double VT_state_dot_VT_psi = VT_state * VT_psi;
      // fun += VT_state_dot_VT_psi * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += VT_state_dot_VT_psi * w;
      double VT_state_dot_VT_psi_bar = fun_bar * w;
      double w_bar = fun_bar * VT_state_dot_VT_psi;

      /// const double VT_state_dot_VT_psi = VT_state * VT_psi;
      VT_state_bar = 0.0;
      VT_psi_bar = 0.0;
      add(VT_psi_bar, VT_state_dot_VT_psi_bar, VT_state, VT_psi_bar);
      add(VT_state_bar, VT_state_dot_VT_psi_bar, VT_psi, VT_state_bar);
      
      vshapedxt_bar = 0.0;
      /// vshapedxt.MultTranspose(psi, VT_psi);
      AddMultVWt(psi, VT_psi_bar, vshapedxt_bar);

      /// vshapedxt.MultTranspose(elfun, VT_state);
      AddMultVWt(elfun, VT_state_bar, vshapedxt_bar);

      /// Mult(vshape, trans.AdjugateJacobian(), vshapedxt);
      adj_jac_bar = 0.0;
      MultAtB(vshape, vshapedxt_bar, adj_jac_bar);

      /// double w = alpha * ip.weight / trans.Weight();
      double trans_weight_bar = -w_bar * ip.weight / pow(trans.Weight(), 2);

      PointMat_bar = 0.0;
      isotrans.WeightRevDiff(PointMat_bar);
      PointMat_bar *= trans_weight_bar;

      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);
      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            mesh_coords_bar(d*ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void VectorFEDomainLFIntegratorMeshSens::AssembleRHSElementVect(
   const mfem::FiniteElement &mesh_el,
   mfem::ElementTransformation &mesh_trans,
   mfem::Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and adjoint vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
   int element = mesh_trans.ElementNo;
   auto &el = *adjoint->FESpace()->GetFE(element);
   auto &trans = *adjoint->FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int spaceDim = trans.GetSpaceDim();
   mesh_coords_bar.SetSize(ndof*dim);
   mesh_coords_bar = 0.0;

   adjoint->FESpace()->GetElementVDofs(element, vdofs);
   adjoint->GetSubVector(vdofs, psi);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix vshape(el_ndof, dim),
   DenseMatrix vshapedxt(el_ndof, spaceDim);
   DenseMatrix vshapedxt_bar(el_ndof, dim);
   DenseMatrix PointMat_bar(dim, ndof);
#else
   vshape.SetSize(el_ndof, dim);
   vshapedxt.SetSize(el_ndof, spaceDim);
   vshapedxt_bar.SetSize(el_ndof, dim);
   PointMat_bar.SetSize(dim ,ndof);
#endif

   /// these vector's size is the spatial dimension we can stack allocate
   double vec_buffer[3];
   Vector vec(vec_buffer, dim);
   double VT_psi_buffer[3];
   Vector VT_psi(VT_psi_buffer, dim);

   double vec_bar_buffer[3];
   Vector vec_bar(vec_bar_buffer, dim);
   double VT_psi_bar_buffer[3];
   Vector VT_psi_bar(VT_psi_bar_buffer, dim);

   double adj_jac_bar_buffer[9];
   DenseMatrix adj_jac_bar(adj_jac_bar_buffer, dim, dim);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);
      double w = ip.weight * alpha;

      el.CalcVShape(ip, vshape);
      Mult(vshape, mesh_trans.AdjugateJacobian(), vshapedxt);

      F.Eval(vec, trans, ip);

      vshapedxt.MultTranspose(psi, VT_psi);

      /// dummy functional for adjoint-weighted residual
      // const double VT_psi_dot_vec = VT_psi * vec;

      /// fun += VT_psi_dot_vec * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += VT_psi_dot_vec * w;
      double VT_psi_dot_vec_bar = fun_bar * w;

      /// const double VT_psi_dot_vec = VT_psi * vec;
      vec_bar = 0.0;
      VT_psi_bar = 0.0;
      add(VT_psi_bar, VT_psi_dot_vec_bar, vec, VT_psi_bar);
      add(vec_bar, VT_psi_dot_vec_bar, VT_psi, vec_bar);

      /// F.Eval(vec, trans, ip);
      PointMat_bar = 0.0;
      F.EvalRevDiff(vec_bar, trans, ip, PointMat_bar);

      /// vshapedxt.MultTranspose(psi, VT_psi);
      vshapedxt_bar = 0.0;
      AddMultVWt(psi, VT_psi_bar, vshapedxt_bar);

      /// Mult(vshape, trans.AdjugateJacobian(), vshapedxt);
      adj_jac_bar = 0.0;
      MultAtB(vshape, vshapedxt_bar, adj_jac_bar);

      /// w = ip.weight * alpha;

      isotrans.AdjugateJacobianRevDiff(adj_jac_bar, PointMat_bar);
      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            mesh_coords_bar(d*ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

void VectorFEDomainLFCurlIntegratorMeshSens::AssembleRHSElementVect(
   const mfem::FiniteElement &mesh_el,
   mfem::ElementTransformation &mesh_trans,
   mfem::Vector &mesh_coords_bar)
{
   /// get the proper element, transformation, and adjoint vector
#ifdef MFEM_THREAD_SAFE
   mfem::Array<int> vdofs;
   mfem::Vector psi;
#endif
   int element = mesh_trans.ElementNo;
   auto &el = *adjoint->FESpace()->GetFE(element);
   auto &trans = *adjoint->FESpace()->GetElementTransformation(element);

   const int ndof = mesh_el.GetDof();
   const int el_ndof = el.GetDof();
   const int dim = el.GetDim();
   const int dimc = (dim == 3) ? 3 : 1;

   // const int spaceDim = trans.GetSpaceDim();
   mesh_coords_bar.SetSize(ndof*dim);
   mesh_coords_bar = 0.0;

   adjoint->FESpace()->GetElementVDofs(element, vdofs);
   adjoint->GetSubVector(vdofs, psi);

#ifdef MFEM_THREAD_SAFE
   DenseMatrix curlshape(el_ndof,dimc);
   DenseMatrix curlshape_bar(el_ndof,dimc);
   DenseMatrix PointMat_bar(dim, ndof);
#else
   curlshape.SetSize(el_ndof, dimc);
   curlshape_bar.SetSize(el_ndof, dimc);
   PointMat_bar.SetSize(dim ,ndof);
#endif
   auto &F = integ.F;
   auto &alpha = integ.alpha;

   /// these vector's size is the spatial dimension we can stack allocate
   double vec_buffer[3];
   Vector vec(vec_buffer, dim);

   double vec_bar_buffer[3];
   Vector vec_bar(vec_bar_buffer, dim);

   double CT_psi_buffer[3];
   Vector CT_psi(CT_psi_buffer, dim);

   double CT_psi_bar_buffer[3];
   Vector CT_psi_bar(CT_psi_bar_buffer, dim);

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
   dynamic_cast<IsoparametricTransformation&>(trans);

   const IntegrationRule *ir = IntRule;
   if (ir == nullptr)
   {
      // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      trans.SetIntPoint(&ip);
      double w = ip.weight * trans.Weight() * alpha;

      el.CalcPhysCurlShape(trans, curlshape);
      curlshape.MultTranspose(psi, CT_psi);

      F.Eval(vec, trans, ip);

      /// dummy functional for adjoint-weighted residual
      const double CT_psi_dot_vec = CT_psi * vec;

      /// fun += CT_psi_dot_vec * w;

      /// start reverse pass
      double fun_bar = 1.0;

      /// fun += CT_psi_dot_vec * w;
      double CT_psi_dot_vec_bar = fun_bar * w;
      double w_bar = fun_bar * CT_psi_dot_vec;

      /// const double CT_psi_dot_vec = CT_psi * vec;
      vec_bar = 0.0;
      CT_psi_bar = 0.0;
      add(CT_psi_bar, CT_psi_dot_vec_bar, vec, CT_psi_bar);
      add(vec_bar, CT_psi_dot_vec_bar, CT_psi, vec_bar);

      /// F.Eval(vec, trans, ip);
      PointMat_bar = 0.0;
      F.EvalRevDiff(vec_bar, trans, ip, PointMat_bar);

      /// curlshape.MultTranspose(psi, CT_psi);
      curlshape_bar = 0.0;
      AddMultVWt(psi, CT_psi_bar, curlshape_bar);

      /// el.CalcPhysCurlShape(trans, curlshape);
      el.CalcPhysCurlShapeRevDiff(trans, curlshape_bar, PointMat_bar);

      /// double w = ip.weight * trans.Weight() * alpha;
      double weight_bar = w_bar * ip.weight * alpha;
      isotrans.WeightRevDiff(weight_bar, PointMat_bar);

      // code to insert PointMat_bar into mesh_coords_bar;
      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            mesh_coords_bar(d*ndof + j) += PointMat_bar(d, j);
         }
      }
   }
}

double TestLFIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &trans,
   const Vector &elfun)
{
   const IntegrationRule *ir = NULL;
   {
      ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder());
   }

   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
                     dynamic_cast<IsoparametricTransformation&>(trans);

   double fun = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      fun += ip.weight * Q.Eval(isotrans, ip);
   }
   return fun;
}

void TestLFMeshSensIntegrator::AssembleRHSElementVect(
   const FiniteElement &mesh_el,
   ElementTransformation &mesh_trans,
   Vector &elvect)
{
   const IntegrationRule *ir = NULL;
   {
      ir = &IntRules.Get(mesh_el.GetGeomType(), 8);
   }

   int ndof = mesh_el.GetDof();
   int dim = mesh_el.GetDim();
   elvect.SetSize(ndof*dim);
   elvect = 0.0;

   DenseMatrix PointMat_bar(dim, ndof);
   
   // cast the ElementTransformation
   IsoparametricTransformation &isotrans =
                     dynamic_cast<IsoparametricTransformation&>(mesh_trans);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      PointMat_bar = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(i);

      isotrans.SetIntPoint(&ip);

      double Q_bar = 1.0;
      Q.EvalRevDiff(Q_bar, isotrans, ip, PointMat_bar);

      for (int j = 0; j < ndof ; ++j)
      {
         for (int d = 0; d < dim; ++d)
         {
            elvect(d * ndof + j) += ip.weight * PointMat_bar(d,j);
         }
      }
   }
}

void DomainResIntegrator::AssembleElementVector(const FiniteElement &elx,
                                                ElementTransformation &Trx,
                                                const Vector &elfunx,
                                                Vector &elvect)
{   
   /// get the proper element, transformation, and adjoint vector
   Array<int> vdofs;
   Vector psi;
   int element = Trx.ElementNo;
   const FiniteElement *el = adjoint->FESpace()->GetFE(element);
   ElementTransformation *Tr = adjoint->FESpace()->GetElementTransformation(element);
   adjoint->FESpace()->GetElementVDofs(element, vdofs);
   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = oa * el->GetOrder() + ob;
      ir = &IntRules.Get(el->GetGeomType(), order);
   }
   adjoint->GetSubVector(vdofs, psi);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    elvect.SetSize(dof*dim);
    elvect = 0.0;
    shape.SetSize(dofu);

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix PointMat_bar(dim, dof);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        PointMat_bar = 0.0;
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        el->CalcShape(ip, shape);
        double Weight_bar = Q.Eval(isotrans, ip) * (psi*shape); // dR/dWeight
        isotrans.WeightRevDiff(PointMat_bar); // dWeight/dX        
        PointMat_bar *= Weight_bar;

        /// Implement Q sensitivity
        double Q_bar = isotrans.Weight() * (psi * shape); //dR/dQ
        Q.EvalRevDiff(Q_bar, isotrans, ip, PointMat_bar);

        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d * dof + j) += ip.weight * PointMat_bar(d,j);
            }
        }
    }
}

void MassResIntegrator::AssembleElementVector(const FiniteElement &elx,
                                       ElementTransformation &Trx,
                                       const Vector &elfunx, Vector &elvect)
{
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj; 
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetSubVector(vdofs, elfun);
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    elvect.SetSize(dof*dim);
    elvect = 0.0;
    shape.SetSize(dofu);

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix elmat(dofu);
    DenseMatrix PointMat_bar(dim, dof);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        PointMat_bar = 0.0;

        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        double deriv = ip.weight; //dR/dWeight
        isotrans.WeightRevDiff(PointMat_bar);        
        el->CalcShape(ip, shape);
        if (Q)
        {
            deriv *= Q->Eval(*Tr, ip);
        }

        // perform deriv*(adj*shape)*(shape*elfun)
        double rw = deriv*(eladj*shape)*(shape*elfun);
        PointMat_bar.Set(rw, PointMat_bar); //dWeight/dX

        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d*dof + j) += PointMat_bar(d,j);
            }
        }
    }
}

void DiffusionResIntegrator::AssembleElementVector(const FiniteElement &elx,
                                                   ElementTransformation &Trx,
                                                   const Vector &elfunx,
                                                   Vector &elvect)
{
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj; 
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = NULL;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetSubVector(vdofs, elfun);
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    int spaceDim = Tr->GetSpaceDim();
    bool square = (dim == spaceDim);
    double tw, w, dw;
    Vector av(dim); Vector bv(dim);
    Vector ad(dim); Vector bd(dim);
    elvect.SetSize(dof*dim);
    elvect = 0.0;

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix elmat(dofu);
    DenseMatrix PointMat_bar(dim, dof);
    DenseMatrix jac_bar(dim);
    dshape.SetSize(dofu, dim);

    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        DenseMatrix K = Tr->AdjugateJacobian();
        PointMat_bar = 0.0;
        jac_bar = 0.0;

        el->CalcDShape(ip, dshape);
        tw = Tr->Weight();
        w = ip.weight / (square ? tw : tw*tw*tw);
        dw = -ip.weight / (square ? tw*tw : tw*tw*tw*tw/3);
        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        if (Q)
        {
            w *= Q->Eval(*Tr, ip);
            dw *= Q->Eval(*Tr, ip);
        }
        dshape.MultTranspose(eladj, ad); //D^T\psi
        K.MultTranspose(ad, av); //K^TD^T\psi
        dshape.MultTranspose(elfun, bd); //D^T\u
        K.MultTranspose(bd, bv); //K^TD^T\u

        // compute partials wrt weight
        double rw = dw*(av*bv);
        isotrans.WeightRevDiff(PointMat_bar);
        PointMat_bar.Set(rw, PointMat_bar);

        // compute partials wrt adjugate
        AddMult_a_VWt(w, ad, bv, jac_bar);
        AddMult_a_VWt(w, bd, av, jac_bar);
        isotrans.AdjugateJacobianRevDiff(jac_bar, PointMat_bar);
        
        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d*dof + j) += PointMat_bar(d,j);
            }
        }
    }
}

void DiffusionResIntegrator::AssembleRHSElementVect(
    const mfem::FiniteElement &elx,
    mfem::ElementTransformation &Trx,
    mfem::Vector &elvect)
{
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj; 
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = NULL;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetSubVector(vdofs, elfun);
    adjoint->GetSubVector(vdofs, eladj);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    const int dim = el->GetDim();
    int spaceDim = Tr->GetSpaceDim();
    bool square = (dim == spaceDim);
    double tw, w, dw;
    Vector av(dim); Vector bv(dim);
    Vector ad(dim); Vector bd(dim);
    elvect.SetSize(dof*dim);
    elvect = 0.0;

    // cast the ElementTransformation
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Tr);

    DenseMatrix elmat(dofu);
    DenseMatrix PointMat_bar(dim, dof);
    DenseMatrix jac_bar(dim);
    dshape.SetSize(dofu, dim);

    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);

        DenseMatrix K = Tr->AdjugateJacobian();
        PointMat_bar = 0.0;
        jac_bar = 0.0;

        el->CalcDShape(ip, dshape);
        tw = Tr->Weight();
        w = ip.weight / (square ? tw : tw*tw*tw);
        dw = -ip.weight / (square ? tw*tw : tw*tw*tw*tw/3);
        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        if (Q)
        {
            w *= Q->Eval(*Tr, ip);
            dw *= Q->Eval(*Tr, ip);
        }
        dshape.MultTranspose(eladj, ad); //D^T\psi
        K.MultTranspose(ad, av); //K^TD^T\psi
        dshape.MultTranspose(elfun, bd); //D^T\u
        K.MultTranspose(bd, bv); //K^TD^T\u

        // compute partials wrt weight
        double rw = dw*(av*bv);
        isotrans.WeightRevDiff(PointMat_bar);
        PointMat_bar.Set(rw, PointMat_bar);

        // compute partials wrt adjugate
        AddMult_a_VWt(w, ad, bv, jac_bar);
        AddMult_a_VWt(w, bd, av, jac_bar);
        isotrans.AdjugateJacobianRevDiff(jac_bar, PointMat_bar);
        
        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < dim; ++d)
            {
                elvect(d*dof + j) += PointMat_bar(d,j);
            }
        }
    }
}

void BoundaryNormalResIntegrator::AssembleRHSElementVect(
   const FiniteElement &elx,
   FaceElementTransformations &Trx,
   Vector &elvect)
{   
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.Elem1No;
    const FiniteElementCollection *fec = state->FESpace()->FEColl();
    const FiniteElement *el = state->FESpace()->GetFE(element);

    const int dof = elx.GetDof();
    const int dofu = el->GetDof();
    // const int dim = Trx.Face->GetDimension();
    int space_dim = Trx.Face->GetSpaceDim();
    shape.SetSize(dofu);
    elvect.SetSize(space_dim*dof);
    elvect = 0.0;

    // get the right boundary element
    const FiniteElement *el_bnd;
    switch (space_dim)
    {
        case 1: el_bnd = fec->FiniteElementForGeometry(Geometry::POINT);
              break;
        case 2: el_bnd = fec->FiniteElementForGeometry(Geometry::SEGMENT);
              break;
        case 3:
            if(Trx.Elem1->GetGeometryType() == Geometry::TETRAHEDRON)
            {
                el_bnd = fec->FiniteElementForGeometry(Geometry::TRIANGLE);
            }
            if(Trx.Elem1->GetGeometryType() == Geometry::CUBE)
            {
                el_bnd = fec->FiniteElementForGeometry(Geometry::SQUARE);
            } 
             break;
    }
    ElementTransformation *Tr_bnd = Trx.Face;
    ElementTransformation *Tr = Trx.Elem1;
    
    //boundary element integration rule
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el_bnd->GetGeomType(), oa * el_bnd->GetOrder() + ob);
    }

    state->FESpace()->GetElementVDofs(element, vdofs);
    state->GetSubVector(vdofs, elfun); //don't need this one
    adjoint->GetSubVector(vdofs, eladj);

    // cast the ElementTransformation (for the domain element)
    IsoparametricTransformation &isotrans =
    dynamic_cast<IsoparametricTransformation&>(*Trx.Elem1);

    DenseMatrix PointMat_bar(space_dim, dof);
    DenseMatrix R;
    Vector Qvec(space_dim);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        //compute dR/dnor*dnor/dJ_bnd*dJ_bnd/dJ_el*dJ_el/dX

        //get corresponding element integration point
        const IntegrationPoint &ip = ir->IntPoint(i);
        IntegrationPoint eip;
        Trx.Loc1.Transform(ip, eip);
        Tr_bnd->SetIntPoint(&ip);
        Tr->SetIntPoint(&eip);

        //get el to bnd jacobian transformation
        DenseMatrix J = Tr_bnd->Jacobian();
        DenseMatrix Jinv_el = Tr->InverseJacobian();
        R.SetSize(Jinv_el.Height(), J.Width());
        Mult(Jinv_el, J, R);
        DenseMatrix J_bar(J.Height(), J.Width());
        DenseMatrix J_bar_el(space_dim, space_dim);

        PointMat_bar = 0.0;

        /// NOTE: Q may or may not have sensitivity to x. Need to tailor to
        /// different coefficients 
        el->CalcShape(eip, shape);
        Q.Eval(Qvec, *Tr_bnd, ip);
        Vector nor_bar(Qvec.Size()); 
        for (int p = 0; p < nor_bar.Size(); ++p) //dR/dnor
        {
            nor_bar(p) = ip.weight*Qvec(p)*(eladj*shape); 
        }
        CalcOrthoRevDiff(J, nor_bar, J_bar); //dnor/dJbnd

        //convert face jacobian bar to element jacobian bar by inverting
        MultABt(J_bar, R, J_bar_el); //dJbnd/dJel

        isotrans.JacobianRevDiff(J_bar_el, PointMat_bar); //dJel/dX

        for (int j = 0; j < dof ; ++j)
        {
            for (int d = 0; d < space_dim; ++d)
            {
                elvect(d* dof + j) += PointMat_bar (d,j);
            }
        }
    }
}


#if 0
double DomainResIntegrator::GetElementEnergy(const FiniteElement &elx,
                                             ElementTransformation &Trx,
                                             const Vector &elfunx)
{
    double Rpart = 0;
    
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), oa * el->GetOrder() + ob);
    }
    state->GetValues(element, *ir, elfun); //don't need this one
    adjoint->GetValues(element, *ir, eladj);

    const int dof = el->GetDof();
    const int dim = el->GetDim();
    
    Vector x_q(dim);
    DenseMatrix Jac_q(dim, dim);
    
    // loop through nodes
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        Tr->SetIntPoint(&ip);
        Tr->Transform(ip, x_q);
        Jac_q = Tr->Jacobian();
        double r_q = Tr->Weight()*Q.Eval(*Tr, ip);
        //double r_q = calcFunctional(Tr->ElementNo,
        //                             ip, x_q, Tr, Jac_q);
        
        //skipping shape function step
        Rpart += ip.weight*r_q;
    }

   return Rpart;
}

double MassResIntegrator::GetElementEnergy(const FiniteElement &elx,
                                           ElementTransformation &Trx,
                                           const Vector &elfunx)
{
    double Rpart = 0;
    
    /// get the proper element, transformation, and state vector
    Array<int> vdofs; Vector elfun; Vector eladj;
    int element = Trx.ElementNo;
    const FiniteElement *el = state->FESpace()->GetFE(element);
    ElementTransformation *Tr = state->FESpace()->GetElementTransformation(element);
    state->FESpace()->GetElementVDofs(element, vdofs);
    int order = 2*el->GetOrder() + Tr->OrderW();
    const IntegrationRule *ir = IntRule;
    if (ir == NULL)
    {
        ir = &IntRules.Get(el->GetGeomType(), order);
    }
    state->GetValues(element, *ir, elfun); 
    adjoint->GetValues(element, *ir, eladj);

    Array<int> dofs;
    const int dof = el->GetDof();
    const int dim = el->GetDim();
    
    Vector x_q(dim);
    Vector rvect(dof);
    DenseMatrix Jac_q(dim, dim);
    DenseMatrix elmat(dof);
    
    // assemble the matrix
    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); ++i)
    {
        const IntegrationPoint &ip = ir->IntPoint(i);
        el->CalcShape(ip, shape);

        Tr->SetIntPoint (&ip);
        double w = Tr->Weight() * ip.weight;
        if (Q)
        {
            w *= Q->Eval(*Tr, ip);
        }

        AddMult_a_VVt(w, shape, elmat);
    }

    elmat.Mult(elfun, rvect);

    return rvect.Sum();
}

#endif

}


       