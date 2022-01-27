#include "mach_input.hpp"

#include "l2_transfer_operator.hpp"

namespace
{
void identity_operator(const mfem::FiniteElement &state_fe,
                       const mfem::FiniteElement &output_fe,
                       mfem::ElementTransformation &trans,
                       const mfem::Vector &el_state,
                       mfem::Vector &el_output)
{
   int state_dof = state_fe.GetDof();
   int output_dof = output_fe.GetDof();

   int space_dim = trans.GetSpaceDim();

   mfem::DenseMatrix vshape(state_dof, space_dim);

   double shape_vec_buffer[3];
   mfem::Vector shape_vec(shape_vec_buffer, space_dim);

   mfem::DenseMatrix output(el_output.GetData(), output_dof, space_dim);

   const auto &ir = output_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcVShape(trans, vshape);
      vshape.MultTranspose(el_state, shape_vec);

      for (int j = 0; j < space_dim; j++)
      {
         output(i, j) = shape_vec(j);
      }
   }
}

void identity_operator_state_bar(const mfem::FiniteElement &state_fe,
                                 const mfem::FiniteElement &output_fe,
                                 mfem::ElementTransformation &trans,
                                 const mfem::Vector &el_output_adj,
                                 const mfem::Vector &el_state,
                                 mfem::Vector &el_state_bar)
{
   int state_dof = state_fe.GetDof();
   int output_dof = output_fe.GetDof();

   int space_dim = trans.GetSpaceDim();

   mfem::Vector shape(output_dof);
   mfem::DenseMatrix vshape(state_dof, space_dim);

   double state_vec_buffer[3];
   mfem::Vector state_vec(state_vec_buffer, space_dim);
   double output_adj_vec_buffer[3];
   mfem::Vector output_adj_vec(output_adj_vec_buffer, space_dim);

   mfem::DenseMatrix output_adj(el_output_adj.GetData(), output_dof, space_dim);

   const auto &ir = output_fe.GetNodes();
   el_state_bar = 0.0;
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcVShape(trans, vshape);
      vshape.MultTranspose(el_state, state_vec);

      output_fe.CalcPhysShape(trans, shape);
      output_adj.MultTranspose(shape, output_adj_vec);

      /// dummy functional for adjoint-weighted residual
      // double fun = output_adj_vec * state_vec;

      /// start reverse pass
      double fun_bar = 1.0;

      /// double fun = output_adj_vec * state_vec;

      /// only need state derivative
      // double output_adj_vec_bar_buffer[3];
      // mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer, space_dim);
      // output_adj_vec_bar = 0.0;
      // add(output_adj_vec_bar, fun_bar, state_vec, output_adj_vec_bar);

      double state_vec_bar_buffer[3];
      mfem::Vector state_vec_bar(state_vec_bar_buffer, space_dim);
      state_vec_bar = 0.0;

      add(state_vec_bar, fun_bar, output_adj_vec, state_vec_bar);

      /// only need state derivative
      /// output_adj.MultTranspose(shape, output_adj_vec);
      /// output_fe.CalcPhysShape(trans, shape);

      /// vshape.MultTranspose(el_state, state_vec);
      vshape.AddMult(state_vec_bar, el_state_bar);

      /// only need state derivative
      /// state_fe.CalcVShape(trans, vshape);
   }
}

void curl_operator(const mfem::FiniteElement &state_fe,
                   const mfem::FiniteElement &output_fe,
                   mfem::ElementTransformation &trans,
                   const mfem::Vector &el_state,
                   mfem::Vector &el_output)
{
   int state_dof = state_fe.GetDof();
   int output_dof = output_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(state_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(state_dof, curl_dim);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

   mfem::DenseMatrix output(el_output.GetData(), output_dof, curl_dim);

   const auto &ir = output_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_state, curl_vec);

      curl_vec /= trans.Weight();

      for (int j = 0; j < curl_dim; j++)
      {
         output(i, j) = curl_vec(j);
      }
   }
}

void curl_operator_state_bar(const mfem::FiniteElement &state_fe,
                             const mfem::FiniteElement &output_fe,
                             mfem::ElementTransformation &trans,
                             const mfem::Vector &el_output_adj,
                             const mfem::Vector &el_state,
                             mfem::Vector &el_state_bar)
{
   int state_dof = state_fe.GetDof();
   int output_dof = output_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(state_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(state_dof, curl_dim);
   mfem::Vector shape(output_dof);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);
   double output_adj_vec_buffer[3];
   mfem::Vector output_adj_vec(output_adj_vec_buffer, curl_dim);

   mfem::DenseMatrix output_adj(el_output_adj.GetData(), output_dof, curl_dim);

   const auto &ir = output_fe.GetNodes();
   el_state_bar = 0.0;
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_state, curl_vec);

      curl_vec /= trans.Weight();

      output_fe.CalcPhysShape(trans, shape);
      output_adj.MultTranspose(shape, output_adj_vec);

      /// dummy functional for adjoint-weighted residual
      // double fun = output_adj_vec * curl_vec;

      /// start reverse pass
      double fun_bar = 1.0;

      /// double fun = output_adj_vec * curl_vec;
      /// only need state derivative
      // double output_adj_vec_bar_buffer[3];
      // mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer, space_dim);
      // output_adj_vec_bar = 0.0;
      // add(output_adj_vec_bar, fun_bar, curl_vec, output_adj_vec_bar);
      double curl_vec_bar_buffer[3];
      mfem::Vector curl_vec_bar(curl_vec_bar_buffer, space_dim);
      curl_vec_bar = 0.0;
      add(curl_vec_bar, fun_bar, output_adj_vec, curl_vec_bar);

      /// only need state derivative
      /// output_adj.MultTranspose(shape, output_adj_vec);
      /// output_fe.CalcPhysShape(trans, shape);

      /// only need state derivative
      /// curl_vec /= trans.Weight();
      curl_vec_bar /= trans.Weight();

      /// curlshape_dFt.MultTranspose(el_state, curl_vec);
      curlshape_dFt.AddMult(curl_vec_bar, el_state_bar);

      /// only need state derivative
      /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);

      /// only need state derivative
      /// state_fe.CalcVShape(trans, vshape);
   }
}

void curl_magnitude_operator(const mfem::FiniteElement &state_fe,
                             const mfem::FiniteElement &output_fe,
                             mfem::ElementTransformation &trans,
                             const mfem::Vector &el_state,
                             mfem::Vector &el_output)
{
   int state_dof = state_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(state_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(state_dof, curl_dim);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

   const auto &ir = output_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_state, curl_vec);

      const double curl_vec_norm = curl_vec.Norml2();
      const double curl_mag = curl_vec_norm / trans.Weight();
      el_output(i) = curl_mag;
   }
}

void curl_magnitude_operator_state_bar(const mfem::FiniteElement &state_fe,
                                       const mfem::FiniteElement &output_fe,
                                       mfem::ElementTransformation &trans,
                                       const mfem::Vector &el_output_adj,
                                       const mfem::Vector &el_state,
                                       mfem::Vector &el_state_bar)
{ }

}  // anonymous namespace

namespace mach
{

L2IdentityProjection::L2IdentityProjection(FiniteElementState &state,
                                           FiniteElementState &output)
 : L2TransferOperator(state,
                      output,
                      identity_operator,
                      identity_operator_state_bar)
{ }

L2CurlProjection::L2CurlProjection(FiniteElementState &state,
                                   FiniteElementState &output)
 : L2TransferOperator(state, output, curl_operator, curl_operator_state_bar)
{ }

L2CurlMagnitudeProjection::L2CurlMagnitudeProjection(FiniteElementState &state,
                                                     FiniteElementState &output)
 : L2TransferOperator(state,
                      output,
                      curl_magnitude_operator,
                      curl_magnitude_operator_state_bar)
{ }

void L2TransferOperator::apply(const MachInputs &inputs, mfem::Vector &out_vec)
{
   out_vec = 0.0;
   output.gridFunc() = 0.0;
   if (!operation)
   {
      return;
   }

   mfem::Vector state_tv;
   setVectorFromInputs(inputs, "state", state_tv, false, true);

   state.distributeSharedDofs(state_tv);

   const auto &state_fes = state.space();
   const auto &output_fes = output.space();
   mfem::Array<int> state_vdofs;
   mfem::Array<int> output_vdofs;
   mfem::Vector el_state;
   mfem::Vector el_output;

   for (int i = 0; i < output_fes.GetNE(); ++i)
   {
      const auto &state_fe = *state_fes.GetFE(i);
      const auto &output_fe = *output_fes.GetFE(i);
      auto &trans = *output_fes.GetElementTransformation(i);

      auto *state_dof_trans = state_fes.GetElementVDofs(i, state_vdofs);
      el_state.SetSize(state_vdofs.Size());
      auto *output_dof_trans = output_fes.GetElementVDofs(i, output_vdofs);
      el_output.SetSize(output_vdofs.Size());

      state.gridFunc().GetSubVector(state_vdofs, el_state);
      if (state_dof_trans)
      {
         state_dof_trans->InvTransformPrimal(el_state);
      }

      /// apply the operation
      operation(state_fe, output_fe, trans, el_state, el_output);

      if (output_dof_trans)
      {
         output_dof_trans->TransformPrimal(el_output);
      }
      output.gridFunc().AddElementVector(output_vdofs, el_output);
   }

   output.setTrueVec(out_vec);
}

void L2TransferOperator::vectorJacobianProduct(const std::string &wrt,
                                               const MachInputs &inputs,
                                               const mfem::Vector &out_bar,
                                               mfem::Vector &wrt_bar)
{
   if (wrt == "out")
   {
      wrt_bar -= out_bar;
   }
   else if (wrt == "state")
   {
      output_adjoint.distributeSharedDofs(out_bar);

      mfem::Vector state_tv;
      setVectorFromInputs(inputs, "state", state_tv, false, true);

      state.distributeSharedDofs(state_tv);

      const auto &state_fes = state.space();
      const auto &output_fes = output.space();
      mfem::Array<int> state_vdofs;
      mfem::Array<int> output_adj_vdofs;
      mfem::Vector el_state;
      mfem::Vector el_output_adj;
      mfem::Vector el_state_bar;

      for (int i = 0; i < state_fes.GetNE(); ++i)
      {
         const auto &state_fe = *state_fes.GetFE(i);
         const auto &output_fe = *output_fes.GetFE(i);
         auto &trans = *output_fes.GetElementTransformation(i);

         auto *output_adj_dof_trans =
             output_fes.GetElementVDofs(i, output_adj_vdofs);
         el_output_adj.SetSize(output_adj_vdofs.Size());
         auto *state_dof_trans = state_fes.GetElementVDofs(i, state_vdofs);
         el_state.SetSize(state_vdofs.Size());
         el_state_bar.SetSize(state_vdofs.Size());

         state.gridFunc().GetSubVector(state_vdofs, el_state);
         if (state_dof_trans)
         {
            state_dof_trans->InvTransformPrimal(el_state);
         }
         output_adjoint.gridFunc().GetSubVector(output_adj_vdofs,
                                                el_output_adj);
         if (output_adj_dof_trans)
         {
            output_adj_dof_trans->InvTransformPrimal(el_output_adj);
         }

         /// apply the reverse mode differentiated operation
         operation_state_bar(
             state_fe, output_fe, trans, el_output_adj, el_state, el_state_bar);

         if (state_dof_trans)
         {
            state_dof_trans->TransformDual(el_state_bar);
         }
         state_bar.localVec().AddElementVector(state_vdofs, el_state_bar);
      }

      /// this should maybe accumulate into wrt_bar
      state_bar.setTrueVec(wrt_bar);
   }
}

}  // namespace mach
