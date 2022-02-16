#include <iomanip>

#include "mach_input.hpp"

#include "l2_transfer_operator.hpp"

namespace
{
void scalar_identity_operator(const mfem::FiniteElement &state_fe,
                              const mfem::FiniteElement &output_fe,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &el_state,
                              mfem::Vector &el_output)
{
   int state_dof = state_fe.GetDof();
   mfem::Vector shape(state_dof);

   const auto &ir = output_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcPhysShape(trans, shape);
      el_output(i) = shape * el_state;
   }
}

void scalar_identity_operator_state_bar(const mfem::FiniteElement &state_fe,
                                        const mfem::FiniteElement &output_fe,
                                        mfem::ElementTransformation &trans,
                                        const mfem::Vector &el_output_adj,
                                        const mfem::Vector &el_state,
                                        mfem::Vector &el_state_bar)
{
   int state_dof = state_fe.GetDof();
   int output_dof = output_fe.GetDof();

   mfem::Vector shape(state_dof);
   mfem::Vector adj_shape(output_dof);

   const auto &ir = output_fe.GetNodes();
   el_state_bar = 0.0;
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcPhysShape(trans, shape);

      output_fe.CalcPhysShape(trans, adj_shape);
      double adj = el_output_adj * adj_shape;

      /// dummy functional
      /// double fun = adj * state;
      double fun_bar = 1.0;
      double state_bar = fun_bar * adj;

      /// double state = shape * el_state;
      add(el_state_bar, state_bar, shape, el_state_bar);
   }
   el_state_bar.Print(std::cout, 9);
   // for (int j = 0; j < state_fe.GetNodes().GetNPoints(); ++j)
   // {
   //    if (abs(el_state_bar(j) - 0.243865) < 1e-6)
   //    {
   //       std::cout << "target\n";
   //    }
   // }
}
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

   const auto &ir = output_fe.GetNodes();
   el_state_bar = 0.0;
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      state_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_state, curl_vec);

      const double curl_vec_norm = curl_vec.Norml2();
      // const double curl_mag = curl_vec_norm / trans.Weight();

      output_fe.CalcPhysShape(trans, shape);

      double output_adj = el_output_adj * shape;

      /// dummy functional for adjoint-weighted residual
      // double fun = output_adj * curl_mag;

      /// start reverse pass
      double fun_bar = 1.0;

      /// double fun = output_adj * curl_mag;
      /// only need state derivative
      // double output_adj_bar = fun_bar * curl_mag;
      double curl_mag_bar = fun_bar * output_adj;

      /// const double curl_mag = curl_vec_norm / trans.Weight();
      double curl_vec_norm_bar = curl_mag_bar / trans.Weight();

      /// const double curl_vec_norm = curl_vec.Norml2();
      double curl_vec_bar_buffer[3];
      mfem::Vector curl_vec_bar(curl_vec_bar_buffer, space_dim);
      curl_vec_bar = 0.0;
      add(curl_vec_bar,
          curl_vec_norm_bar / curl_vec_norm,
          curl_vec,
          curl_vec_bar);

      /// only need state derivative
      // double output_adj_vec_bar_buffer[3];
      // mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer, space_dim);
      // output_adj_vec_bar = 0.0;
      // add(output_adj_vec_bar, fun_bar, curl_vec, output_adj_vec_bar);

      /// only need state derivative
      /// output_adj.MultTranspose(shape, output_adj_vec);
      /// output_fe.CalcPhysShape(trans, shape);

      /// curlshape_dFt.MultTranspose(el_state, curl_vec);
      curlshape_dFt.AddMult(curl_vec_bar, el_state_bar);

      /// only need state derivative
      /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);

      /// only need state derivative
      /// state_fe.CalcVShape(trans, vshape);
   }
}

}  // anonymous namespace

namespace mach
{
ScalarL2IdentityProjection::ScalarL2IdentityProjection(
    FiniteElementState &state,
    FiniteElementState &output)
 : L2TransferOperator(state,
                      output,
                      scalar_identity_operator,
                      scalar_identity_operator_state_bar)
{ }

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

   // output.setTrueVec(out_vec);
   output.gridFunc().ParallelAssemble(out_vec);

   auto nranks = state.space().GetNRanks();
   auto rank = state.space().GetMyRank();
   std::cout.precision(16);
   for (int i = 0; i < nranks; ++i)
   {
      if (rank == i)
      {
         std::cout << "\nrank: " << i << "\n";

         std::cout << "\nstate:\n";
         state_tv.Print(std::cout, 100);
         std::cout << "\nout_vec:\n";
         out_vec.Print(std::cout, 100);
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }

   /// Print fields
   mfem::ParaViewDataCollection pv("DG_transfer_outout", &output.mesh());
   pv.SetPrefixPath("ParaView");
   pv.SetLevelsOfDetail(3);
   pv.SetDataFormat(mfem::VTKFormat::ASCII);
   pv.SetHighOrderOutput(true);
   pv.RegisterField("DG_field", &output.gridFunc());
   pv.Save();
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
      // output_adjoint.space().GetRestrictionOperator()->MultTranspose(out_bar, output_adjoint.gridFunc());

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

      int rank, nranks;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nranks);
      for (int i = 0; i < nranks; ++i)
      {
         if (rank == i)
         {
            std::cout << "rank " << i << ":\n";

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
         }
         MPI_Barrier(MPI_COMM_WORLD);
      }

      // state_bar.localVec() = state_bar.space().GetMyRank();
      // state_bar.localVec() = 1.0;

      // auto &grid_func =
      //     dynamic_cast<mfem::ParGridFunction &>(state_bar.localVec());
      // grid_func.ExchangeFaceNbrData();

      /// this should maybe accumulate into wrt_bar
      state_bar.setTrueVec(wrt_bar);

      std::cout.precision(4);
      for (int i = 0; i < nranks; ++i)
      {
         if (rank == i)
         {
            std::cout << "rank " << i << ":\n";
            for (int i = 0; i < state_bar.space().GetVSize(); ++i)
            {
               double lval = state_bar.localVec()(i);
               auto gtdof = state_bar.space().GetGlobalTDofNumber(i);
               auto ltdof = state_bar.space().GetLocalTDofNumber(i);
               std::cout << "ldof: ";
               std::cout << std::left << std::setw(3) << std::setfill(' ') << i;
               std::cout << std::left << std::setw(6) << std::setfill(' ') << " lval: ";
               std::cout << std::left << std::setw(6) << std::setfill(' ') << lval;
               std::cout << std::left << std::setw(3) << std::setfill(' ') << " tdof: ";
               std::cout << std::left << std::setw(6) << std::setfill(' ') << gtdof;
               if (ltdof >= 0)
               {
                  std::cout << std::left << std::setw(6) << std::setfill(' ') << " tval: ";
                  std::cout << std::left << std::setw(6) << std::setfill(' ') << wrt_bar(ltdof);
               }
               std::cout << "\n";
            }
         }
         MPI_Barrier(MPI_COMM_WORLD);
      }
      mfem::ParGridFunction gf2(&state_bar.space());
      gf2 = state_bar.space().GetMyRank();
      mfem::Vector tv(wrt_bar.Size());
      gf2.GetTrueDofs(tv);
      gf2.Distribute(tv);

      mfem::ParGridFunction gf(&state_bar.space());
      gf.Distribute(wrt_bar);

      /// Print fields
      mfem::ParaViewDataCollection pv("state_bar", &state_bar.mesh());
      pv.SetPrefixPath("ParaView");
      pv.SetLevelsOfDetail(2*state_bar.space().GetElementOrder(0));
      pv.SetDataFormat(mfem::VTKFormat::ASCII);
      pv.SetHighOrderOutput(true);
      // pv.RegisterField(
      //     "state_bar",
      //     dynamic_cast<mfem::ParGridFunction *>(&state_bar.localVec()));
      pv.RegisterField("state_bar", &gf);
      pv.RegisterField("state_bar_2", &gf2);
      pv.RegisterField("output_adj", &output_adjoint.gridFunc());
      pv.Save();

      // /// this should maybe accumulate into wrt_bar
      // state_bar.setTrueVec(wrt_bar);
   }
}

}  // namespace mach
