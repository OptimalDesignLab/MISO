#include <iomanip>

#include "mach_input.hpp"

#include "l2_transfer_operator.hpp"

namespace
{
class ScalarIdentityOperator : public mach::L2TransferOperation
{
public:
   void apply(const mfem::FiniteElement &state_fe,
              const mfem::FiniteElement &output_fe,
              mfem::ElementTransformation &trans,
              const mfem::Vector &el_state,
              mfem::Vector &el_output) const override
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

   void apply_state_bar(const mfem::FiniteElement &state_fe,
                        const mfem::FiniteElement &output_fe,
                        mfem::ElementTransformation &trans,
                        const mfem::Vector &el_output_adj,
                        const mfem::Vector &el_state,
                        mfem::Vector &el_state_bar) const override
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
   }

   void apply_mesh_coords_bar(const mfem::FiniteElement &state_fe,
                              const mfem::FiniteElement &output_fe,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &el_output_adj,
                              const mfem::Vector &el_state,
                              mfem::Vector &mesh_coords_bar) const override
   {
      auto &isotrans = dynamic_cast<mfem::IsoparametricTransformation &>(trans);
      auto &mesh_fe = *isotrans.GetFE();
      int space_dim = isotrans.GetSpaceDim();

      int mesh_dof = mesh_fe.GetDof();
      int state_dof = state_fe.GetDof();
      int output_dof = output_fe.GetDof();

      mfem::Vector shape(state_dof);
      mfem::Vector adj_shape(output_dof);

      mfem::Vector adj_shape_bar(output_dof);
      mfem::Vector shape_bar(state_dof);

      mfem::DenseMatrix PointMat_bar(space_dim, mesh_dof);
      mesh_coords_bar.SetSize(mesh_dof * space_dim);
      mesh_coords_bar = 0.0;

      const auto &ir = output_fe.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const auto &ip = ir.IntPoint(i);
         trans.SetIntPoint(&ip);

         state_fe.CalcPhysShape(trans, shape);
         double state = shape * el_state;

         output_fe.CalcPhysShape(trans, adj_shape);
         double adj = el_output_adj * adj_shape;

         /// dummy functional
         /// double fun = adj * state;
         double fun_bar = 1.0;
         double state_bar = fun_bar * adj;
         double adj_bar = fun_bar * state;

         /// double adj = el_output_adj * adj_shape;
         adj_shape_bar = 0.0;
         add(adj_shape_bar, adj_bar, el_output_adj, adj_shape_bar);

         /// output_fe.CalcPhysShape(trans, adj_shape);
         PointMat_bar = 0.0;
         output_fe.CalcPhysShapeRevDiff(trans, adj_shape_bar, PointMat_bar);

         /// double state = shape * el_state;
         shape_bar = 0.0;
         add(shape_bar, state_bar, el_state, shape_bar);

         /// state_fe.CalcPhysShape(trans, shape);
         state_fe.CalcPhysShapeRevDiff(trans, shape_bar, PointMat_bar);

         /// insert PointMat_bar into mesh_coords_bar
         for (int j = 0; j < mesh_dof; ++j)
         {
            for (int d = 0; d < space_dim; ++d)
            {
               mesh_coords_bar(d * mesh_dof + j) += PointMat_bar(d, j);
            }
         }
      }
   }
};

class IdentityOperator : public mach::L2TransferOperation
{
public:
   void apply(const mfem::FiniteElement &state_fe,
              const mfem::FiniteElement &output_fe,
              mfem::ElementTransformation &trans,
              const mfem::Vector &el_state,
              mfem::Vector &el_output) const override
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

   void apply_state_bar(const mfem::FiniteElement &state_fe,
                        const mfem::FiniteElement &output_fe,
                        mfem::ElementTransformation &trans,
                        const mfem::Vector &el_output_adj,
                        const mfem::Vector &el_state,
                        mfem::Vector &el_state_bar) const override
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

      mfem::DenseMatrix output_adj(
          el_output_adj.GetData(), output_dof, space_dim);

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
         // mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer,
         // space_dim); output_adj_vec_bar = 0.0; add(output_adj_vec_bar,
         // fun_bar, state_vec, output_adj_vec_bar);

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

   void apply_mesh_coords_bar(const mfem::FiniteElement &state_fe,
                              const mfem::FiniteElement &output_fe,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &el_output_adj,
                              const mfem::Vector &el_state,
                              mfem::Vector &mesh_coords_bar) const override
   {
      auto &isotrans = dynamic_cast<mfem::IsoparametricTransformation &>(trans);
      auto &mesh_fe = *isotrans.GetFE();
      int space_dim = isotrans.GetSpaceDim();

      int mesh_dof = mesh_fe.GetDof();
      int state_dof = state_fe.GetDof();
      int output_dof = output_fe.GetDof();

      mfem::DenseMatrix vshape(state_dof, space_dim);
      double state_vec_buffer[3];
      mfem::Vector state_vec(state_vec_buffer, space_dim);

      mfem::Vector adj_shape(output_dof);
      double output_adj_vec_buffer[3];
      mfem::Vector output_adj_vec(output_adj_vec_buffer, space_dim);

      mfem::DenseMatrix output_adj(
          el_output_adj.GetData(), output_dof, space_dim);

      double output_adj_vec_bar_buffer[3];
      mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer, space_dim);

      double state_vec_bar_buffer[3];
      mfem::Vector state_vec_bar(state_vec_bar_buffer, space_dim);

      mfem::Vector adj_shape_bar(output_dof);
      mfem::DenseMatrix vshape_bar(state_dof, space_dim);

      mesh_coords_bar.SetSize(mesh_dof * space_dim);
      mesh_coords_bar = 0.0;
      mfem::DenseMatrix PointMat_bar(space_dim, mesh_dof);
      const auto &ir = output_fe.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const auto &ip = ir.IntPoint(i);
         trans.SetIntPoint(&ip);

         state_fe.CalcVShape(trans, vshape);
         vshape.MultTranspose(el_state, state_vec);

         output_fe.CalcPhysShape(trans, adj_shape);
         output_adj.MultTranspose(adj_shape, output_adj_vec);

         /// dummy functional for adjoint-weighted residual
         // double fun = output_adj_vec * state_vec;

         /// start reverse pass
         double fun_bar = 1.0;

         /// double fun = output_adj_vec * state_vec;

         output_adj_vec_bar = 0.0;
         add(output_adj_vec_bar, fun_bar, state_vec, output_adj_vec_bar);

         state_vec_bar = 0.0;
         add(state_vec_bar, fun_bar, output_adj_vec, state_vec_bar);

         /// output_adj.MultTranspose(adj_shape, output_adj_vec);
         output_adj.AddMult(output_adj_vec_bar, adj_shape_bar);

         /// output_fe.CalcPhysShape(trans, adj_shape);
         PointMat_bar = 0.0;
         output_fe.CalcPhysShapeRevDiff(trans, adj_shape_bar, PointMat_bar);

         /// vshape.MultTranspose(el_state, state_vec);
         vshape_bar = 0.0;
         AddMultVWt(el_state, state_vec_bar, vshape_bar);

         /// state_fe.CalcVShape(trans, vshape);
         state_fe.CalcVShapeRevDiff(trans, vshape_bar, PointMat_bar);

         /// insert PointMat_bar into mesh_coords_bar
         for (int j = 0; j < mesh_dof; ++j)
         {
            for (int d = 0; d < space_dim; ++d)
            {
               mesh_coords_bar(d * mesh_dof + j) += PointMat_bar(d, j);
            }
         }
      }
   }
};

class CurlOperator : public mach::L2TransferOperation
{
public:
   void apply(const mfem::FiniteElement &state_fe,
              const mfem::FiniteElement &output_fe,
              mfem::ElementTransformation &trans,
              const mfem::Vector &el_state,
              mfem::Vector &el_output) const override
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

   void apply_state_bar(const mfem::FiniteElement &state_fe,
                        const mfem::FiniteElement &output_fe,
                        mfem::ElementTransformation &trans,
                        const mfem::Vector &el_output_adj,
                        const mfem::Vector &el_state,
                        mfem::Vector &el_state_bar) const override
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

      mfem::DenseMatrix output_adj(
          el_output_adj.GetData(), output_dof, curl_dim);

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
         // mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer,
         // space_dim); output_adj_vec_bar = 0.0; add(output_adj_vec_bar,
         // fun_bar, curl_vec, output_adj_vec_bar);
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

   void apply_mesh_coords_bar(const mfem::FiniteElement &state_fe,
                              const mfem::FiniteElement &output_fe,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &el_output_adj,
                              const mfem::Vector &el_state,
                              mfem::Vector &mesh_coords_bar) const override
   {
      auto &isotrans = dynamic_cast<mfem::IsoparametricTransformation &>(trans);
      auto &mesh_fe = *isotrans.GetFE();
      int space_dim = isotrans.GetSpaceDim();
      int curl_dim = space_dim == 3 ? 3 : 1;

      int mesh_dof = mesh_fe.GetDof();
      int state_dof = state_fe.GetDof();
      int output_dof = output_fe.GetDof();

      mfem::DenseMatrix curlshape(state_dof, curl_dim);
      mfem::DenseMatrix curlshape_dFt(state_dof, curl_dim);
      mfem::Vector adj_shape(output_dof);

      double curl_vec_buffer[3];
      mfem::Vector curl_vec(curl_vec_buffer, curl_dim);
      double output_adj_vec_buffer[3];
      mfem::Vector output_adj_vec(output_adj_vec_buffer, curl_dim);

      mfem::DenseMatrix output_adj(
          el_output_adj.GetData(), output_dof, curl_dim);

      double output_adj_vec_bar_buffer[3];
      mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer, space_dim);

      double curl_vec_bar_buffer[3];
      mfem::Vector curl_vec_bar(curl_vec_bar_buffer, space_dim);

      mfem::Vector adj_shape_bar(output_dof);

      mfem::DenseMatrix curlshape_dFt_bar(curl_dim, state_dof);

      mesh_coords_bar.SetSize(mesh_dof * space_dim);
      mesh_coords_bar = 0.0;
      mfem::DenseMatrix PointMat_bar(space_dim, mesh_dof);
      const auto &ir = output_fe.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const auto &ip = ir.IntPoint(i);
         trans.SetIntPoint(&ip);

         state_fe.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
         curlshape_dFt.MultTranspose(el_state, curl_vec);

         output_fe.CalcPhysShape(trans, adj_shape);
         output_adj.MultTranspose(adj_shape, output_adj_vec);

         double trans_weight = trans.Weight();
         /// dummy functional for adjoint-weighted residual
         // double fun = output_adj_vec * curl_vec / trans_weight;

         /// start reverse pass
         double fun_bar = 1.0;
         /// double fun = output_adj_vec * curl_vec / trans_weight;
         output_adj_vec_bar = 0.0;
         add(output_adj_vec_bar, fun_bar / trans_weight, curl_vec, output_adj_vec_bar);

         curl_vec_bar = 0.0;
         add(curl_vec_bar, fun_bar / trans_weight, output_adj_vec, curl_vec_bar);

         double trans_weight_bar = 0.0;
         trans_weight_bar += -fun_bar * (output_adj_vec * curl_vec) / pow(trans_weight, 2);

         /// double trans_weight = trans.Weight();
         PointMat_bar = 0.0;
         isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

         /// only need state derivative
         /// output_adj.MultTranspose(adj_shape, output_adj_vec);
         output_adj.AddMult(output_adj_vec_bar, adj_shape_bar);

         /// output_fe.CalcPhysShape(trans, adj_shape);
         output_fe.CalcPhysShapeRevDiff(trans, adj_shape_bar, PointMat_bar);

         /// curlshape_dFt.MultTranspose(el_state, curl_vec);
         curlshape_dFt_bar = 0.0;
         AddMultVWt(curl_vec_bar, el_state, curlshape_dFt_bar);

         /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
         double jac_bar_buffer[9];
         mfem::DenseMatrix jac_bar(jac_bar_buffer, space_dim, space_dim);
         jac_bar = 0.0;
         AddMult(curlshape_dFt_bar, curlshape, jac_bar);
         isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

         /// state_fe.CalcCurlShape(ip, curlshape);

         /// insert PointMat_bar into mesh_coords_bar
         for (int j = 0; j < mesh_dof; ++j)
         {
            for (int d = 0; d < space_dim; ++d)
            {
               mesh_coords_bar(d * mesh_dof + j) += PointMat_bar(d, j);
            }
         }
      }
   }
};

class CurlMagnitudeOperator : public mach::L2TransferOperation
{
public:
   void apply(const mfem::FiniteElement &state_fe,
              const mfem::FiniteElement &output_fe,
              mfem::ElementTransformation &trans,
              const mfem::Vector &el_state,
              mfem::Vector &el_output) const override
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
         const double trans_weight = trans.Weight();
         const double curl_mag = curl_vec_norm / trans_weight;
         el_output(i) = curl_mag;
      }
   }

   void apply_state_bar(const mfem::FiniteElement &state_fe,
                        const mfem::FiniteElement &output_fe,
                        mfem::ElementTransformation &trans,
                        const mfem::Vector &el_output_adj,
                        const mfem::Vector &el_state,
                        mfem::Vector &el_state_bar) const override
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
         // mfem::Vector output_adj_vec_bar(output_adj_vec_bar_buffer,
         // space_dim); output_adj_vec_bar = 0.0; add(output_adj_vec_bar,
         // fun_bar, curl_vec, output_adj_vec_bar);

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
   void apply_mesh_coords_bar(const mfem::FiniteElement &state_fe,
                              const mfem::FiniteElement &output_fe,
                              mfem::ElementTransformation &trans,
                              const mfem::Vector &el_output_adj,
                              const mfem::Vector &el_state,
                              mfem::Vector &mesh_coords_bar) const override
   {
      auto &isotrans = dynamic_cast<mfem::IsoparametricTransformation &>(trans);
      auto &mesh_fe = *isotrans.GetFE();
      int space_dim = isotrans.GetSpaceDim();
      int curl_dim = space_dim == 3 ? 3 : 1;

      int mesh_dof = mesh_fe.GetDof();
      int state_dof = state_fe.GetDof();
      int output_dof = output_fe.GetDof();

      mfem::DenseMatrix curlshape(state_dof, curl_dim);
      mfem::DenseMatrix curlshape_dFt(state_dof, curl_dim);
      mfem::Vector adj_shape(output_dof);

      double curl_vec_buffer[3];
      mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

      mfem::Vector adj_shape_bar(output_dof);
      mfem::DenseMatrix curlshape_dFt_bar(curl_dim, state_dof);

      mesh_coords_bar.SetSize(mesh_dof * space_dim);
      mesh_coords_bar = 0.0;
      mfem::DenseMatrix PointMat_bar(space_dim, mesh_dof);
      const auto &ir = output_fe.GetNodes();
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const auto &ip = ir.IntPoint(i);
         trans.SetIntPoint(&ip);

         state_fe.CalcCurlShape(ip, curlshape);
         MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
         curlshape_dFt.MultTranspose(el_state, curl_vec);

         const double curl_vec_norm = curl_vec.Norml2();
         const double trans_weight = trans.Weight();
         const double curl_mag = curl_vec_norm / trans_weight;

         output_fe.CalcPhysShape(trans, adj_shape);
         const double adj = el_output_adj * adj_shape;

         /// dummy functional for adjoint-weighted residual
         // double fun = adj * curl_mag;

         /// start reverse pass
         const double fun_bar = 1.0;

         /// double fun = adj * curl_mag;
         double adj_bar = fun_bar * curl_mag;
         double curl_mag_bar = fun_bar * adj;

         /// const double adj = el_output_adj * adj_shape;
         adj_shape_bar = 0.0;
         add(adj_shape_bar, adj_bar, el_output_adj, adj_shape_bar);

         /// output_fe.CalcPhysShape(trans, adj_shape);
         PointMat_bar = 0.0;
         output_fe.CalcPhysShapeRevDiff(trans, adj_shape_bar, PointMat_bar);

         /// const double curl_mag = curl_vec_norm / trans_weight;
         double curl_vec_norm_bar = curl_mag_bar / trans_weight;
         double trans_weight_bar = -curl_mag_bar * curl_vec_norm / pow(trans_weight, 2);

         /// double trans_weight = trans.Weight();
         isotrans.WeightRevDiff(trans_weight_bar, PointMat_bar);

         /// const double curl_vec_norm = curl_vec.Norml2();
         double curl_vec_bar_buffer[3];
         mfem::Vector curl_vec_bar(curl_vec_bar_buffer, space_dim);
         curl_vec_bar = 0.0;
         add(curl_vec_bar, curl_vec_norm_bar / curl_vec_norm, curl_vec, curl_vec_bar);

         /// curlshape_dFt.MultTranspose(el_state, curl_vec);
         curlshape_dFt_bar = 0.0;
         AddMultVWt(curl_vec_bar, el_state, curlshape_dFt_bar);

         /// MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
         double jac_bar_buffer[9];
         mfem::DenseMatrix jac_bar(jac_bar_buffer, space_dim, space_dim);
         jac_bar = 0.0;
         AddMult(curlshape_dFt_bar, curlshape, jac_bar);
         isotrans.JacobianRevDiff(jac_bar, PointMat_bar);

         /// state_fe.CalcCurlShape(ip, curlshape);

         /// insert PointMat_bar into mesh_coords_bar
         for (int j = 0; j < mesh_dof; ++j)
         {
            for (int d = 0; d < space_dim; ++d)
            {
               mesh_coords_bar(d * mesh_dof + j) += PointMat_bar(d, j);
            }
         }
      }
   }

};

}  // anonymous namespace

namespace mach
{
ScalarL2IdentityProjection::ScalarL2IdentityProjection(
    FiniteElementState &state,
    FiniteElementState &output)
 : L2TransferOperator(state,
                      output,
                      std::make_unique<ScalarIdentityOperator>())
{ }

L2IdentityProjection::L2IdentityProjection(FiniteElementState &state,
                                           FiniteElementState &output)
 : L2TransferOperator(state,
                      output,
                      std::make_unique<IdentityOperator>())
{ }

L2CurlProjection::L2CurlProjection(FiniteElementState &state,
                                   FiniteElementState &output)
 : L2TransferOperator(state, output, std::make_unique<CurlOperator>())
{ }

L2CurlMagnitudeProjection::L2CurlMagnitudeProjection(FiniteElementState &state,
                                                     FiniteElementState &output)
 : L2TransferOperator(state,
                      output,
                      std::make_unique<CurlMagnitudeOperator>())
{ }

void L2TransferOperator::apply(const MachInputs &inputs, mfem::Vector &out_vec)
{
   out_vec = 0.0;
   output.gridFunc() = 0.0;

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
      operation->apply(state_fe, output_fe, trans, el_state, el_output);

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
         operation->apply_state_bar(
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
   else if (wrt == "mesh_coords")
   {
      output_adjoint.distributeSharedDofs(out_bar);

      mfem::Vector state_tv;
      setVectorFromInputs(inputs, "state", state_tv, false, true);

      state.distributeSharedDofs(state_tv);
      const auto &state_fes = state.space();
      const auto &output_fes = output.space();
      const auto &mesh_coords_fes = mesh_coords_bar.space();
      mfem::Array<int> state_vdofs;
      mfem::Array<int> output_adj_vdofs;
      mfem::Array<int> mesh_coords_vdofs;
      mfem::Vector el_state;
      mfem::Vector el_output_adj;
      mfem::Vector el_mesh_coords_bar;

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

         auto *mesh_coords_dof_trans = mesh_coords_fes.GetElementVDofs(i, mesh_coords_vdofs);
         el_mesh_coords_bar.SetSize(mesh_coords_vdofs.Size());

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
         operation->apply_mesh_coords_bar(
             state_fe, output_fe, trans, el_output_adj, el_state, el_mesh_coords_bar);

         if (mesh_coords_dof_trans)
         {
            mesh_coords_dof_trans->TransformDual(el_mesh_coords_bar);
         }
         mesh_coords_bar.localVec().AddElementVector(mesh_coords_vdofs, el_mesh_coords_bar);
      }

      /// this should maybe accumulate into wrt_bar
      mesh_coords_bar.setTrueVec(wrt_bar);
   }
}

}  // namespace mach
