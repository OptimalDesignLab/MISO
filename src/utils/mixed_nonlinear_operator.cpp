#include "mach_input.hpp"

#include "mixed_nonlinear_operator.hpp"

namespace
{
void identity_operator(const mfem::FiniteElement &domain_fe,
                       const mfem::FiniteElement &range_fe,
                       mfem::ElementTransformation &trans,
                       const mfem::Vector &el_domain,
                       mfem::Vector &el_range)
{
   int domain_dof = domain_fe.GetDof();
   int range_dof = range_fe.GetDof();

   int space_dim = trans.GetSpaceDim();

   mfem::DenseMatrix vshape(domain_dof, space_dim);

   double shape_vec_buffer[3];
   mfem::Vector shape_vec(shape_vec_buffer, space_dim);

   mfem::DenseMatrix range(el_range.GetData(), range_dof, space_dim);

   const auto &ir = range_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      domain_fe.CalcVShape(trans, vshape);
      vshape.MultTranspose(el_domain, shape_vec);

      for (int j = 0; j < space_dim; j++)
      {
         range(i, j) = shape_vec(j);
      }
   }
}

void curl_operator(const mfem::FiniteElement &domain_fe,
                   const mfem::FiniteElement &range_fe,
                   mfem::ElementTransformation &trans,
                   const mfem::Vector &el_domain,
                   mfem::Vector &el_range)
{
   int domain_dof = domain_fe.GetDof();
   int range_dof = range_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(domain_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(domain_dof, curl_dim);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

   mfem::DenseMatrix range(el_range.GetData(), range_dof, curl_dim);

   const auto &ir = range_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      domain_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_domain, curl_vec);

      curl_vec /= trans.Weight();

      for (int j = 0; j < curl_dim; j++)
      {
         range(i, j) = curl_vec(j);
      }
   }
}

void curl_magnitude_operator(const mfem::FiniteElement &domain_fe,
                             const mfem::FiniteElement &range_fe,
                             mfem::ElementTransformation &trans,
                             const mfem::Vector &el_domain,
                             mfem::Vector &el_range)
{
   int domain_dof = domain_fe.GetDof();

   int space_dim = trans.GetSpaceDim();
   int curl_dim = space_dim == 3 ? 3 : 1;

   mfem::DenseMatrix curlshape(domain_dof, curl_dim);
   mfem::DenseMatrix curlshape_dFt(domain_dof, curl_dim);

   double curl_vec_buffer[3];
   mfem::Vector curl_vec(curl_vec_buffer, curl_dim);

   const auto &ir = range_fe.GetNodes();
   for (int i = 0; i < ir.GetNPoints(); ++i)
   {
      const auto &ip = ir.IntPoint(i);
      trans.SetIntPoint(&ip);

      domain_fe.CalcCurlShape(ip, curlshape);
      MultABt(curlshape, trans.Jacobian(), curlshape_dFt);
      curlshape_dFt.MultTranspose(el_domain, curl_vec);

      const double curl_vec_norm = curl_vec.Norml2();
      const double curl_mag = curl_vec_norm / trans.Weight();
      el_range(i) = curl_mag;
   }
}

}  // anonymous namespace

namespace mach
{

L2IdentityProjection::L2IdentityProjection(FiniteElementState &domain,
                                           FiniteElementState &range)
 : L2TransferOperator(domain, range, identity_operator)
{ }

L2CurlProjection::L2CurlProjection(FiniteElementState &domain,
                                   FiniteElementState &range)
 : L2TransferOperator(domain, range, curl_operator)
{ }

L2CurlMagnitudeProjection::L2CurlMagnitudeProjection(FiniteElementState &domain,
                                                     FiniteElementState &range)
 : L2TransferOperator(domain, range, curl_magnitude_operator)
{ }

void L2TransferOperator::apply(const MachInputs &inputs, mfem::Vector &out_vec)
{
   out_vec = 0.0;
   if (!operation)
   {
      return;
   }

   mfem::Vector state;
   setVectorFromInputs(inputs, "state", state, false, true);

   domain.distributeSharedDofs(state);

   const auto &domain_fes = domain.space();
   const auto &range_fes = range.space();
   mfem::Array<int> domain_vdofs;
   mfem::Array<int> range_vdofs;
   mfem::Vector el_domain;
   mfem::Vector el_range;

   for (int i = 0; i < range_fes.GetNE(); ++i)
   {
      const auto &domain_fe = *domain_fes.GetFE(i);
      const auto &range_fe = *range_fes.GetFE(i);
      auto &trans = *range_fes.GetElementTransformation(i);

      auto *domain_dof_trans = domain_fes.GetElementVDofs(i, domain_vdofs);
      el_domain.SetSize(domain_vdofs.Size());
      auto *range_dof_trans = range_fes.GetElementVDofs(i, range_vdofs);
      el_range.SetSize(range_vdofs.Size());

      domain.gridFunc().GetSubVector(domain_vdofs, el_domain);
      if (domain_dof_trans)
      {
         domain_dof_trans->InvTransformPrimal(el_domain);
      }

      /// apply the operation
      operation(domain_fe, range_fe, trans, el_domain, el_range);

      if (range_dof_trans)
      {
         range_dof_trans->TransformPrimal(el_range);
      }
      range.gridFunc().AddElementVector(range_vdofs, el_range);
   }

   range.setTrueVec(out_vec);
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
   { }
}

}  // namespace mach
