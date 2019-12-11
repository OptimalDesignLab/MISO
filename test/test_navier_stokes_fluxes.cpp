#include "catch.hpp"
#include "mfem.hpp"
#include "navier_stokes_fluxes.hpp"
#include "euler_test_data.hpp"
TEMPLATE_TEST_CASE_SIG("navierstokes flux functions, etc, produce correct values",
                       "[navierstokes]", ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   // copy the data into mfem vectors for convenience
   mfem::Vector q(dim + 2);
   // create the derivative vector
   mfem::Vector del_wxj(dim + 2);
   mfem::Vector del_qxj(dim + 2);
   mfem::Densematrix fv(dim + 2, dim);
   mfem::Densematrix del_w(dim + 2, dim);
   mfem::Densematrix del_vxj(dim + 2, dim);
   mfem::Densematrix stress_tensor(dim);
   del_w = 0;
   del_vxj = 0;
   // conservative variables
   q(0) = rho;
   q(dim + 1) = rhoe;
   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   // spatial derivatives of entropy variables
   for (int j = 0; j < dim; ++j)
   {
      for (int i = 0; i < dim + 2; ++i)
      {
         del_w(i, j) = 0.02 + (i * 0.01);
         del_w(i, j) += j * 0.02;
      }
   }
   // get spatial derivatives of conservative variables
   // and use them to get respective derivatives for primitive variables
   for (int j = 0; j < dim; ++j)
   {
      del_vxj(dim + 1, j) = 0;
      del_w.GetColumn(j, del_wxj);
      mach::calcdQdWProduct<double, dim>(q.GetData(), del_wxj.GetData(),
                                         del_qxj.GetData());
      del_vxj(0, j) = del_qxj(0);
      for (int i = 1; i < dim + 1; ++i)
      {
         del_vxj(i, j) = (del_qxj(i) - (q(i) * del_qxj(0) / q(0))) / q(0);
         del_vxj(dim + 1, j) += (q(i) * ((q(i) * del_qxj(0) / q(0)) - del_qxj(i))) / q(0) * q(0);
      }
      del_vxj(dim + 1, j) = (del_qxj(dim + 1) - (q(dim + 1) * del_qxj(0) / q(0))) / q(0);
      del_vxj(dim + 1, j) *= euler::gami;
   }

}
