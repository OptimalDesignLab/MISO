#include "catch.hpp"
#include "mfem.hpp"
#include "navier_stokes_fluxes.hpp"
#include "euler_test_data.hpp"
#include "euler_fluxes.hpp"

TEMPLATE_TEST_CASE_SIG("navierstokes flux functions, etc, produce correct values",
                       "[navierstokes]", ((int dim), dim), 1, 2, 3)
{
   using namespace euler_data;
   using namespace std;
   double gamma = 1.4;
   double gami = gamma - 1.0;
   // copy the data into mfem vectors for convenience
   mfem::Vector q(dim + 2);
   // stores shear stress vector for given direction
   mfem::Vector tau_ij(dim);
   mfem::Vector cdel_wxj(dim + 2);
   mfem::Vector cdwij(dim + 2);
   // stores analytical flux vector for given direction
   mfem::Vector flux(dim + 2);
   // stores computed flux vector for given direction
   mfem::Vector flux_vec(dim + 2);
   // create the derivative vector
   mfem::Vector del_wxj(dim + 2);
   mfem::Vector del_qxj(dim + 2);
   // stores analytical fluxes for all directions
   mfem::DenseMatrix fv(dim + 2, dim);
   // stores entropy variables derivatives along all directions
   mfem::DenseMatrix del_w(dim + 2, dim);
   // stores spatial derivatives of primitive variables
   mfem::DenseMatrix del_vxj(dim + 2, dim);
   // stores computed fluxes for all directions
   mfem::DenseMatrix fcdwij(dim + 2, dim);
   // define the constants
   double Re = 1000;
   double Pr = 0.71;
   //double mu = 1.4;
   // conservative variables
   q(0) = rho;
   q(dim + 1) = rhoe;

   for (int di = 0; di < dim; ++di)
   {
      q(di + 1) = rhou[di];
   }
   // viscosity coefficient
   double mu = mach::calcSutherlandViscosity<double, dim>(q) / Re;
   // spatial derivatives of entropy variables
   for (int j = 0; j < dim; ++j)
   {
      for (int i = 0; i < dim + 2; ++i)
      {
         del_w(i, j) = 0.02 + (i * 0.01) +
                       j * 0.02;
      }
   }
   // get spatial derivatives of conservative variables
   // and use them to get respective derivatives for primitive variables
   for (int j = 0; j < dim; ++j)
   {
      del_w.GetColumn(j, del_wxj);
      mach::calcdQdWProduct<double, dim>(q.GetData(), del_wxj.GetData(),
                                         del_qxj.GetData());
      del_vxj(dim + 1, j) = (gami * del_qxj(dim + 1)) / q(0);
      del_vxj(0, j) = del_qxj(0);
      for (int i = 1; i < dim + 1; ++i)
      {

         del_vxj(i, j) = (del_qxj(i) - (q(i) * del_qxj(0) / q(0))) / q(0);
         del_vxj(dim + 1, j) -= gami * q(i) * del_qxj(i) / (q(0) * q(0));

         del_vxj(dim + 1, j) += gami * (q(i) * q(i) * del_qxj(0)) / (2 * q(0) * q(0) * q(0));
      }
      del_vxj(dim + 1, j) -= mach::pressure<double, dim>(q) * del_qxj(0) / (q(0) * q(0));
      del_vxj(dim + 1, j) *= mu * gamma / (Pr * gami);
   }

   // get the analytical fluxes
   for (int i = 0; i < dim; ++i)
   {
      for (int j = 0; j < dim; ++j)
      {
         tau_ij(j) = del_vxj(i + 1, j) + del_vxj(j + 1, i);
         if (i == j)
         {
            for (int k = 0; k < dim; ++k)
            {
               tau_ij(j) -= 2 * del_vxj(k + 1, k) / 3;
            }
         }
         tau_ij(j) *= mu;
         fv(0, j) = 0;
         fv(j + 1, i) = tau_ij(j);
      }
      for (int k = 0; k < dim; ++k)
      {
         fv(dim + 1, k) += tau_ij(k) * q(i + 1) / q(0);
      }
      fv(dim + 1, i) += del_vxj(dim + 1, i);
   }

   SECTION("applyCijMatrix is correct")
   {
      // get flux using c_{hat} matrices
      for (int i = 0; i < dim; ++i)
      {
         for (int k = 0; k < dim + 2; ++k)
         {
            cdwij(k) = 0;
         }
         for (int j = 0; j < dim; ++j)
         {
            // `cdel_wxj` should be initialized to zero
            for (int di = 0; di < dim + 2; ++di)
            {
               cdel_wxj(di) = 0;
            }
            del_w.GetColumn(j, del_wxj);
            mach::applyCijMatrix<double, dim>(i, j, mu, Pr, q.GetData(), del_wxj.GetData(), cdel_wxj.GetData());
            for (int k = 0; k < dim + 2; ++k)
            {
               cdwij(k) += cdel_wxj(k);
            }
         }
         for (int s = 0; s < dim + 2; ++s)
         {
            fcdwij(s, i) = cdwij(s);
         }
      }
      // compare numerical vs analytical fluxes
      for (int i = 0; i < dim; ++i)
      {
         for (int k = 0; k < dim + 2; ++k)
         {
            flux(k) = 0;
            flux_vec(k) = 0;
         }
         fv.GetColumn(i, flux);
         fcdwij.GetColumn(i, flux_vec);
         for (int j = 0; j < dim + 2; ++j)
         {
            REQUIRE(flux(j) == Approx(flux_vec(j)));
         }
      }
   }

   SECTION("applyViscousScaling is correct")
   {
      // get flux using c_{hat} matrices
      for (int i = 0; i < dim; ++i)
      {
         double mu_Re = mach::calcSutherlandViscosity<double, dim>(q.GetData());
         mu_Re /= Re;
         // no need to initialize `cdwij` here, it's done in the function itself
         mach::applyViscousScaling<double, dim>(i, mu_Re, Pr, q.GetData(), del_w.GetData(), cdwij.GetData());
         for (int s = 0; s < dim + 2; ++s)
         {
            fcdwij(s, i) = cdwij(s);
         }
      }
      // compare numerical vs analytical fluxes
      for (int i = 0; i < dim; ++i)
      {
         for (int k = 0; k < dim + 2; ++k)
         {
            flux(k) = 0;
            flux_vec(k) = 0;
         }
         fv.GetColumn(i, flux);
         fcdwij.GetColumn(i, flux_vec);
         for (int j = 0; j < dim + 2; ++j)
         {
            REQUIRE(flux(j) == Approx(flux_vec(j)));
         }
      }
   }
}
