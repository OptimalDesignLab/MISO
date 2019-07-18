#include "catch.hpp"
#include "mfem.hpp"
#include "sbp_fe.hpp"

using namespace mfem;

/// Used for floating point checks on various SBP operations
const double abs_tol = std::numeric_limits<float>::epsilon()*100;

/// Used to build polynomials of the form u = x^p * y^q
void polynomial2D(const Vector &x, int p, const Vector &y, int q, Vector &u)
{
   MFEM_ASSERT( x.Size() == y.Size() && x.Size() == u.Size(), "");
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = std::pow(x(i),p)*std::pow(y(i),q);
   }
}

TEST_CASE( "Triangle SBP difference operator is accurate...", "[sbp-diff]")
{
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TRIANGLE)));
         DenseMatrix D(sbp.GetDof());
         Vector x, y;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         Vector u(sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector dudy(sbp.GetDof());
         Vector du(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               int i = r - j;
               polynomial2D(x, i, y, j, u);
               polynomial2D(x, std::max<int>(0, i - 1), y, j, dudx);
               dudx *= i;
               polynomial2D(x, i, y, std::max<int>(0, j - 1), dudy);
               dudy *= j;
               sbp.getStrongOperator(0, D);
               D.Mult(u, du);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( du(k) == Approx(dudx(k)).margin(abs_tol) );
               }
               sbp.getStrongOperator(1, D);
               D.Mult(u, du);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( du(k) == Approx(dudy(k)).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle SBP multWeakOperator is accurate...", "[sbp-apply-Q]")
{
   // This test indirectly checks multWeakOperator by forming H^{-}*Q*u = D*u
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TRIANGLE)));
         Vector x, y;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector du(sbp.GetDof());
         DenseMatrix du_mat(du.GetData(), 1, sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector dudy(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               int i = r - j;
               polynomial2D(x, i, y, j, u);
               polynomial2D(x, std::max<int>(0, i - 1), y, j, dudx);
               dudx *= i;
               polynomial2D(x, i, y, std::max<int>(0, j - 1), dudy);
               dudy *= j;
               du = 0.0;
               sbp.multWeakOperator(0, u_mat, du_mat, false);
               sbp.multNormMatrixInv(du_mat, du_mat);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( du(k) == Approx(dudx(k)).margin(abs_tol) );
               }
               du = 0.0;
               sbp.multWeakOperator(1, u_mat, du_mat, false);
               sbp.multNormMatrixInv(du_mat, du_mat);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( du(k) == Approx(dudy(k)).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle SBP projection operator is accurate...", "[sbp-proj]")
{
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TRIANGLE)));
         DenseMatrix P(sbp.GetDof());
         sbp.getLocalProjOperator(P);
         Vector x, y;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         Vector u(sbp.GetDof());
         Vector Pu(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               int i = r - j;
               polynomial2D(x, i, y, j, u);
               P.Mult(u, Pu);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( Pu(k) == Approx(0.0).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}
