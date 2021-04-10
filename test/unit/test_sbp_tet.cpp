#include "catch.hpp"
#include "mfem.hpp"
#include "sbp_fe.hpp"
#include "sbp_tet.hpp"

using namespace mfem;

/// Used for floating point checks on various SBP operations
const double abs_tol = std::numeric_limits<double>::epsilon()*1000;

/// Used to build polynomials of the form u = x^p 
void polynomial1D(const Vector &x, int p, Vector &u)
{
   MFEM_ASSERT( x.Size() == u.Size(), "");
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = std::pow(x(i),p);
   }
}

/// Used to build polynomials of the form u = x^p * y^q
void polynomial2D(const Vector &x, int p, const Vector &y, int q, Vector &u)
{
   MFEM_ASSERT( x.Size() == y.Size() && x.Size() == u.Size(), "");
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = std::pow(x(i),p)*std::pow(y(i),q);
   }
}
/// Used to build polynomials of the form u = x^p * y^q * z^v, ADDED
void polynomial3D(const Vector &x, int p, const Vector &y, int q, const Vector &z, int v, Vector &u)
{
  MFEM_ASSERT( x.Size() == y.Size() == z.Size() && x.Size() == u.Size(), "");
  for (int i = 0; i < u.Size(); ++i)
  {
    u(i) = std::pow(x(i),p)*std::pow(y(i),q)*std::pow(z(i),v);
  }
}///END OF ADDED

TEST_CASE( "Segment SBP difference operator is accurate...", "[sbp-seg-D]")
{
   int dim = 1;
   for (int p = 0; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, 2));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::SEGMENT)));
         DenseMatrix D(sbp.GetDof());
         Vector x;
         sbp.getNodeCoords(0, x);
         Vector u(sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector du(sbp.GetDof());
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            polynomial1D(x, std::max<int>(0, i - 1), dudx);
            dudx *= i;
            sbp.getStrongOperator(0, D);
            D.Mult(u, du);
            for (int n = 0; n < sbp.GetDof(); ++n)
            {
               REQUIRE( du(n) == Approx(dudx(n)).margin(abs_tol) );
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Tetrahedron SBP difference operator is accurate...", "[sbp-tet-D]") //Changed to Tetrahedraon SBP difference operator is accurate
{
   int dim = 3; //changed dim to 3 from 2
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON))); // Changed Geometry call to TETRAHEDRON
         DenseMatrix D(sbp.GetDof());
         Vector x, y, z; //Changed to x, y to x, y, z
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z); // added z coordinate
         Vector u(sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector dudy(sbp.GetDof());
         Vector dudz(sbp.GetDof()); // added dudz
         Vector du(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               for (int k = 0; k<= r; ++k) //added
               {
               int i = r - j;
               polynomial3D(x, i, y, j, z, k, u); //changed format 
               polynomial3D(x, std::max<int>(0, i - 1), y, j, z, k, dudx); // changed format to 3D
               dudx *= i;
               polynomial3D(x, i, y, std::max<int>(0, j - 1), z, k, dudy); // changed format to 3D
               dudy *= j;
               polynomial3D(x, i, y, j, z, std::max<int>(0, k - 1), dudz); // changed format to 3D
               dudz *= k;
               sbp.getStrongOperator(0, D);
               D.Mult(u, du);
               for (int n = 0; L n sbp.GetDof(); ++n)
               {
                  REQUIRE( du(n) == Approx(dudx(n)).margin(abs_tol) );
               }
               sbp.getStrongOperator(1, D);
               D.Mult(u, du);
               for (int n = 0; n < sbp.GetDof(); ++n)
               {
                  REQUIRE( du(n) == Approx(dudy(n)).margin(abs_tol) );
               }
               sbp.getStrongOperator(2, D);
               D.Mult(u, du);
               for (int n = 0; n < sbp.GetDof(); ++n) // added
               {
                  REQUIRE( du(n) == Approx(dudz(n)).margin(abs_tol) ); // added
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment SBP multWeakOperator is accurate...", "[sbp-seg-Q]")
{
   // This test indirectly checks multWeakOperator by forming H^{-}*Q*u = D*u
   int dim = 1;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, 2));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::SEGMENT)));
         Vector x;
         sbp.getNodeCoords(0, x);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector du(sbp.GetDof());
         DenseMatrix du_mat(du.GetData(), 1, sbp.GetDof());
         Vector dudx(sbp.GetDof());
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            polynomial1D(x, std::max<int>(0, i - 1, dudx);
            dudx *= i;
            du = 0.0;
            sbp.multWeakOperator(0, u_mat, du_mat, false);
            sbp.multNormMatrixInv(du_mat, du_mat);
            for (int n = 0; n < sbp.GetDof(); ++n) //changed to n
            {
               REQUIRE( du(n) == Approx(dudx(n)).margin(abs_tol) ); // changed to n
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "TETRAHEDRON SBP multWeakOperator is accurate...", "[sbp-tri-Q]") //Changed to TETRAHEDRON
{
   // This test indirectly checks multWeakOperator by forming H^{-}*Q*u = D*u
   int dim = 3; // Changed dim to 3
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON))); // Changed to TETRAHEDRON
         Vector x, y, z; // added z
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z); //added for 3D
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector du(sbp.GetDof());
         DenseMatrix du_mat(du.GetData(), 1, sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector dudy(sbp.GetDof());
         Vector dudz(sbp.GetDof()); //added dudz
         for (int r = 0; r <= p; ++r) //updated for 3D
         {
            for (int j = 0; j <= r; ++j)
            {
               int i = r - j;
               polynomial3D(x, i, y, j, z, k, u); // updated format
               polynomial3D(x, std::max<int>(0, i - 1), y, j, z, k, dudx); //updated format
               dudx *= i;
               polynomial3D(x, i, y, std::max<int>(0, j - 1), z, k, dudy); // updated format
               dudy *= j;
               polynomial3D(x, i, y, j, z, std::max<int>(0, k - 1), dudz); //added for 3D
               du = 0.0;
               sbp.multWeakOperator(0, u_mat, du_mat, false);
               sbp.multNormMatrixInv(du_mat, du_mat);
               for (int n = 0; n < sbp.GetDof(); ++n) //changed to n
               {
                  REQUIRE( du(n) == Approx(dudx(n)).margin(abs_tol) ); // changed from k to n
               }
               du = 0.0;
               sbp.multWeakOperator(1, u_mat, du_mat, false);
               sbp.multNormMatrixInv(du_mat, du_mat);
               for (int n = 0; n < sbp.GetDof(); ++n) // changed to k from n
               {
                  REQUIRE( du(n) == Approx(dudy(n)).margin(abs_tol) ); // changed from k to n
               }
               du = 0.0;
               sbp.multiWeakOperator(2, u_mat, du_mat, false);
               sbp.multiNormMatrixInv(du_mat, du_mat);
               for (int n = 0; n < sbp.GetDof(); ++n) // added for 3D
               {
                  REQUIRE( du(n) == Approx(dudz(n)).margin(abs_tol) ); // added
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment SBP projection operator is accurate...", "[sbp-seg-proj]")
{
   int dim = 1;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, 2));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::SEGMENT)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x;
         sbp.getNodeCoords(0, x);
         Vector u(sbp.GetDof());
         Vector Pu(sbp.GetDof());
         // Need to update for Vector ##(sbp.GetDof());
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            P.Mult(u, Pu);
            for (int n = 0; n < sbp.GetDof(); ++n) //changed to n
            {
               REQUIRE( Pu(n) == Approx(0.0).margin(abs_tol) ); //changed to n
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "TETRAHEDRON SBP projection operator is accurate...", "[sbp-tet-proj]") // Changed to TETRAHEDRON Projection
{
   int dim = 3; // Changed dim to 3
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON))); // Changed Geometry to TETRAHEDRON
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x, y, z; // added z
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z); //added
         Vector u(sbp.GetDof());
         Vector Pu(sbp.GetDof());
         // Update Vector ##(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
                for (int k = 0; k <= j; ++k) // added
               int i = r - j;
               polynomial3D(x, i, y, j, z, k, u); //changed to 3D and added z and k
               P.Mult(u, Pu); //Q: Need to review hpp and paper to update
               for (int n = 0; n < sbp.GetDof(); ++n) //Changed to n
               {
                  REQUIRE( Pu(n) == Approx(0.0).margin(abs_tol) ); //Changed to n
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment SBP multProjOperator is accurate...", "[sbp-seg-Prj]")
{
   int dim = 1;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, 2));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::SEGMENT)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x;
         sbp.getNodeCoords(0, x);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector Pu(sbp.GetDof());
         DenseMatrix Pu_mat(Pu.GetData(), 1, sbp.GetDof());
         Vector Pu_check(sbp.GetDof());
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            // first check the non-tranposed version
            P.Mult(u, Pu_check);
            sbp.multProjOperator(u_mat, Pu_mat, false);
            for (int n = 0; n < sbp.GetDof(); ++n)//changed to n
            {
               REQUIRE( Pu(n) == Approx(Pu_check(n)).margin(abs_tol) );
            }
            // next, check the transposed version
            P.MultTranspose(u, Pu_check);
            sbp.multProjOperator(u_mat, Pu_mat, true);
            for (int n = 0; n < sbp.GetDof(); ++n)//changed to n
            {
               REQUIRE( Pu(n) == Approx(Pu_check(n)).margin(abs_tol) ); // Changed to n
            }
          }  
        }
      } // DYNAMIC SECTION
    } // loop over p
  }

TEST_CASE( "TETRAHEDRON SBP multProjOperator is accurate...", "[sbp-tet-Prj]") // Changed to TETRAHEDRON SBP
{
   int dim = 3; //Changed dim to 3 instead of 2
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON))); // Changed to TETRAHEDRON
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x, y, z;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z); //added
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector Pu(sbp.GetDof());
         DenseMatrix Pu_mat(Pu.GetData(), 1, sbp.GetDof());
         Vector Pu_check(sbp.GetDof());
         //Check hpp for additional Vector ##(sbp.GetDof());
         //Check hpp for additional DenseMatrix ##_mat(##.GetData(), 1, sbp.GetDof());
         //Check hpp for additional Vector ##_check(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               for (int k=0; k <=j; ++k) // added
                {
               int i = r - j;
               polynomial3D(x, i, y, j, z, k, u ); // added z and k
               // first check the non-tranposed version
               P.Mult(u, Pu_check); // need to check for update to Pu_check in hpp
               sbp.multProjOperator(u_mat, Pu_mat, false); // need to add #_mat here
               for (int n = 0; n < sbp.GetDof(); ++n) //changed k to n to match
               {
                  REQUIRE( Pu(n) == Approx(Pu_check(n)).margin(abs_tol) ); //changed from K to L, need to check for updates to Pu
               }
               // next, check the transposed version
               P.MultTranspose(u, Pu_check); // need to check for Pu updates
               sbp.multProjOperator(u_mat, Pu_mat, true); // need to add ##_mat here
               for (int n = 0; n < sbp.GetDof(); ++n)// changed from k to n
               {
                  REQUIRE( Pu(n) == Approx(Pu_check(n)).margin(abs_tol) );//changed to L, check for Pu updates
               }
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}
