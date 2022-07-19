#include "catch.hpp"
#include "mfem.hpp"
#include "sbp_fe.hpp"

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

/// Used to build polynomials of the form u = x^p * y^q * z^m
void polynomial3D(const Vector &x, int p, const Vector &y, int q,
                  const Vector &z, int m, Vector &u)
{
   MFEM_ASSERT( x.Size() == y.Size() && x.Size() == z.Size() 
                && x.Size() == u.Size() , "");
   for (int i = 0; i < u.Size(); i++)
   {
      u(i) = std::pow(x(i),p)*std::pow(y(i),q)*std::pow(z(i),m);
   }
}

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
         Vector Dk(sbp.GetDof());
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            polynomial1D(x, std::max<int>(0, i - 1), dudx);
            dudx *= i;
            sbp.getStrongOperator(0, D);
            D.Mult(u, du);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( du(k) == Approx(dudx(k)).margin(abs_tol) );
               // Check that row extraction of D is correct
               sbp.getStrongOperator(0, k, Dk);
               for (int j = 0; j < sbp.GetDof(); ++j)
               {
                  REQUIRE( D(k,j) == Approx(Dk(j)).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment DSBP difference operator is accurate...", "[dsbp-seg-D]")
{
   int dim = 1;
   for (int p = 0; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, 2));
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
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( du(k) == Approx(dudx(k)).margin(abs_tol) );
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle SBP difference operator is accurate...", "[sbp-tri-D]")
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
         Vector Dk(sbp.GetDof());
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

                  // Check that row extraction of D is correct
                  sbp.getStrongOperator(0, k, Dk);
                  for (int col = 0; col < sbp.GetDof(); ++col)
                  {
                     REQUIRE(D(k, col) == Approx(Dk(col)).margin(abs_tol));
                  }
               }
               sbp.getStrongOperator(1, D);
               D.Mult(u, du);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( du(k) == Approx(dudy(k)).margin(abs_tol) );

                  // Check that row extraction of D is correct
                  sbp.getStrongOperator(1, k, Dk);
                  for (int col = 0; col < sbp.GetDof(); ++col)
                  {
                     REQUIRE(D(k, col) == Approx(Dk(col)).margin(abs_tol));
                  }
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle DSBP difference operator is accurate...", "[dsbp-tri-D]")
{
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, dim));
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

TEST_CASE( "Tetrahedron SBP difference operator is accurate...", "sbp-tet-D" )
{
   int dim = 3;
   for (int p = 0; p <= 1; p++)
   {  
      DYNAMIC_SECTION( "... for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON)));
         DenseMatrix D(sbp.GetDof());
         Vector x, y, z;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z);
         Vector u(sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector dudy(sbp.GetDof());
         Vector dudz(sbp.GetDof());
         Vector du(sbp.GetDof());
         Vector Dk(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {  
               for (int k = 0; k <= j; k++)
               {
                  int i = r-j-k;
                  int m = j-k;
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0,m), z, std::max<int>(0,k), u);
                  polynomial3D(x, std::max<int>(0, i-1), y, std::max<int>(0,m), z, std::max<int>(0,k), dudx);
                  dudx *= std::max<int>(0,i);
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0, m-1), z, std::max<int>(0,k), dudy);
                  dudy *= std::max<int>(0,m);
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0,m), z, std::max<int>(0, k-1), dudz);
                  dudz *= std::max<int>(0,k);
                  sbp.getStrongOperator(0,D);
                  D.Mult(u, du);

                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {  
                     REQUIRE( du(l) == Approx(dudx(l)).margin(abs_tol) );

                     // Check that row extraction of D is correct
                     sbp.getStrongOperator(0, l, Dk);
                     for (int col = 0; col < sbp.GetDof(); ++col)
                     {
                        REQUIRE(D(l, col) == Approx(Dk(col)).margin(abs_tol));
                     }
                  }    

                  sbp.getStrongOperator(1, D);
                  D.Mult(u, du);
                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {  
                     REQUIRE( du(l) == Approx(dudy(l)).margin(abs_tol) );

                     // Check that row extraction of D is correct
                     sbp.getStrongOperator(1, l, Dk);
                     for (int col = 0; col < sbp.GetDof(); ++col)
                     {
                        REQUIRE(D(l, col) == Approx(Dk(col)).margin(abs_tol));
                     }
                  }  

                  sbp.getStrongOperator(2, D);
                  D.Mult(u, du);
                  for (int l = 0; l < sbp.GetDof(); l++)
                  {  
                     REQUIRE( du(l) == Approx(dudz(l)).margin(abs_tol) );

                     // Check that row extraction of D is correct
                     sbp.getStrongOperator(2, l, Dk);
                     for (int col = 0; col < sbp.GetDof(); col++)
                     {
                        REQUIRE( D(l, col) == Approx(Dk(col)).margin(abs_tol) );
                     }
                  }                              
               }
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment SBP multWeak/StrongOperator are accurate...", "[sbp-seg-Q]")
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
         Vector du_vec(1);
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            polynomial1D(x, std::max<int>(0, i-1), dudx);
            dudx *= i;
            du = 0.0;
            sbp.multWeakOperator(0, u_mat, du_mat, false);
            sbp.multNormMatrixInv(du_mat, du_mat);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( du(k) == Approx(dudx(k)).margin(abs_tol) );
               // Now test the application at a node
               sbp.multStrongOperator(0, k, u_mat, du_vec);
               REQUIRE( du_vec(0) == Approx(dudx(k)).margin(abs_tol) );
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment DSBP multWeakOperator is accurate...", "[dsbp-seg-Q]")
{
   // This test indirectly checks multWeakOperator by forming H^{-}*Q*u = D*u
   int dim = 1;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, 2));
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
            polynomial1D(x, std::max<int>(0, i-1), dudx);
            dudx *= i;
            du = 0.0;
            sbp.multWeakOperator(0, u_mat, du_mat, false);
            sbp.multNormMatrixInv(du_mat, du_mat);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( du(k) == Approx(dudx(k)).margin(abs_tol) );
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle SBP multWeak/StrongOperator are accurate...", "[sbp-tri-Q]")
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
         Vector du_vec(1);
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

                  // Now test the application at a node
                  sbp.multStrongOperator(0, k, u_mat, du_vec);
                  REQUIRE( du_vec(0) == Approx(dudx(k)).margin(abs_tol) );
               }
               du = 0.0;
               sbp.multWeakOperator(1, u_mat, du_mat, false);
               sbp.multNormMatrixInv(du_mat, du_mat);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( du(k) == Approx(dudy(k)).margin(abs_tol) );

                  // Now test the application at a node
                  sbp.multStrongOperator(1, k, u_mat, du_vec);
                  REQUIRE( du_vec(0) == Approx(dudy(k)).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Tetrahedron SBP multWeak/StrongOperator are accurate...", "[sbp-tet-Q]")
{
   // This test indirectly checks multWeakOperator by forming H^{-}*Q*u = D*u
   int dim = 3;
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON)));
         Vector x, y, z;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector du(sbp.GetDof());
         DenseMatrix du_mat(du.GetData(), 1, sbp.GetDof());
         Vector dudx(sbp.GetDof());
         Vector dudy(sbp.GetDof());
         Vector dudz(sbp.GetDof());
         Vector du_vec(1);
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {  
               for (int k = 0; k <= j; ++k){
                  int i = r - j - k;
                  int m = j - k;
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0,m), z, std::max<int>(0,k), u);
                  polynomial3D(x, std::max<int>(0, i-1), y, std::max<int>(0,m), z, std::max<int>(0,k), dudx);
                  dudx *= std::max<int>(0,i);
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0, m-1), z, std::max<int>(0,k), dudy);
                  dudy *= std::max<int>(0,m);
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0,m), z, std::max<int>(0, k-1), dudz);
                  dudz *= std::max<int>(0,k);

                  du *= 0.0;
                  sbp.multWeakOperator(0, u_mat, du_mat, false);
                  sbp.multNormMatrixInv(du_mat, du_mat);
                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {
                     REQUIRE( du(l) == Approx(dudx(l)).margin(abs_tol) );

                     // Now test the application at a node
                     sbp.multStrongOperator(0, l, u_mat, du_vec);
                     REQUIRE( du_vec(0) == Approx(dudx(l)).margin(abs_tol) );
                  }

                  du *= 0.0;
                  sbp.multWeakOperator(1, u_mat, du_mat, false);
                  sbp.multNormMatrixInv(du_mat, du_mat);
                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {
                     REQUIRE( du(l) == Approx(dudy(l)).margin(abs_tol) );

                     // Now test the application at a node
                     sbp.multStrongOperator(1, l, u_mat, du_vec);
                     REQUIRE( du_vec(0) == Approx(dudy(l)).margin(abs_tol) );
                  }

                  du *= 0.0;
                  sbp.multWeakOperator(2, u_mat, du_mat, false);
                  sbp.multNormMatrixInv(du_mat, du_mat);
                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {
                     REQUIRE( du(l) == Approx(dudz(l)).margin(abs_tol) );

                     // Now test the application at a node
                     sbp.multStrongOperator(2, l, u_mat, du_vec);
                     REQUIRE( du_vec(0) == Approx(dudz(l)).margin(abs_tol) );
                  }
               }
            
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle DSBP multWeakOperator is accurate...", "[dsbp-tri-Q]")
{
   // This test indirectly checks multWeakOperator by forming H^{-}*Q*u = D*u
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, dim));
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
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            P.Mult(u, Pu);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( Pu(k) == Approx(0.0).margin(abs_tol) );
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment DSBP projection operator is accurate...", "[dsbp-seg-proj]")
{
   int dim = 1;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, 2));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::SEGMENT)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x;
         sbp.getNodeCoords(0, x);
         Vector u(sbp.GetDof());
         Vector Pu(sbp.GetDof());
         for (int i = 0; i <= p; ++i)
         {
            polynomial1D(x, i, u);
            P.Mult(u, Pu);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( Pu(k) == Approx(0.0).margin(abs_tol) );
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle SBP projection operator is accurate...", "[sbp-tri-proj]")
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
         sbp.getProjOperator(P);
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

TEST_CASE( "Tetrahedron SBP projection operator is accurate...", "[sbp-tet-proj]")
{
   int dim = 3;
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x, y, z;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z);
         Vector u(sbp.GetDof());
         Vector Pu(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {  
               for (int k = 0; k <= j; ++k)
               {
                  int i = r - j - k;
                  int m = j - k;
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0,m), z, std::max<int>(0,k), u);
                  P.Mult(u, Pu);
                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {
                     REQUIRE( Pu(l) == Approx(0.0).margin(abs_tol) );
                  }
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle DSBP projection operator is accurate...", "[dsbp-tri-proj]")
{
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TRIANGLE)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
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
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
            }
            // next, check the transposed version
            P.MultTranspose(u, Pu_check);
            sbp.multProjOperator(u_mat, Pu_mat, true);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Segment DSBP multProjOperator is accurate...", "[dsbp-seg-Prj]")
{
   int dim = 1;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, 2));
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
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
            }
            // next, check the transposed version
            P.MultTranspose(u, Pu_check);
            sbp.multProjOperator(u_mat, Pu_mat, true);
            for (int k = 0; k < sbp.GetDof(); ++k)
            {
               REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
            }
         }
      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle SBP multProjOperator is accurate...", "[sbp-tri-Prj]")
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
         sbp.getProjOperator(P);
         Vector x, y;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector Pu(sbp.GetDof());
         DenseMatrix Pu_mat(Pu.GetData(), 1, sbp.GetDof());
         Vector Pu_check(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               int i = r - j;
               polynomial2D(x, i, y, j, u);
               // first check the non-tranposed version
               P.Mult(u, Pu_check);
               sbp.multProjOperator(u_mat, Pu_mat, false);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
               }
               // next, check the transposed version
               P.MultTranspose(u, Pu_check);
               sbp.multProjOperator(u_mat, Pu_mat, true);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Tetrahedron SBP multProjOperator is accurate...", "[sbp-tet-Prj]")
{
   int dim = 3;
   for (int p = 0; p <= 1; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new SBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TETRAHEDRON)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x, y, z;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         sbp.getNodeCoords(2, z);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector Pu(sbp.GetDof());
         DenseMatrix Pu_mat(Pu.GetData(), 1, sbp.GetDof());
         Vector Pu_check(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {  
               for (int k = 0; k <= j; ++k)
               {
                  int i = r - j - k;
                  int m = j - k;
                  polynomial3D(x, std::max<int>(0,i), y, std::max<int>(0,m), z, std::max<int>(0,k), u);
                  // first check the non-transposed version
                  P.Mult(u, Pu_check);
                  sbp.multProjOperator(u_mat, Pu_mat, false);
                  for (int l = 0; l < sbp.GetDof(); ++l)
                  {
                     REQUIRE( Pu(l) == Approx(Pu_check(l)).margin(abs_tol) );
                  }
               }
               
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}

TEST_CASE( "Triangle DSBP multProjOperator is accurate...", "[dsbp-tri-Prj]")
{
   int dim = 2;
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION( "...for degree p = " << p )
      {
         std::unique_ptr<FiniteElementCollection> fec(new DSBPCollection(p, dim));
         const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(
            *(fec->FiniteElementForGeometry(Geometry::TRIANGLE)));
         DenseMatrix P(sbp.GetDof());
         sbp.getProjOperator(P);
         Vector x, y;
         sbp.getNodeCoords(0, x);
         sbp.getNodeCoords(1, y);
         Vector u(sbp.GetDof());
         DenseMatrix u_mat(u.GetData(), 1, sbp.GetDof());
         Vector Pu(sbp.GetDof());
         DenseMatrix Pu_mat(Pu.GetData(), 1, sbp.GetDof());
         Vector Pu_check(sbp.GetDof());
         for (int r = 0; r <= p; ++r)
         {
            for (int j = 0; j <= r; ++j)
            {
               int i = r - j;
               polynomial2D(x, i, y, j, u);
               // first check the non-tranposed version
               P.Mult(u, Pu_check);
               sbp.multProjOperator(u_mat, Pu_mat, false);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
               }
               // next, check the transposed version
               P.MultTranspose(u, Pu_check);
               sbp.multProjOperator(u_mat, Pu_mat, true);
               for (int k = 0; k < sbp.GetDof(); ++k)
               {
                  REQUIRE( Pu(k) == Approx(Pu_check(k)).margin(abs_tol) );
               }
            }
         }

      } // DYNAMIC SECTION
   } // loop over p
}