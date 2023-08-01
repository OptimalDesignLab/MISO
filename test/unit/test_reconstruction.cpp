#include "catch.hpp"
#include "mfem.hpp"
#include "utils.hpp"

TEST_CASE("Reconstruction operator passed the test... ",
          "[reconstruction operator]")
{
   #ifdef MFEM_USE_LAPACK
   const double abs_tol = std::numeric_limits<double>::epsilon()*100;
   int dim = 2;
   int num_quad = 18;
   int num_cent = 6;
   int order = 2;

   mfem::DenseMatrix x_cent(dim, num_cent);
   mfem::DenseMatrix x_quad(dim, num_quad);
   mfem::DenseMatrix interp(num_quad, num_cent);
   
   // const double barycenters[2*num_cent] = 
   //    {0.166666666666667, 0.621839873491039,
   //     0.359209348920498, 0.438349810923898,
   //     0.629304983490984, 0.284898873409849};

   // const double quadratures[2*num_quad] = 
   //    {0.103873498738494, 0.384876234987589,
   //     0.129384745859873, 0.783987498713894,
   //     0.263874983408948, 0.603455230983840,
       
   //     0.168723984859487, 0.183987349872834,
   //     0.349871983479491, 0.589873498748194,
   //     0.589298374919349, 0.374987340985849,
       
   //     0.319847893847284, 0.123873498748294,
   //     0.693487239847824, 0.319487583949834,
   //     0.928374987348194, 0.123873485794834};

   const double barycenters[12] =
      {0.903990418271183,  0.812791939414471,
       0.940947447226948,  0.697425752267883,
       0.802520365203252,  0.269452959819304,
       0.242045182197730,  0.589581312791356,
       0.975656775859888,  0.833047083813182,
       0.317231562283264,  0.363781892662813};

   const double quadratures[36] =
      {0.709794173746531,  0.380097949454923,
       0.338284837383885,  0.477113409671625,
       0.046575558064835,  0.990824967160828,
       0.763553384902618,  0.170752062302837,
       0.738489280203154,  0.499778660765846,
       0.941033160487678,  0.740991015257259,
       0.219598984372954,  0.771536559840653,
       0.335508122190693,  0.395683791716262,
       0.403081872404153,  0.781509564323069,
       0.607958430631424,  0.348630827320080,
       0.146709774626913,  0.094267917910556,
       0.391926609716924,  0.335581420793768,
       0.214605868296320,  0.089692404277073,
       0.506231622456638,  0.438991039893711,
       0.431339384965312,  0.237586037865968,
       0.807440331805559,  0.604182962239129,
       0.475283523586553,  0.912027542895358,
       0.919038054203134,  0.734926949150810};
   for (int i = 0; i < num_cent; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         x_cent(j,i) = barycenters[i*dim+j]; 
      }
   }
   for (int i = 0; i < num_quad; i++)
   {
      for (int j = 0; j < dim; j ++)
      {
         x_quad(0,i) = quadratures[i*dim+j];
      }
   }

   miso::buildInterpolation(dim, order, x_cent, x_quad, interp);

   //std::cout << "Check the accuracy:\n";
   mfem::Vector x_coord(num_cent), x_coord_interp(num_quad);
   mfem::Vector y_coord(num_cent), y_coord_interp(num_quad);
   mfem::Vector quad_x(num_quad), quad_y(num_quad);
   
   for (int i = 0; i < num_cent; i++)
   {
      x_coord(i) = x_cent(0,i);
      if (2 == dim)
      {
         y_coord(i) = x_cent(1,i);
      }
   }
   for (int i = 0; i < num_quad; i++)
   {
      quad_x(i) = x_quad(0,i);
      if (2 == dim)
      {
         quad_y(i) = x_quad(1,i);
      }
   }
   interp.Mult(x_coord, x_coord_interp);
   if (2 == dim)
   {
      interp.Mult(y_coord, y_coord_interp);
   }

   mfem::Vector x_diff(x_coord_interp);
   mfem::Vector y_diff(y_coord_interp);
   x_diff -= quad_x;
   if (2 == dim)
   {
      y_diff -= quad_y;
   }

   //std::cout << "x difference norm is " << x_diff.Norml2() << '\n';
   //std::cout << "y difference norm is " << y_diff.Norml2() << '\n';

   REQUIRE( x_diff.Norml2() == Approx(0.0).margin(abs_tol));
   if (2 == dim)
   {
      REQUIRE( y_diff.Norml2() == Approx(0.0).margin(abs_tol));
   }
   #endif
}

TEMPLATE_TEST_CASE_SIG("Least-squares reconstruction operator is accurate.",
                       "[least-squares]", ((int dim), dim), 1, 2, 3)
{
   #ifdef MFEM_USE_LAPACK
   const double abs_tol = std::numeric_limits<double>::epsilon()*100;

   // use these for the element barycenters
   // TODO: these should just be randomly generated, so we can extend the tests
   // for higher dimensions
   int num_cent = 12/dim;
   const double barycenters[12] =
      {0.903990418271183,  0.812791939414471,
       0.940947447226948,  0.697425752267883,
       0.802520365203252,  0.269452959819304,
       0.242045182197730,  0.589581312791356,
       0.975656775859888,  0.833047083813182,
       0.317231562283264,  0.363781892662813};

   // Use these for generic quadrature points
   int num_quad = 36/dim;
   const double quadratures[36] =
      {0.709794173746531,  0.380097949454923,
       0.338284837383885,  0.477113409671625,
       0.046575558064835,  0.990824967160828,
       0.763553384902618,  0.170752062302837,
       0.738489280203154,  0.499778660765846,
       0.941033160487678,  0.740991015257259,
       0.219598984372954,  0.771536559840653,
       0.335508122190693,  0.395683791716262,
       0.403081872404153,  0.781509564323069,
       0.607958430631424,  0.348630827320080,
       0.146709774626913,  0.094267917910556,
       0.391926609716924,  0.335581420793768,
       0.214605868296320,  0.089692404277073,
       0.506231622456638,  0.438991039893711,
       0.431339384965312,  0.237586037865968,
       0.807440331805559,  0.604182962239129,
       0.475283523586553,  0.912027542895358,
       0.919038054203134,  0.734926949150810};

   mfem::DenseMatrix x_cent(dim, num_cent);
   for (int i = 0; i < num_cent; i++)
   {
      for (int j = 0; j < dim; j++)
      {
         x_cent(j, i) = barycenters[i*dim+j]; 
      }
   }
   mfem::DenseMatrix x_quad(dim, num_quad);
   for (int i = 0; i < num_quad; i++)
   {
      for (int j = 0; j < dim; j ++)
      {
         x_quad(j, i) = quadratures[i*dim+j];
      }
   }

   mfem::DenseMatrix interp(num_quad, num_cent);
   mfem::Vector fun(num_cent);
   mfem::Vector fun_interp(num_quad);
   // Maximum polynomial degree is limited by number of element barycenters
   if (1 == dim)
   {
      for (int degree = 1; degree <= 4; ++degree)
      {
         miso::buildLSInterpolation(dim, degree, x_cent, x_quad, interp);
         // check that interp exactly interpolates all total degree polynomials
         for (int p = 0; p <= degree; ++p)
         {
            for (int i = 0; i < num_cent; ++i)
            {
               fun(i) = pow(x_cent(0, i), p);
            }
            interp.Mult(fun, fun_interp);
            for (int j = 0; j < num_quad; ++j)
            {
               double fun_exact = pow(x_quad(0, j), p);
               REQUIRE(fun_interp(j) == Approx(fun_exact).margin(abs_tol));
            }
         }
      }
   }
   else if (2 == dim)
   {
      for (int degree = 1; degree <= 2; ++degree)
      {
         miso::buildLSInterpolation(dim, degree, x_cent, x_quad, interp);
         // check that interp exactly interpolates all total degree polynomials
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               for (int i = 0; i < num_cent; ++i)
               {
                  fun(i) = pow(x_cent(0, i), p - q)*pow(x_cent(1, i), q);
               }
               interp.Mult(fun, fun_interp);
               for (int j = 0; j < num_quad; ++j)
               {
                  double fun_exact = pow(x_quad(0, j), p - q) *
                                     pow(x_quad(1, j), q);
                  REQUIRE(fun_interp(j) == Approx(fun_exact).margin(abs_tol));
               }
            }
         }
      }
   }
   else if (3 == dim)
   {
      for (int degree = 1; degree <= 1; ++degree)
      {
         miso::buildLSInterpolation(dim, degree, x_cent, x_quad, interp);
         // check that interp exactly interpolates all total degree polynomials
         for (int p = 0; p <= degree; ++p)
         {
            for (int q = 0; q <= p; ++q)
            {
               for (int r = 0; r <= p - q; ++r)
               {
                  for (int i = 0; i < num_cent; ++i)
                  {
                     fun(i) = pow(x_cent(0, i), p - q - r) *
                              pow(x_cent(1, i), r) * pow(x_cent(2, i), q);
                  }
                  interp.Mult(fun, fun_interp);
                  for (int j = 0; j < num_quad; ++j)
                  {
                     double fun_exact = pow(x_quad(0, j), p - q - r) *
                                        pow(x_quad(1, j), r) *
                                        pow(x_quad(2, j), q);
                     REQUIRE(fun_interp(j) ==
                             Approx(fun_exact).margin(abs_tol));
                  }
               }
            }
         }
      }
   }
   #endif
}
