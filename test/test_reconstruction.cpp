#include "catch.hpp"
#include "mfem.hpp"
#include "utils.hpp"

TEST_CASE(" Reconstruction operator passed the test... ", "[reconstruction operator]")
{
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

   const double barycenters[dim*num_cent] =
      {0.903990418271183,  0.812791939414471,
       0.940947447226948,  0.697425752267883,
       0.802520365203252,  0.269452959819304,
       0.242045182197730,  0.589581312791356,
       0.975656775859888,  0.833047083813182,
       0.317231562283264,  0.363781892662813};

   const double quadratures[dim*num_quad] =
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
   for(int i = 0; i < num_cent; i++)
   {
      x_cent(0,i) = barycenters[i*dim];
      x_cent(1,i) = barycenters[i*dim+1];
   }
   for(int i = 0; i < num_quad; i++)
   {
      x_quad(0,i) = quadratures[i*dim];
      x_quad(1,i) = quadratures[i*dim+1];
   }
   mach::buildInterpolation(order, x_cent, x_quad, interp);

   std::cout << "\nInterplation operator is:\n";
   interp.Print();

   std::cout << "Check the accuracy:\n";
   mfem::Vector x_coord(num_cent), x_coord_interp(num_quad);
   mfem::Vector y_coord(num_cent), y_coord_interp(num_quad);
   mfem::Vector quad_x(num_quad), quad_y(num_quad);
   
   for(int i = 0; i < num_cent; i++)
   {
      x_coord(i) = x_cent(0,i);
      y_coord(i) = x_cent(1,i);
   }
   for(int i = 0; i < num_quad; i++)
   {
      quad_x(i) = x_quad(0,i);
      quad_y(i) = x_quad(1,i);
   }
   interp.Mult(x_coord, x_coord_interp);
   interp.Mult(y_coord, y_coord_interp);

   mfem::Vector x_diff(x_coord_interp), y_diff(y_coord_interp);
   x_diff -= quad_x;
   y_diff -= quad_y;

   REQUIRE( x_diff.Norml2() == Approx(0.0).margin(abs_tol));
   REQUIRE( y_diff.Norml2() == Approx(0.0).margin(abs_tol));

   std::cout << "x difference norm is " << x_diff.Norml2() << '\n';
   std::cout << "y difference norm is " << y_diff.Norml2() << '\n';
}



// extern "C" void
// dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
//       double *, int *, double *, int *, int *);
// void buildInterpolation(int degree, const DenseMatrix &x_center,
//     const DenseMatrix &x_quad, DenseMatrix &interp);


// void buildInterpolation(int degree, const DenseMatrix &x_center,
//     const DenseMatrix &x_quad, DenseMatrix &interp)
// {
//    // number of quadrature points
//    int m = x_quad.Width();
//    // number of elements
//    int n = x_center.Width();

//    // number of rows in little r matrix
//    int rows = (degree + 1) * (degree + 2) / 2; 

//    // Set the size of interpolation operator
//    interp.SetSize(m,n);
//    Vector rhs(rows);
//    // number of column 
//    int nrhs = 1;

//    // construct each row of R (also loop over each quadrature point)
//    for(int i = 0; i < m; i++)
//    {
//       // reset the rhs
//       rhs = 0.0; rhs(0) = 1.0;
//       // construct the aux matrix to solve each row of R
//       DenseMatrix r(rows, n);
//       r = 0.0;
//       // loop over each column of r
//       for(int j = 0; j < n; j++)
//       {
//          double x_diff = x_center(0,j) - x_quad(0,i);
//          double y_diff = x_center(1,j) - x_quad(1,i);
//          r(0,j) = 1.0;
//          int index = 1;
//          // loop over different orders
//          for(int order = 1; order <= degree; order++)
//          {
//             for(int c = order; c >= 0; c--)
//             {
//                r(index, j) = pow(x_diff,c) * pow(y_diff, order-c);
//                index++;
//             }
//          }
//       }
//       // Solve each row of R and put them back to R
//       int info;
//       mfem::Vector sv;
//       sv.SetSize(std::min(rows, n));
//       int rank;
//       double rcond = -1.0;
//       double *work = NULL;
//       double qwork;
//       int lwork = -1;
//       // query and allocate the optimal workspace
//       dgelss_(&rows, &n, &nrhs, r.GetData(), &rows, rhs.GetData(), &rows,
//               sv.GetData(), &rcond, &rank, &qwork, &lwork, &info);
//       lwork = (int) qwork;
//       work = new double [lwork];
//       // solve the equation rx = rhs
//       dgelss_(&rows, &n, &nrhs, r.GetData(), &rows, rhs.GetData(), &rows,
//               sv.GetData(), &rcond, &rank, work, &lwork, &info);
//       delete [] work;
//       for(int k = 0; k < n; k++)
//       {
//          interp(i,k) = rhs(k);
//       }
//    } // end of constructing interp
// }

// int main(int argc, char *argv[])
// {
//    int dim = 2;
//    int num_quad = 9;
//    int num_cent = 3;

//    DenseMatrix x_cent(dim, num_cent);
//    DenseMatrix x_quad(dim, num_quad);
//    DenseMatrix interp(num_quad, num_cent);
   
//    const double barycenters[6] = 
//       {0.166666666666667, 0.621839873491039,
//        0.359209348920498, 0.438349810923898,
//        0.629304983490984, 0.284898873409849};
//    const double quadratures[18] = 
//       {0.103873498738494, 0.384876234987589,
//        0.129384745859873, 0.783987498713894,
//        0.263874983408948, 0.603455230983840,
       
//        0.168723984859487, 0.183987349872834,
//        0.349871983479491, 0.589873498748194,
//        0.589298374919349, 0.374987340985849,
       
//        0.319847893847284, 0.123873498748294,
//        0.693487239847824, 0.319487583949834,
//        0.928374987348194, 0.123873485794834};

//    for(int i = 0; i < num_cent; i++)
//    {
//       x_cent(0,i) = barycenters[i*dim];
//       x_cent(1,i) = barycenters[i*dim+1];
//    }
//    for(int i = 0; i < num_quad; i++)
//    {
//       x_quad(0,i) = quadratures[i*dim];
//       x_quad(1,i) = quadratures[i*dim+1];
//    }
//    std::cout << "x_cent is: \n";
//    x_cent.Print();
//    std::cout << "\nx_quad is:\n";
//    x_quad.Print();

//    int degree = 1;
//    buildInterpolation(degree, x_cent, x_quad, interp);

//    std::cout << "\nInterplation operator is:\n";
//    interp.Print();

//    std::cout << "Check the accuracy:\n";
//    mfem::Vector x_coord(num_cent), x_coord_interp(num_quad);
//    mfem::Vector y_coord(num_cent), y_coord_interp(num_quad);
//    mfem::Vector quad_x(num_quad), quad_y(num_quad);
   
//    for(int i = 0; i < num_cent; i++)
//    {
//       x_coord(i) = x_cent(0,i);
//       y_coord(i) = x_cent(1,i);
//    }
//    for(int i = 0; i < num_quad; i++)
//    {
//       quad_x(i) = x_quad(0,i);
//       quad_y(i) = x_quad(1,i);
//    }
//    interp.Mult(x_coord, x_coord_interp);
//    interp.Mult(y_coord, y_coord_interp);

//    mfem::Vector x_diff(x_coord_interp), y_diff(y_coord_interp);
//    x_diff -= quad_x;
//    y_diff -= quad_y;

//    std::cout << "x difference norm is " << x_diff.Norml2() << '\n';
//    std::cout << "y difference norm is " << y_diff.Norml2() << '\n';
//    return 0;
// }