
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"

// Define a random (but physical) state for the following tests
const double rho = 0.9856566615165173;
const double rhoe = 2.061597236955558;
const double rhou[3] = {0.09595562550099601, -0.030658751626551423, -0.13471469906596886};

// Define a second ("right") state for the tests
const double rho2 = 0.8575252486261279;
const double rhoe2 = 2.266357718749846;
const double rhou2[3] = {0.020099729730903737, -0.2702434209304979, -0.004256150573245826};

// Define a random direction vector
const double dir[3] = {0.6541305612927484, -0.0016604759052086802, -0.21763228465741322}; 

const double press_check[3] = {0.8227706007961364, 0.8225798733170867, 0.8188974449720051};
const double spect_check[3] = {0.7708202616595441, 0.7707922224516813, 0.8369733021138251};

// Define the Euler flux values for checking; The first 3 entries are for the 1D flux,
// the next 4 for the 2D flux, and the last 5 for the 3D flux
double flux_check[12] = { 0.06276750716816328, 0.5443099358828419, 0.18367915116927888,
                          0.06281841528652295, 0.5441901312159292, -0.003319834568556836,
                          0.18381597015154405, 0.09213668302118563, 0.5446355336473805,
                         -0.004225661763216877, -0.19081130999838336, 0.2692613318765901};

// Define the Ismail-Roe flux values for checking; note that direction dim has dim
// fluxes to check, each with dim+2 values (so these arrays have dim*(dim+2) entries)
double fluxIR_1D_check[3] = { 0.05762997059393852, 0.8657490584200118, 0.18911342719531313};
double fluxIR_2D_check[8] = { 0.05745695853179271, 0.8577689686179764,
                             -0.00950417495796846, 0.1876024933934876,
                             -0.15230563477618272, -0.00950417495796846,
                              0.8793769967224431, -0.4972925398771235};
double fluxIR_3D_check[15] = { 0.0574981892393032, 0.8557913559735177, -0.009501872816742403,
                              -0.004281782020677902, 0.18745940261538557, -0.1521680689750138,
                              -0.009501872816742403, 0.8773475401633251, 0.011331669926974292,
                              -0.49610841114443704, -0.06857074541246752, -0.004281782020677901,
                               0.011331669926974292, 0.8573073150960174, -0.22355888319220793};

// Define the entropy variables for checking; The first 3 entries are for the 1D variables,
// the next 4 for the 2D variables, and the last 5 for the 3D variables
double entvar_check[12] = { 3.9314525991262625, 0.11662500508421983, -1.1979726312082222,
                            3.931451215675034, 0.11665204634055908, -0.037271458518573726,
                           -1.1982503991275848, 3.9313978743154965, 0.11717660873184964,
                           -0.037439061282697646, -0.16450741163391253, -1.2036387066151037};

// Define products between dq/dw, evaluated at q, with vector qR.  The first 3
// entries are for the 1D product, the next 4 for the 2D product, and the last 5 for the 3D
double dqdw_prod_check[12] = { 5.519470966793266, 0.7354003853089198, 15.455145738300104,
                               5.527756292714283, 0.7361610635597204, -0.4522247321815538,
                               15.479385147854865, 5.528329658757937, 0.7353303956712847,
                              -0.4509878224828504, -1.0127274881940238, 15.480857480526556};

TEST_CASE( "Log-average is correct", "[log-avg]")
{
   REQUIRE( mach::logavg(rho, rho) == Approx(rho) );
   REQUIRE( mach::logavg(rho, 2.0*rho) == Approx(1.422001977589051) );
}

TEMPLATE_TEST_CASE_SIG( "Euler flux functions, etc, produce correct values", "[euler]",
                        ((int dim), dim), 1, 2, 3 )
{
   // copy the data into mfem vectors for convenience 
   mfem::Vector q(dim+2);
   mfem::Vector qR(dim+2);
   mfem::Vector flux(dim+2);
   mfem::Vector nrm(dim);
   q(0) = rho;
   q(dim+1) = rhoe;
   qR(0) = rho2;
   qR(dim+1) = rhoe2;
   for (int di = 0; di < dim; ++di)
   {
      q(di+1) = rhou[di];
      qR(di+1) = rhou2[di];
      nrm(di) = dir[di];
   }

   SECTION( "Pressure function is correct" )
   {
      REQUIRE( mach::pressure<double,dim>(q.GetData()) == 
               Approx(press_check[dim-1]) );
   }

   SECTION( "Spectral radius of flux Jacobian is correct" )
   {
      REQUIRE( mach::calcSpectralRadius<double,dim>(nrm.GetData(), q.GetData()) ==
               Approx(spect_check[dim-1]) );
   }

   SECTION( "Euler flux is correct" )
   {
      // Load the data to test the Euler flux; the pointer arithmetic in the 
      // following constructor is to find the appropriate offset
      int offset = div((dim+1)*(dim+2),2).quot - 3;
      mfem::Vector flux_vec(flux_check + offset, dim + 2);
      mach::calcEulerFlux<double,dim>(nrm.GetData(), q.GetData(), flux.GetData());
      for (int i = 0; i < dim+2; ++i)
      {
         REQUIRE( flux(i) == Approx(flux_vec(i)) );
      }
   }

   // load the data to test the IR flux function into an mfem DenseMatrix
   // TODO: I could not find an elegant way to do this
   mfem::DenseMatrix fluxIR_check;
   if (dim == 1)
   {
      fluxIR_check.Reset(fluxIR_1D_check, dim+2, dim);
   }
   else if (dim == 2)
   {
      fluxIR_check.Reset(fluxIR_2D_check, dim+2, dim);
   }
   else 
   {
      fluxIR_check.Reset(fluxIR_3D_check, dim+2, dim);
   }
   for (int di = 0; di < dim; ++di)
   {
      DYNAMIC_SECTION( "Ismail-Roe flux is correct in direction " << di )
      {
         mach::calcIsmailRoeFlux<double,dim>(di, q.GetData(), qR.GetData(),
                                             flux.GetData());
         for (int i = 0; i < dim+2; ++i)
         {
            REQUIRE( flux(i) == Approx(fluxIR_check(i,di)) );
         }
      }
   }

   SECTION( "Entropy variables are correct" )
   {
      // Load the data to test the entropy variables; the pointer arithmetic in the 
      // following constructor is to find the appropriate offset
      int offset = div((dim+1)*(dim+2),2).quot - 3;
      mfem::Vector entvar_vec(entvar_check + offset, dim + 2);
      mach::calcEntropyVars<double,dim>(q.GetData(), qR.GetData());
      for (int i = 0; i < dim+2; ++i)
      {
         REQUIRE( qR(i) == Approx(entvar_vec(i)) );
      }
   }

   SECTION( "dQ/dW * vec product is correct" )
   {
      // Load the data to test the dQ/dW * vec product; the pointer arithmetic in the 
      // following constructor is to find the appropriate offset
      int offset = div((dim+1)*(dim+2),2).quot - 3;
      mfem::Vector dqdw_prod(dqdw_prod_check + offset, dim+2);
      mach::calcdQdWProduct<double,dim>(q.GetData(), qR.GetData(), flux.GetData());
      for (int i = 0; i < dim+2; ++i)
      {
         REQUIRE( flux(i) == Approx(dqdw_prod(i)) );
      }
   }

}
