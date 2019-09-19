/// Defines data that can be used by test_euler_fluxes.hpp and test_euler.hpp

//#ifndef EULER_TEST_DATA
//#define EULER_TEST_DATA

/// Used for floating point checks when the benchmark value is zero
const double abs_tol = std::numeric_limits<double>::epsilon()*100;

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

// Define the flux returns by calcBoundaryFlux; note, only the 2d version is tested so far
double flux_bnd_check[4] = {0.026438482001990546, 0.5871756903516657, 0.008033780082953402,
                            0.05099700195316398};

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

//#endif