#include "euler_test_data.hpp"
#include "euler_fluxes.hpp"

using namespace mfem;

namespace euler_data
{

// Define the Euler flux values for checking; The first 3 entries are for the
// 1D flux, the next 4 for the 2D flux, and the last 5 for the 3D flux
double flux_check[12] = {
    0.06276750716816328, 0.5443099358828419, 0.18367915116927888,
    0.06281841528652295, 0.5441901312159292, -0.003319834568556836,
    0.18381597015154405, 0.09213668302118563, 0.5446355336473805,
    -0.004225661763216877, -0.19081130999838336, 0.2692613318765901};

// Define the Ismail-Roe flux values for checking; note that direction dim has
// dim fluxes to check, each with dim+2 values (so these arrays have dim*(dim+2)
// entries)
double fluxIR_1D_check[3] = {
    0.05762997059393852, 0.8657490584200118, 0.18911342719531313};
double fluxIR_2D_check[8] = {
    0.05745695853179271, 0.8577689686179764, -0.00950417495796846,
    0.1876024933934876, -0.15230563477618272, -0.00950417495796846,
    0.8793769967224431, -0.4972925398771235};
double fluxIR_3D_check[15] = {
    0.0574981892393032, 0.8557913559735177, -0.009501872816742403,
    -0.004281782020677902, 0.18745940261538557, -0.1521680689750138,
    -0.009501872816742403, 0.8773475401633251, 0.011331669926974292,
    -0.49610841114443704, -0.06857074541246752, -0.004281782020677901,
    0.011331669926974292, 0.8573073150960174, -0.22355888319220793};

// Define the flux returns by calcBoundaryFlux; note, only the 2d version is
// tested so far
const double flux_bnd_check[4] = {0.026438482001990546, 0.5871756903516657,
                                  0.008033780082953402, 0.05099700195316398};

// Define the entropy variables for checking; The first 3 entries are for the
// 1D variables, the next 4 for the 2D variables, and the last 5 for the 3D
// variables
double entvar_check[12] = {
    3.9314525991262625, 0.11662500508421983, -1.1979726312082222,
    3.931451215675034, 0.11665204634055908, -0.037271458518573726,
    -1.1982503991275848, 3.9313978743154965, 0.11717660873184964,
    -0.037439061282697646, -0.16450741163391253, -1.2036387066151037};

// Define products between dq/dw, evaluated at q, with vector qR.  The first 3
// entries are for the 1D product, the next 4 for the 2D product, and the last
// 5 for the 3D
double dqdw_prod_check[12] = {
    5.519470966793266, 0.7354003853089198, 15.455145738300104,
    5.527756292714283, 0.7361610635597204, -0.4522247321815538,
    15.479385147854865, 5.528329658757937, 0.7353303956712847,
    -0.4509878224828504, -1.0127274881940238, 15.480857480526556};

// Use this for finite-difference direction-derivative checks
double vec_pert[9] = {
    0.12338014544564024, -0.09515811381248972, -0.8546949642571233,
    -0.43724706495167226, -0.23245170541453294, 0.19554342457115859,
    -0.6550915049869203, 0.6064661887024042, 0.5870937295355494};

// Use this for LPS applyscaling function and its derivatives
double adjJ_data[9] = {0.964888535199277, 0.157613081677548,
                       0.970592781760616, 0.957166948242946, 0.485375648722841,
                       0.800280468888800, 0.141886338627215, 0.421761282626275,
                       0.915735525189067};
// spatial derivatives of entropy-variables
double delw_data[15] = {0.964888535199277, 0.7354003853089198, 0.157613081677548,
                   0.970592781760616, 0.7353303956712847, 0.957166948242946, 0.485375648722841,
                   0.800280468888800, 0.11662500508421983, 0.141886338627215, 0.421761282626275,
                   0.915735525189067, 0.6064661887024042, 0.19554342457115859, 0.12338014544564024};

// define the random-number generator; uniform between 0 and 1
static std::default_random_engine gen(std::random_device{}());
static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);

template <int dim, bool entvar>
void randBaselineVectorPert(const Vector &x, Vector &u)
{
    const double scale = 0.01;
    u(0) = rho * (1.0 + scale * uniform_rand(gen));
    u(dim + 1) = rhoe * (1.0 + scale * uniform_rand(gen));
    for (int di = 0; di < dim; ++di)
    {
        u(di + 1) = rhou[di] * (1.0 + scale * uniform_rand(gen));
    }
    if (entvar)
    {
       Vector q(u);
       mach::calcEntropyVars<double, dim>(q.GetData(), u.GetData());
    }
}
// explicit instantiation of the templated function above
template void randBaselineVectorPert<1, true>(const Vector &x, Vector &u);
template void randBaselineVectorPert<2, true>(const Vector &x, Vector &u);
template void randBaselineVectorPert<3, true>(const Vector &x, Vector &u);
template void randBaselineVectorPert<1, false>(const Vector &x, Vector &u);
template void randBaselineVectorPert<2, false>(const Vector &x, Vector &u);
template void randBaselineVectorPert<3, false>(const Vector &x, Vector &u);

template <int dim, bool entvar>
void randBaselinePertSA(const mfem::Vector &x, mfem::Vector &u)
{
    const double scale = 0.01;
    u(0) = rho * (1.0 + scale * uniform_rand(gen));
    u(dim + 1) = rhoe * (1.0 + scale * uniform_rand(gen));
    for (int di = 0; di < dim; ++di)
    {
        u(di + 1) = rhou[di]* (1.0 + scale * uniform_rand(gen));
    }
    if (entvar)
    {
       mfem::Vector q(u);
       mach::calcEntropyVars<double, dim>(q.GetData(), u.GetData());
    }
    u(dim + 2) = 3.0 *(1.0 + scale*(uniform_rand(gen) - 0.1));
}
// explicit instantiation of the templated function above
template void randBaselinePertSA<1, true>(const mfem::Vector &x, mfem::Vector &u);
template void randBaselinePertSA<2, true>(const mfem::Vector &x, mfem::Vector &u);
template void randBaselinePertSA<3, true>(const mfem::Vector &x, mfem::Vector &u);
template void randBaselinePertSA<1, false>(const mfem::Vector &x, mfem::Vector &u);
template void randBaselinePertSA<2, false>(const mfem::Vector &x, mfem::Vector &u);
template void randBaselinePertSA<3, false>(const mfem::Vector &x, mfem::Vector &u);

void randVectorState(const Vector &x, Vector &u)
{
    for (int i = 0; i < u.Size(); ++i)
    {
        u(i) = 2.0 * uniform_rand(gen) - 1.0;
    }
}

double randBaselinePert(const Vector &x)
{
    const double scale = 0.01;
    return 1.0 + scale * uniform_rand(gen);
}

double randState(const Vector &x)
{
    return 2.0 * uniform_rand(gen) - 1.0;
}

} // namespace euler_data