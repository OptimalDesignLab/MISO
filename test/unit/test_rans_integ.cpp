#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_integ.hpp"
#include "rans_fluxes.hpp"
#include "rans_integ.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG("SA Inviscid Flux Test", "[sa_inviscid_flux_test]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    mfem::Vector qL(dim + 3);
    mfem::Vector qR(dim + 3);
    mfem::Vector flux1(dim + 2);
    mfem::Vector flux2(dim + 3);
    mfem::Vector nrm(dim);
    qL(0) = rho; qR(0) = rho*1.1;
    qL(dim + 1) = rhoe; qR(dim + 1) = rhoe*1.1;
    for (int di = 0; di < dim; ++di)
    {
       qL(di + 1) = rhou[di];
       qR(di + 1) = rhou[di]*1.1;
       nrm(di) = dir[di];
    }
    qL(dim + 2) = 5.0; qR(dim + 2) = 5.5;
    adept::Stack stack;
    mach::IsmailRoeIntegrator<dim> irinteg(stack);
    mach::SAInviscidIntegrator<dim> sainteg(stack);

    SECTION("Check if SA integrator flux function matches the output for the conservative variables")
    {
        // calculate the flux 
        irinteg.calcFlux(0, qL, qR, flux1);
        sainteg.calcFlux(0, qL, qR, flux2);

        // check if euler variables are the same as before
        for(int n = 0; n < dim+2; n++)
        {
            std::cout << "Ismail-Roe " << n << " Flux: " << flux1(n) << std::endl; 
            std::cout << "Spalart-Allmaras " << n << " Flux: " << flux2(n) << std::endl; 
            REQUIRE(flux1(n) - flux2(n) == Approx(0.0));
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SAInviscid Jacobian", "[SAInviscid]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector qL(dim + 3);
    mfem::Vector qR(dim + 3);
    mfem::Vector flux(dim + 3);
    mfem::Vector flux_plus(dim + 3);
    mfem::Vector flux_minus(dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix jacL(dim + 3, 2 * (dim + 3));
    mfem::DenseMatrix jacR(dim + 3, 2 * (dim + 3));
    double delta = 1e-5;
    qL(0) = rho;
    qL(dim + 1) = rhoe;
    qL(dim + 2) = 4;
    qR(0) = rho2;
    qR(dim + 1) = rhoe2;
    qR(dim + 2) = 4.5;
    for (int di = 0; di < dim; ++di)
    {
       qL(di + 1) = rhou[di];
       qR(di + 1) = rhou2[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector qL_plus(qL), qL_minus(qL);
    mfem::Vector qR_plus(qR), qR_minus(qR);
    adept::Stack diff_stack;
    mach::SAInviscidIntegrator<dim> sainvinteg(diff_stack);
    // +ve perturbation
    qL_plus.Add(delta, v);
    qR_plus.Add(delta, v);
    // -ve perturbation
    qL_minus.Add(-delta, v);
    qR_minus.Add(-delta, v);
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Ismail-Roe flux jacsainvintegian is correct w.r.t left state ")
        {
            // get perturbed states flux vector
            sainvinteg.calcFlux(di, qL_plus, qR, flux_plus);
            sainvinteg.calcFlux(di, qL_minus, qR, flux_minus);
            // compute the jacobian
            sainvinteg.calcFluxJacStates(di, qL, qR, jacL, jacR);
            jacL.Mult(v, jac_v);
            // finite difference jacobian
            mfem::Vector jac_v_fd(flux_plus);
            jac_v_fd -= flux_minus;
            jac_v_fd /= 2.0 * delta;
            // compare each component of the matrix-vector products
            for (int i = 0; i < dim + 2; ++i)
            {
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SAFarFieldBC Jacobian", "[SAFarFieldBC]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector nrm(dim); mfem::Vector x(dim);
    mfem::Vector q(dim + 3);
    mfem::Vector qfar(dim + 3);
    mfem::Vector flux(dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix Dw;
    mfem::DenseMatrix jac(dim + 3, dim + 3);
    double delta = 1e-5;
    
    x = 0.0;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }
    qfar.Set(1.1, q);
    // create direction vector
    for (int di = 0; di < dim; ++di)
    {
       nrm(di) = dir[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector q_plus(q), q_minus(q);
    mfem::Vector flux_plus(q), flux_minus(q);
    adept::Stack diff_stack;
    mfem::H1_FECollection fe_coll(1); //dummy
    mach::SAFarFieldBC<dim> safarfieldinteg(diff_stack, &fe_coll, qfar, 1.0);
    // +ve perturbation
    q_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    for (int di = 0; di < 2; di++) //reverse direction to check both inflow and outflow
    {
        DYNAMIC_SECTION("SA Far-Field BC safarfieldinteg jacobian is correct w.r.t state ")
        {
            nrm *= -1.0;
            // get perturbed states flux vector
            safarfieldinteg.calcFlux(x, nrm, 1.0, q_plus, Dw, flux_plus);
            safarfieldinteg.calcFlux(x, nrm, 1.0, q_minus, Dw, flux_minus);
            // compute the jacobian
            safarfieldinteg.calcFluxJacState(x, nrm, 1.0, q, Dw, jac);
            jac.Mult(v, jac_v);
            // finite difference jacobian
            mfem::Vector jac_v_fd(flux_plus);
            jac_v_fd -= flux_minus;
            jac_v_fd /= 2.0 * delta;
            // compare each component of the matrix-vector products
            for (int i = 0; i < dim + 3; ++i)
            {
                std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
                std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}


#if 0
TEMPLATE_TEST_CASE_SIG("SA Inviscid Integrator Test", "[sa_inviscid_integ_test]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    mfem::Vector qL(dim + 3);
    mfem::Vector qR(dim + 3);
    mfem::Vector flux1(dim + 2);
    mfem::Vector flux2(dim + 3);
    mfem::Vector nrm(dim);
    qL(0) = rho; qR(0) = rho*1.1;
    qL(dim + 1) = rhoe; qR(dim + 1) = rhoe*1.1;
    for (int di = 0; di < dim; ++di)
    {
       qL(di + 1) = rhou[di];
       qR(di + 1) = rhou[di]*1.1;
       nrm(di) = dir[di];
    }
    qL(dim + 2) = 5.0; qR(dim + 2) = 5.5;
    adept::Stack stack;
    mach::IsmailRoeIntegrator<dim> irinteg(stack);
    mach::SAInviscidIntegrator<dim> sainteg(stack);

    SECTION("Check if SA integrator flux function matches the output for the conservative variables")
    {
        // calculate the flux 
        irinteg.calcFlux(0, qL, qR, flux1);
        sainteg.calcFlux(0, qL, qR, flux2);

        // check if euler variables are the same as before
        for(int n = 0; n < dim+2; n++)
        {
            std::cout << "Ismail-Roe " << n << " Flux: " << flux1(n) << std::endl; 
            std::cout << "Spalart-Allmaras " << n << " Flux: " << flux2(n) << std::endl; 
            REQUIRE(flux1(n) - flux2(n) == Approx(0.0));
        }
    }
}
#endif