#include <random>
#include <fstream>
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_integ.hpp"
#include "rans_fluxes.hpp"
#include "rans_integ.hpp"
#include "euler_test_data.hpp"

void uinit(const mfem::Vector &x, mfem::Vector& u);

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
            // std::cout << "Ismail-Roe " << n << " Flux: " << flux1(n) << std::endl; 
            // std::cout << "Spalart-Allmaras " << n << " Flux: " << flux2(n) << std::endl; 
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
    double Re = 1000.0; double Pr = 1.0;
    mach::SAFarFieldBC<dim> safarfieldinteg(diff_stack, &fe_coll, Re, Pr, qfar, 1.0);
    // +ve perturbation
    q_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    for (int di = 0; di < 2; di++) //reverse direction to check both inflow and outflow
    {
        if(di == 1)
            nrm *= -1.0;
        
        DYNAMIC_SECTION("SA Far-Field BC jacobian is correct w.r.t state when di = "<<di)
        {
            
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
                // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
                // std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SANoSlipAdiabaticWallBC Jacobian", "[SANoSlipAdiabaticWallBC]",
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
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix jac(dim + 3, dim + 3);
    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }
    double delta = 1e-5;
    
    x = 0.0;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
       Dw(dim+2, di) = 0.7;
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
    double Re = 1000.0; double Pr = 1.0;
    mach::SANoSlipAdiabaticWallBC<dim> sanoslipinteg(diff_stack, &fe_coll, Re, Pr, sacs, qfar, 1.5);
    // +ve perturbation
    q_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    DYNAMIC_SECTION("SA No-Slip BC jacobian is correct w.r.t state")
    {
            
        // get perturbed states flux vector
        sanoslipinteg.calcFlux(x, nrm, 1.0, q_plus, Dw, flux_plus);
        sanoslipinteg.calcFlux(x, nrm, 1.0, q_minus, Dw, flux_minus);
        // compute the jacobian
        sanoslipinteg.calcFluxJacState(x, nrm, 1.0, q, Dw, jac);
        jac.Mult(v, jac_v);
        // finite difference jacobian
        mfem::Vector jac_v_fd(flux_plus);
        jac_v_fd -= flux_minus;
        jac_v_fd /= 2.0 * delta;
        // compare each component of the matrix-vector products
        for (int i = 0; i < dim + 3; ++i)
        {
            // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            // std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SANoSlipAdiabaticWallBC Dw Jacobian", "[SANoSlipAdiabaticWallBC]",
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
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix jac(dim + 3, dim + 3);
    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }
    double delta = 1e-5;
    
    x = 0.0;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
       Dw(dim+2, di) = 0.7;
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
    mfem::Vector flux_plus(q), flux_minus(q);
    adept::Stack diff_stack;
    mfem::H1_FECollection fe_coll(1); //dummy
    double Re = 1000.0; double Pr = 1.0;
    mach::SANoSlipAdiabaticWallBC<dim> sanoslipinteg(diff_stack, &fe_coll, Re, Pr, sacs, qfar, 1.5);
    DYNAMIC_SECTION("SA No-Slip BC Dw jacobian is correct w.r.t state")
    {
        std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
        for (int d = 0; d < dim; ++d)
        {
            mat_vec_jac[d].SetSize(dim+3);
        }
        // compute the jacobian
        sanoslipinteg.calcFluxJacDw(x, nrm, 1.0, q, Dw, mat_vec_jac);

        for (int d = 0; d < dim; ++d)
        {
            // perturb one column of delw everytime
            mfem::DenseMatrix Dw_plus(Dw), Dw_minus(Dw);
            for (int s = 0; s < dim+3; ++s)
            {
                Dw_plus.GetColumn(d)[s] += v(s) * delta;
                Dw_minus.GetColumn(d)[s] -= v(s) * delta;
            }
            // get perturbed states flux vector
            sanoslipinteg.calcFlux(x, nrm, 1.0, q, Dw_plus, flux_plus);
            sanoslipinteg.calcFlux(x, nrm, 1.0, q, Dw_minus, flux_minus);

            mat_vec_jac[d].Mult(v, jac_v);
            // finite difference jacobian
            mfem::Vector jac_v_fd(flux_plus);
            jac_v_fd -= flux_minus;
            jac_v_fd /= 2.0 * delta;
            // compare each component of the matrix-vector products
            for (int i = 0; i < dim + 3; ++i)
            {
                // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
                // std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SAViscousSlipWallBC Jacobian", "[SASlipWallBC]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
//    std::cout << "Slip-Wall State Jac" << std::endl;
    mfem::Vector nrm(dim); mfem::Vector x(dim);
    mfem::Vector q(dim + 3);
    mfem::Vector flux(dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix jac(dim + 3, dim + 3);
    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }
    double delta = 1e-5;
    
    x = 0.0;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
       Dw(dim+2, di) = 0.7+0.1*di;
    }
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
    double Re = 1000.0; double Pr = 0.75;
    mach::SAViscousSlipWallBC<dim> saslipinteg(diff_stack, &fe_coll, Re, Pr, sacs, 1.5);
    // +ve perturbation
    q_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    DYNAMIC_SECTION("SA Slip-Wall BC jacobian is correct w.r.t state")
    {
            
        // get perturbed states flux vector
        saslipinteg.calcFlux(x, nrm, 1.0, q_plus, Dw, flux_plus);
        saslipinteg.calcFlux(x, nrm, 1.0, q_minus, Dw, flux_minus);
        // compute the jacobian
        saslipinteg.calcFluxJacState(x, nrm, 1.0, q, Dw, jac);
        jac.Mult(v, jac_v);
        // finite difference jacobian
        mfem::Vector jac_v_fd(flux_plus);
        jac_v_fd -= flux_minus;
        jac_v_fd /= 2.0 * delta;
        // compare each component of the matrix-vector products
        for (int i = 0; i < dim + 3; ++i)
        {
            // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            // std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SAViscousSlipWallBC Dw Jacobian", "[SASlipWallBC]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
//    std::cout << "Slip-Wall Dw Jac" << std::endl;
    mfem::Vector nrm(dim); mfem::Vector x(dim);
    mfem::Vector q(dim + 3);
    mfem::Vector flux(dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix jac(dim + 3, dim + 3);
    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }
    double delta = 1e-5;
    
    x = 0.0;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
       Dw(dim+2, di) = 0.7+0.1*di;
    }
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
    adept::Stack diff_stack;
    mfem::H1_FECollection fe_coll(1); //dummy
    double Re = 1000.0; double Pr = 0.75;
    mach::SAViscousSlipWallBC<dim> saslipinteg(diff_stack, &fe_coll, Re, Pr, sacs, 1.5);
    DYNAMIC_SECTION("SA Slip-Wall BC Dw jacobian is correct w.r.t state")
    {
        std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
        for (int d = 0; d < dim; ++d)
        {
            mat_vec_jac[d].SetSize(dim+3);
        }
        // compute the jacobian
        saslipinteg.calcFluxJacDw(x, nrm, 1.0, q, Dw, mat_vec_jac);

        for (int d = 0; d < dim; ++d)
        {
            // perturb one column of delw everytime
            mfem::DenseMatrix Dw_plus(Dw), Dw_minus(Dw);
            mfem::Vector flux_plus(q.Size()), flux_minus(q.Size());
            flux_plus = 0.0; flux_minus = 0.0;
            for (int s = 0; s < dim+3; ++s)
            {
                Dw_plus.GetColumn(d)[s] += v(s) * delta;
                Dw_minus.GetColumn(d)[s] -= v(s) * delta;
            }
            // get perturbed states flux vector
            saslipinteg.calcFlux(x, nrm, 1.0, q, Dw_plus, flux_plus);
            saslipinteg.calcFlux(x, nrm, 1.0, q, Dw_minus, flux_minus);

            mat_vec_jac[d].Mult(v, jac_v);
            // finite difference jacobian
            mfem::Vector jac_v_fd(flux_plus);
            jac_v_fd -= flux_minus;
            jac_v_fd /= 2.0 * delta;
            // compare each component of the matrix-vector products
            for (int i = 0; i < dim + 3; ++i)
            {
               //  std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
               //  std::cout << "AN " << i << " Deriv: " << jac_v[i] << std::endl; 
                REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
            }
        }
    }
}


TEMPLATE_TEST_CASE_SIG("SAViscous Jacobian", "[SAViscous]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector q(dim + 3);
    mfem::Vector conv(dim + 3);
    mfem::Vector conv_plus(dim + 3);
    mfem::Vector conv_minus(dim + 3);
    mfem::Vector scale(dim + 3);
    mfem::Vector scale_plus(dim + 3);
    mfem::Vector scale_minus(dim + 3);
    mfem::Vector scale_plus_2(dim + 3);
    mfem::Vector scale_minus_2(dim + 3);
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::Vector v(dim + 3);
    mfem::DenseMatrix vm(dim, dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix adjJ(dim, dim);
    mfem::DenseMatrix jac_conv(dim + 3, dim + 3);
    mfem::DenseMatrix jac_scale(dim + 3, dim + 3);
    double delta = 1e-5;
    double Re = 1000000; double Pr = 1;
    mfem::Vector sacs(13);
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
       Dw(dim+2, di) = 0.7;
    }
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // create perturbation matrix
    for (int di = 0; di < dim + 3; ++di)
    {
        for (int di2 = 0; di2 < dim; ++di2)
        {
            vm(di2, di) = (1.1 + 0.1*di2)*vec_pert[di];
        }
    }
    // perturbed vectors
    mfem::Vector q_plus(q), q_minus(q);
    adept::Stack diff_stack;
    mach::SAViscousIntegrator<dim> saviscousinteg(diff_stack, Re, Pr, sacs);
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Jacobians w.r.t state is correct, dir"<<di)
        {
            // compute the jacobian
            saviscousinteg.convertVarsJacState(q, jac_conv);
            saviscousinteg.applyScalingJacState(di, v, q, Dw, jac_scale);
            for (int i = 0; i < dim + 3; ++i)
            {
                q_plus = q; q_minus = q;
                // +ve perturbation
                q_plus(i) += delta;
                // -ve perturbation
                q_minus(i) -= delta;
                // get perturbed states conv vector
                saviscousinteg.convertVars(q_plus, conv_plus);
                saviscousinteg.convertVars(q_minus, conv_minus);
                saviscousinteg.applyScaling(di, v, q_plus, Dw, scale_plus);
                saviscousinteg.applyScaling(di, v, q_minus, Dw, scale_minus);
                
                //jac_conv.Mult(v, jac_v);
                // finite difference jacobian
                mfem::Vector jac_v_fd(conv_plus);
                jac_v_fd -= conv_minus;
                jac_v_fd /= 2.0 * delta;
                // compare each component of the matrix-vector products
                //std::cout << "viscous convertVars jac:" << std::endl; 
                for (int j = 0; j < dim + 3; ++j)
                {
                    // std::cout << "FD " << j << " Deriv: " << jac_v_fd[j]  << std::endl; 
                    // std::cout << "AN " << j << " Deriv: " << jac_conv(j, i) << std::endl; 
                    REQUIRE(jac_conv(j, i) == Approx(jac_v_fd[j]).margin(1e-10));
                }

                //jac_scale.Mult(v, jac_v);
                // finite difference jacobian
                jac_v_fd = scale_plus;
                jac_v_fd -= scale_minus;
                jac_v_fd /= 2.0 * delta;
                //std::cout << "viscous applyScaling jac:" << std::endl; 
                for (int j = 0; j < dim + 3; ++j)
                {
                    // std::cout << "FD " << j << " Deriv: " << jac_v_fd[j]  << std::endl; 
                    // std::cout << "AN " << j << " Deriv: " << jac_scale(j, i) << std::endl; 
                    REQUIRE(jac_scale(j, i) == Approx(jac_v_fd[j]).margin(1e-10));
                }
            }
        }
    }
}


TEMPLATE_TEST_CASE_SIG("SAViscous Dw Jacobian", "[SAViscous]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector q(dim + 3);
    mfem::Vector qr(dim + 2);
    mfem::Vector scale(dim + 3);
    mfem::Vector scale_plus_2(dim + 3);
    mfem::Vector scale_minus_2(dim + 3);
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix Dwr(delw_data, dim+2, dim);
    mfem::Vector v(dim + 3);
    mfem::Vector vr(dim + 2);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix adjJ(dim, dim);
    mfem::DenseMatrix jac_scale_2(dim + 3, dim + 3);
    double delta = 1e-5;
    double Re = 1000000; double Pr = 1;
    mfem::Vector sacs(13);
    Dw = 0.5;
    Dwr = 0.5;
    q(0) = rho;
    qr(0) = rho;
    q(dim + 1) = rhoe;
    qr(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
       qr(di + 1) = rhou[di];
       Dw(dim+2, di) = 0.7;
    }
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 2; ++di)
    {
       v(di) = vec_pert[di];
       vr(di) = vec_pert[di];
    }
    v(dim+2) = vec_pert[dim+2];
    // perturbed vectors
    mfem::DenseMatrix Dw_plus(Dw), Dw_minus(Dw);
    adept::Stack diff_stack;
    mach::SAViscousIntegrator<dim> saviscousinteg(diff_stack, Re, Pr, sacs, 1.0);
    mach::ESViscousIntegrator<dim> esviscousinteg(diff_stack, Re, Pr, 1.0);

      
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Jacobians w.r.t Dw is correct, dir"<<di)
        {
            std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
            std::vector<mfem::DenseMatrix> mat_vec_jac_ns(dim);
            for (int d = 0; d < dim; ++d)
            {
                mat_vec_jac[d].SetSize(dim+3);
                mat_vec_jac_ns[d].SetSize(dim+2);
            }
            // compute the jacobian
            saviscousinteg.applyScalingJacDw(di, v, q, Dw, mat_vec_jac);
            esviscousinteg.applyScalingJacDw(di, vr, qr, Dwr, mat_vec_jac_ns);

            stringstream nssolname; stringstream ranssolname;
            nssolname << "visc_jac_rans"; ranssolname << "visc_jac_ns";
            std::ofstream matlabns(nssolname.str()); matlabns.precision(15);
            std::ofstream matlabrans(ranssolname.str()); matlabrans.precision(15);
            mat_vec_jac[0].PrintMatlab(matlabrans);
            mat_vec_jac_ns[0].PrintMatlab(matlabns);

            for (int d = 0; d < dim; ++d)
            {
                // perturb one column of delw everytime
                mfem::DenseMatrix Dw_plus(Dw), Dw_minus(Dw);
                for (int s = 0; s < dim+3; ++s)
                {
                    Dw_plus.GetColumn(d)[s] += v(s) * delta;
                    Dw_minus.GetColumn(d)[s] -= v(s) * delta;
                }
                // get perturbed states conv vector
                saviscousinteg.applyScaling(di, v, q, Dw_plus, scale_plus_2);
                saviscousinteg.applyScaling(di, v, q, Dw_minus, scale_minus_2);
                
                mat_vec_jac[d].Mult(v, jac_v);
                // finite difference jacobian
                mfem::Vector jac_v_fd(dim+3);
                jac_v_fd = scale_plus_2;
                jac_v_fd -= scale_minus_2;
                jac_v_fd /= 2.0 * delta;
                //std::cout << "viscous applyScaling jac Dw:" << std::endl; 
                for (int j = 0; j < dim + 3; ++j)
                {
                    // std::cout << "FD " << j << " Deriv: " << jac_v_fd[j]  << std::endl; 
                    // std::cout << "AN " << j << " Deriv: " << jac_v(j) << std::endl; 
                    REQUIRE(jac_v(j) == Approx(jac_v_fd[j]).margin(1e-10));
                }
            }
        }
    }
}
                

TEMPLATE_TEST_CASE_SIG("SALPS Jacobian", "[SALPS]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector q(dim + 3);
    mfem::Vector conv(dim + 3);
    mfem::Vector conv_plus(dim + 3);
    mfem::Vector conv_minus(dim + 3);
    mfem::Vector scale(dim + 3);
    mfem::Vector scale_plus(dim + 3);
    mfem::Vector scale_minus(dim + 3);
    mfem::Vector scale_plus_2(dim + 3);
    mfem::Vector scale_minus_2(dim + 3);
    mfem::Vector vec(vec_pert, dim + 3);
    mfem::Vector v(dim + 3);
    mfem::Vector jac_v(dim + 3);
    mfem::DenseMatrix adjJ(dim, dim);
    mfem::DenseMatrix jac_conv(dim + 3, dim + 3);
    mfem::DenseMatrix jac_scale(dim + 3, dim + 3);
    mfem::DenseMatrix jac_scale_2(dim + 3, dim + 3);
    double delta = 1e-5;
    adjJ = 0.7;
    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector q_plus(q), q_minus(q);
    mfem::Vector vec_plus(vec), vec_minus(vec);
    adept::Stack diff_stack;
    mach::SALPSIntegrator<dim> salpsinteg(diff_stack);
    // +ve perturbation
    q_plus.Add(delta, v);
    vec_plus.Add(delta, v);
    // -ve perturbation
    q_minus.Add(-delta, v);
    vec_minus.Add(-delta, v);
    DYNAMIC_SECTION("LPS salpsinteg convertVars jacobian is correct w.r.t state ")
    {
        // get perturbed states conv vector
        salpsinteg.convertVars(q_plus, conv_plus);
        salpsinteg.convertVars(q_minus, conv_minus);
        salpsinteg.applyScaling(adjJ, q_plus, vec, scale_plus);
        salpsinteg.applyScaling(adjJ, q_minus, vec, scale_minus);
        salpsinteg.applyScaling(adjJ, q, vec_plus, scale_plus_2);
        salpsinteg.applyScaling(adjJ, q, vec_minus, scale_minus_2);
        // compute the jacobian
        salpsinteg.convertVarsJacState(q, jac_conv);
        salpsinteg.applyScalingJacState(adjJ, q, vec, jac_scale);
        salpsinteg.applyScalingJacV(adjJ, q, jac_scale_2);
        jac_conv.Mult(v, jac_v);
        // finite difference jacobian
        mfem::Vector jac_v_fd(conv_plus);
        jac_v_fd -= conv_minus;
        jac_v_fd /= 2.0 * delta;
        // compare each component of the matrix-vector products
        //std::cout << "convertVars jac:" << std::endl; 
        for (int i = 0; i < dim + 3; ++i)
        {
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
        
        jac_scale.Mult(v, jac_v);
        // finite difference jacobian
        jac_v_fd = scale_plus;
        jac_v_fd -= scale_minus;
        jac_v_fd /= 2.0 * delta;
        //std::cout << "applyScaling jac:" << std::endl; 
        for (int i = 0; i < dim + 3; ++i)
        {
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }

        jac_scale_2.Mult(v, jac_v);
        // finite difference jacobian
        jac_v_fd = scale_plus_2;
        jac_v_fd -= scale_minus_2;
        jac_v_fd /= 2.0 * delta;
        //std::cout << "applyScaling jac vec:" << std::endl; 
        for (int i = 0; i < dim + 3; ++i)
        {
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
    }
}

TEST_CASE("SAViscousIntegrator::AssembleElementGrad", "[SAViscousIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   adept::Stack diff_stack;
   double delta = 1e-5;

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         GridFunction dist;
         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::SAViscousIntegrator<dim>(diff_stack, 1000.0, 1.0, sacs));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePertSA<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Ja;cobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            // std::cout << "AN " << i << " Deriv: " << jac_v(i) << std::endl; 
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}


TEST_CASE("SANoSlipAdiabaticWallBC::AssembleElementGrad", "[SANoSlipAdiabaticWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   mfem::Vector qfar(dim + 3);
   adept::Stack diff_stack;
   double delta = 1e-5;

    qfar(0) = rho;
    qfar(dim + 1) = rhoe;
    qfar(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       qfar(di + 1) = rhou[di];
    }

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         GridFunction dist;
         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::SANoSlipAdiabaticWallBC<dim>(diff_stack, fec.get(), 1000.0, 1.0, sacs, qfar, 1.0));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePertSA<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Ja;cobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            // std::cout << "AN " << i << " Deriv: " << jac_v(i) << std::endl; 
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("SAViscousSlipWallBC::AssembleElementGrad", "[SAViscousSlipWallBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   mfem::Vector qfar(dim + 3);
   adept::Stack diff_stack;
   double delta = 1e-5;

    qfar(0) = rho;
    qfar(dim + 1) = rhoe;
    qfar(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       qfar(di + 1) = rhou[di];
    }

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         GridFunction dist;
         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::SAViscousSlipWallBC<dim>(diff_stack, fec.get(), 1000.0, 0.75, sacs, 1.5));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePertSA<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Ja;cobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            // std::cout << "AN " << i << " Deriv: " << jac_v(i) << std::endl; 
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}



TEST_CASE("SAFarFieldBC::AssembleElementGrad", "[SAFarFieldBC]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   mfem::Vector qfar(dim + 3);
   adept::Stack diff_stack;
   double delta = 1e-5;

    qfar(0) = rho;
    qfar(dim + 1) = rhoe;
    qfar(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       qfar(di + 1) = rhou[di];
    }

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 2; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         GridFunction dist;
         NonlinearForm res(fes.get());
         res.AddBdrFaceIntegrator(new mach::SAFarFieldBC<dim>(diff_stack, fec.get(), 1000.0, 1.0, qfar, 1.0));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePertSA<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Ja;cobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            // std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            // std::cout << "AN " << i << " Deriv: " << jac_v(i) << std::endl; 
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

TEST_CASE("SASourceIntegrator::AssembleElementGrad", "[SASourceIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   adept::Stack diff_stack;
   double delta = 1e-5;

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

   // generate a 8 element mesh
   int num_edge = 3;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   
   
   
   for (int p = 1; p <= 1; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));
         std::unique_ptr<FiniteElementSpace> fesx(new FiniteElementSpace(
             mesh.get(), fec.get(), 2, Ordering::byVDIM));

         mesh->EnsureNodes();
         
         GridFunction dist(fes.get()); 
         GridFunction x_nodes(fesx.get()); 
         auto walldist = [](const Vector &x)
         {
               return x(1); 
         };
         auto coord = [](const Vector &x, Vector &u)
         {
            u = x;
         };
         FunctionCoefficient wall_coeff(walldist);
         VectorFunctionCoefficient x_coord(2, coord);
         dist.ProjectCoefficient(wall_coeff);
         x_nodes.ProjectCoefficient(x_coord);
         //dist = 1.0000;
         NonlinearForm res(fes.get());
         double Re = 5000000.0;
         res.AddDomainIntegrator(new mach::SASourceIntegrator<dim>(diff_stack, dist, Re, sacs, 1.0, -1.0, 1.0, 1.0));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePertSA<2>);
         q.ProjectCoefficient(pert); 

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);

         // stringstream nssolname;
         // nssolname << "jac_test";
         // std::ofstream matlab(nssolname.str()); matlab.precision(15);
         // Jac.PrintMatlab(matlab);

         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         GridFunction error(fes.get());
            //jac_v.Size());
         error = jac_v;
         error -= jac_v_fd;

         // res.Mult(q, r);
         // ofstream sol_ofs("why.vtk");
         // sol_ofs.precision(14);
         // mesh->PrintVTK(sol_ofs, 1);
         // r.SaveVTK(sol_ofs, "jac_error", 1);

         std::cout.precision(17);
         std::cout << "Error Norm: " << error.Norml2() << std::endl; 
         for (int i = 0; i < jac_v.Size(); ++i)
         {
            int n = i/5;
            std::cout << "Node Coord: "<< x_nodes(0+2*n) <<", "<< x_nodes(1+2*n) <<std::endl;
            std::cout << "State: "<< q(i) <<std::endl;
            std::cout << "FD " << i << " Deriv: " << jac_v_fd[i]  << std::endl; 
            std::cout << "AN " << i << " Deriv: " << jac_v(i) << std::endl; 
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
         std::cout << "Error Norm: " << error.Norml2() << std::endl; 
      }
   }
}

TEST_CASE("SALPSIntegrator::AssembleElementGrad", "[SALPSIntegrator]")
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   int num_state = dim + 3;
   adept::Stack diff_stack;
   double delta = 1e-5;

   // generate a 2 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         std::unique_ptr<FiniteElementCollection> fec(
             new SBPCollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get(), num_state, Ordering::byVDIM));

         NonlinearForm res(fes.get());
         res.AddDomainIntegrator(new mach::SALPSIntegrator<dim>(diff_stack));

         // initialize state; here we randomly perturb a constant state
         GridFunction q(fes.get());
         VectorFunctionCoefficient pert(num_state, randBaselinePert<2>);
         q.ProjectCoefficient(pert);

         // initialize the vector that the Jacobian multiplies
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(num_state, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate the Jacobian and compute its product with v
         Operator &Jac = res.GetGradient(q);
         GridFunction jac_v(fes.get());
         Jac.Mult(v, jac_v);

         // now compute the finite-difference approximation...
         GridFunction q_pert(q), r(fes.get()), jac_v_fd(fes.get());
         q_pert.Add(-delta, v);
         res.Mult(q_pert, r);
         q_pert.Add(2 * delta, v);
         res.Mult(q_pert, jac_v_fd);
         jac_v_fd -= r;
         jac_v_fd /= (2 * delta);

         for (int i = 0; i < jac_v.Size(); ++i)
         {
            REQUIRE(jac_v(i) == Approx(jac_v_fd(i)).margin(1e-10));
         }
      }
   }
}

#if 0
TEMPLATE_TEST_CASE_SIG("SAInviscid Gradient",
                       "[SAInviscidGrad]",
                       ((bool entvar), entvar), false)
{
   using namespace mfem;
   using namespace euler_data;

   const int dim = 2; // templating is hard here because mesh constructors
   double delta = 1e-5;

   // generate a 8 element mesh
   int num_edge = 2;
   std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                       true /* gen. edges */, 1.0, 1.0, true));
   mesh->EnsureNodes();
   for (int p = 1; p <= 4; ++p)
   {
      DYNAMIC_SECTION("...for degree p = " << p)
      {
         // get the finite-element space for the state and adjoint
         std::unique_ptr<FiniteElementCollection> fec(
             new SBP_FECollection(p, dim));
         std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
             mesh.get(), fec.get()));

         // we use res for finite-difference approximation
         NonLinearForm res(fes.get());
         res.AddDomainIntegrator(
            new SAInviscidIntegrator<dim, entvar>( //Inviscid term
                this->diff_stack, alpha));

         // initialize state ; here we randomly perturb a constant state
         GridFunction state(fes.get())
         VectorFunctionCoefficient v_rand(dim+3, randState);
         state.ProjectCoefficient(pert);

         // initialize the vector that we use to perturb the mesh nodes
         GridFunction v(fes.get());
         VectorFunctionCoefficient v_rand(dim, randState);
         v.ProjectCoefficient(v_rand);

         // evaluate df/dx and contract with v
         GridFunction dfdx(*x_nodes);
         dfdx_form.Mult(*x_nodes, dfdx);
         double dfdx_v = dfdx * v;

         // now compute the finite-difference approximation...
         GridFunction x_pert(*x_nodes);
         GridFunction r(fes.get());
         x_pert.Add(delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         double dfdx_v_fd = adjoint * res;
         x_pert.Add(-2 * delta, v);
         mesh->SetNodes(x_pert);
         res.Assemble();
         dfdx_v_fd -= adjoint * res;
         dfdx_v_fd /= (2 * delta);
         mesh->SetNodes(*x_nodes); // remember to reset the mesh nodes

         REQUIRE(dfdx_v == Approx(dfdx_v_fd).margin(1e-10));
      }
   }
}
#endif

void uinit(const mfem::Vector &x, mfem::Vector& q)
{
   // q.SetSize(4);
   // Vector u(4);
   q.SetSize(5);
   mfem::Vector u(5);
   
   u = 0.0;
   u(0) = 1.0;
   u(1) = 1.5;//u(0)*1.5*cos(0.3*M_PI/180.0);
   u(2) = 0;//u(0)*1.5*sin(0.3*M_PI/180.0);
   u(3) = 1.0/(1.4) + 0.5*1.5*1.5;
   u(4) = u(0)*3;

   q = u;
}