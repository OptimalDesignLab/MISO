#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_integ.hpp"
#include "rans_fluxes.hpp"
#include "rans_integ.hpp"
#include "euler_test_data.hpp"

TEMPLATE_TEST_CASE_SIG("SA calcVorticity Jacobian", "[SAVorticity]",
                       ((int dim), dim), 2, 3)
{
    using namespace euler_data;
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix trans_jac(adjJ_data, dim, dim);
    mfem::Vector curl_plus(3);
    mfem::Vector curl_minus(3);
    mfem::Vector v(dim+3);
    mfem::Vector jac_v(3);
    double delta = 1e-5;

    for (int di = 0; di < dim; ++di)
    {
       Dw(dim+2, di) = 0.7;
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    adept::Stack stack;

    DYNAMIC_SECTION("Jacobians w.r.t Dw is correct")
    {
        std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
        for (int d = 0; d < dim; ++d)
        {
            mat_vec_jac[d].SetSize(3, dim+3);
        }
        // compute the jacobian
        mach::calcVorticityJacDw<dim>(stack, Dw.GetData(), trans_jac.GetData(), 
                        mat_vec_jac);
        
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
            mach::calcVorticity<double, dim>(Dw_plus.GetData(), trans_jac.GetData(),  
                            curl_plus.GetData());
            mach::calcVorticity<double, dim>(Dw_minus.GetData(), trans_jac.GetData(),  
                            curl_minus.GetData());
            
            mat_vec_jac[d].Mult(v, jac_v);

            // finite difference jacobian
            mfem::Vector jac_v_fd(3);
            jac_v_fd = curl_plus;
            jac_v_fd -= curl_minus;
            jac_v_fd /= 2.0 * delta;
            //std::cout << "viscous applyScaling jac Dw:" << std::endl; 
            for (int j = 0; j < 3; ++j)
            {
                std::cout << "FD " << j << " Deriv: " << jac_v_fd[j]  << std::endl; 
                std::cout << "AN " << j << " Deriv: " << jac_v(j) << std::endl; 
                REQUIRE(jac_v(j) == Approx(jac_v_fd[j]).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("SA calcGrad Jacobian", "[SAVorticity]",
                       ((int dim), dim), 2, 3)
{
    using namespace euler_data;
    mfem::DenseMatrix Dw(delw_data, dim+3, dim);
    mfem::DenseMatrix trans_jac(adjJ_data, dim, dim);
    mfem::Vector grad_plus(dim);
    mfem::Vector grad_minus(dim);
    mfem::Vector v(dim+3);
    mfem::Vector jac_v(dim);
    double delta = 1e-5;

    for (int di = 0; di < dim; ++di)
    {
       Dw(dim+2, di) = 0.7;
    }
    // create perturbation vector
    for (int di = 0; di < dim + 3; ++di)
    {
       v(di) = vec_pert[di];
    }
    adept::Stack stack;

    DYNAMIC_SECTION("Jacobians w.r.t Dw is correct")
    {
        std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
        for (int d = 0; d < dim; ++d)
        {
            mat_vec_jac[d].SetSize(dim, dim+3);
        }
        // compute the jacobian
        mach::calcGradJacDw<dim>(stack, dim+2, Dw.GetData(), trans_jac.GetData(), 
                        mat_vec_jac);
        
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
            mach::calcGrad<double, dim>(dim+2, Dw_plus.GetData(), trans_jac.GetData(),  
                            grad_plus.GetData());
            mach::calcGrad<double, dim>(dim+2, Dw_minus.GetData(), trans_jac.GetData(),  
                            grad_minus.GetData());
            
            mat_vec_jac[d].Mult(v, jac_v);

            // finite difference jacobian
            mfem::Vector jac_v_fd(dim);
            jac_v_fd = grad_plus;
            jac_v_fd -= grad_minus;
            jac_v_fd /= 2.0 * delta;
            //std::cout << "viscous applyScaling jac Dw:" << std::endl; 
            for (int j = 0; j < dim; ++j)
            {
                std::cout << "FD " << j << " Deriv: " << jac_v_fd[j]  << std::endl; 
                std::cout << "AN " << j << " Deriv: " << jac_v(j) << std::endl; 
                REQUIRE(jac_v(j) == Approx(jac_v_fd[j]).margin(1e-10));
            }
        }
    }
}


TEST_CASE("SA S derivatives", "[SASource]")
{
    using namespace euler_data;
    const int dim = 2;
    mfem::Vector q(dim+3);
    mfem::Vector grad(dim);
    double delta = 1e-5;

    grad(0) = 0.0;
    grad(1) = 0.0;

    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

    adept::Stack stack;
    std::vector<adouble> sacs_a(sacs.Size());
    adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
    std::vector<adouble> q_a(sacs.Size());
    adept::set_values(q_a.data(), q.Size(), q.GetData());
    std::vector<adouble> grad_a(grad.Size());
    adept::set_values(grad_a.data(), grad.Size(), grad.GetData());

    DYNAMIC_SECTION("Jacobians w.r.t S is correct")
    {
        // compute the jacobian
        stack.new_recording();
        mfem::Vector dSrcdu(1);
        adouble S_a; 
        S_a.set_value(0.000);
        adouble d_a = 10.1;
        adouble mu_a = 1.1;
        adouble Re_a = 5000000;
        adouble src;// = mach::calcSADestruction<adouble, dim>(q_a.data(), S_a, mu_a, d_a, Re_a, sacs_a.data());
        src = mach::calcSASource<adouble,dim>(
         q_a.data(), grad_a.data(), sacs_a.data())/Re_a;
        // use negative model if turbulent viscosity is negative
        // if (q_a[dim+2] < 0)
        // {
        //     src += mach::calcSANegativeProduction<adouble,dim>(
        //         q_a.data(), S_a, sacs_a.data());
        //     src += mach::calcSANegativeDestruction<adouble,dim>(
        //         q_a.data(), d_a, sacs_a.data())/Re_a;
        // }
        // else
        {
            // src += mach::calcSAProduction<adouble,dim>(
            //     q_a.data(), mu_a, d_a, S_a, Re_a, sacs_a.data())/Re_a;
            src += mach::calcSADestruction<adouble,dim>(
                q_a.data(), mu_a, d_a, S_a, Re_a, sacs_a.data())/Re_a;
        }

        stack.independent(S_a);
        stack.dependent(src);
        stack.jacobian(dSrcdu.GetData());

        double Sp = 0.000 + delta; double Sm = 0.000 - delta;
        double d = 10.1;
        double mu = 1.1;
        double Re = 5000000;

        // get perturbed states conv vector
        //double srcp = mach::calcSADestruction<double, dim>(q.GetData(), Sp, mu, d, Re, sacs.GetData());
        //double srcm = mach::calcSADestruction<double, dim>(q.GetData(), Sm, mu, d, Re, sacs.GetData());

        double srcp = mach::calcSASource<double,dim>(
          q.GetData(), grad.GetData(), sacs.GetData())/Re;
        // if (q(dim+2) < 0)
        // {
        //     srcp += mach::calcSANegativeProduction<double,dim>(
        //         q.GetData(), Sp, sacs.GetData());
        //     srcp += mach::calcSANegativeDestruction<double,dim>(
        //         q.GetData(), d, sacs.GetData())/Re;
        // }
        // else
        {
            // srcp += mach::calcSAProduction<double,dim>(
            //     q.GetData(), mu, d, Sp, Re, sacs.GetData())/Re;
            srcp += mach::calcSADestruction<double,dim>(
                q.GetData(), mu, d, Sp, Re, sacs.GetData())/Re;
        }

        double srcm = mach::calcSASource<double,dim>(
          q.GetData(), grad.GetData(), sacs.GetData())/Re;
        // if (q(dim+2) < 0)
        // {
        //     srcm += mach::calcSANegativeProduction<double,dim>(
        //         q.GetData(), Sm, sacs.GetData());
        //     srcm += mach::calcSANegativeDestruction<double,dim>(
        //         q.GetData(), d, sacs.GetData())/Re;
        // }
        // else
        {
            // srcm += mach::calcSAProduction<double,dim>(
            //     q.GetData(), mu, d, Sm, Re, sacs.GetData())/Re;
            srcm += mach::calcSADestruction<double,dim>(
                q.GetData(), mu, d, Sm, Re, sacs.GetData())/Re;
        }

        // finite difference jacobian
        double jac_fd = srcp;
        jac_fd -= srcm;
        jac_fd /= 2.0 * delta;

        std::cout << "FD Deriv: " << jac_fd  << std::endl; 
        std::cout << "AN Deriv: " << dSrcdu(0) << std::endl; 
        REQUIRE(dSrcdu(0) == Approx(jac_fd).margin(1e-10));

    }
}

TEST_CASE("SA grad derivatives", "[SAVorticity]")
{
    using namespace euler_data;
    const int dim = 2;
    mfem::Vector q(dim+3);
    mfem::Vector grad(dim);
    mfem::Vector v(dim);
    double delta = 1e-5;

    grad(0) = 0.0;
    grad(1) = 0.0;

    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }

    for (int di = 0; di < dim; ++di)
    {
       v(di) = vec_pert[di];
    }

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

    adept::Stack stack;
    DYNAMIC_SECTION("Jacobians w.r.t grad is correct")
    {
        // compute the jacobian
        stack.new_recording();
        mfem::Vector dSrcdgrad(dim);
        std::vector<adouble> sacs_a(sacs.Size());
        adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
        std::vector<adouble> q_a(sacs.Size());
        adept::set_values(q_a.data(), q.Size(), q.GetData());
        std::vector<adouble> grad_a(grad.Size());
        adept::set_values(grad_a.data(), grad.Size(), grad.GetData());
        adouble S_a = 0.000; 
        adouble d_a = 10.1;
        adouble mu_a = 1.1;
        adouble Re_a = 5000000;
        adouble src;// = mach::calcSADestruction<adouble, dim>(q_a.data(), S_a, mu_a, d_a, Re_a, sacs_a.data());
        src = mach::calcSASource<adouble,dim>(
         q_a.data(), grad_a.data(), sacs_a.data())/Re_a;
        // use negative model if turbulent viscosity is negative
        if (q_a[dim+2] < 0)
        {
            src += mach::calcSANegativeProduction<adouble,dim>(
                q_a.data(), S_a, sacs_a.data());
            src += mach::calcSANegativeDestruction<adouble,dim>(
                q_a.data(), d_a, sacs_a.data())/Re_a;
        }
        else
        {
            src += mach::calcSAProduction<adouble,dim>(
                q_a.data(), mu_a, d_a, S_a, Re_a, sacs_a.data())/Re_a;
            src += mach::calcSADestruction<adouble,dim>(
                q_a.data(), mu_a, d_a, S_a, Re_a, sacs_a.data())/Re_a;
        }

        stack.independent(grad_a.data(), dim);
        stack.dependent(src);
        stack.jacobian(dSrcdgrad.GetData());

        double S = 0.000;
        double d = 10.1;
        double mu = 1.1;
        double Re = 5000000;
        mfem::Vector gradp(grad); mfem::Vector gradm(grad);
        gradp.Add(delta, v); gradm.Add(-delta, v);

        // get perturbed states conv vector
        //double srcp = mach::calcSADestruction<double, dim>(q.GetData(), Sp, mu, d, Re, sacs.GetData());
        //double srcm = mach::calcSADestruction<double, dim>(q.GetData(), Sm, mu, d, Re, sacs.GetData());

        double srcp = mach::calcSASource<double,dim>(
          q.GetData(), gradp.GetData(), sacs.GetData())/Re;
        if (q(dim+2) < 0)
        {
            srcp += mach::calcSANegativeProduction<double,dim>(
                q.GetData(), S, sacs.GetData());
            srcp += mach::calcSANegativeDestruction<double,dim>(
                q.GetData(), d, sacs.GetData())/Re;
        }
        else
        {
            srcp += mach::calcSAProduction<double,dim>(
                q.GetData(), mu, d, S, Re, sacs.GetData())/Re;
            srcp += mach::calcSADestruction<double,dim>(
                q.GetData(), mu, d, S, Re, sacs.GetData())/Re;
        }

        double srcm = mach::calcSASource<double,dim>(
          q.GetData(), gradm.GetData(), sacs.GetData())/Re;
        if (q(dim+2) < 0)
        {
            srcm += mach::calcSANegativeProduction<double,dim>(
                q.GetData(), S, sacs.GetData());
            srcm += mach::calcSANegativeDestruction<double,dim>(
                q.GetData(), d, sacs.GetData())/Re;
        }
        else
        {
            srcm += mach::calcSAProduction<double,dim>(
                q.GetData(), mu, d, S, Re, sacs.GetData())/Re;
            srcm += mach::calcSADestruction<double,dim>(
                q.GetData(), mu, d, S, Re, sacs.GetData())/Re;
        }

        // finite difference jacobian
        double jac_fd = srcp;
        jac_fd -= srcm;
        jac_fd /= 2.0 * delta;

        double jac_v = dSrcdgrad*v;

        std::cout << "FD Deriv: " << jac_fd  << std::endl; 
        std::cout << "AN Deriv: " << jac_v << std::endl; 
        REQUIRE(jac_v == Approx(jac_fd).margin(1e-10));

    }
}

TEST_CASE("SA q derivatives", "[SAVorticity]")
{
    using namespace euler_data;
    const int dim = 2;
    mfem::Vector q(dim+3);
    mfem::Vector grad(dim);
    mfem::Vector v(dim+3);
    double delta = 1e-5;

    grad(0) = 0.0;
    grad(1) = 0.0;

    q(0) = rho;
    q(dim + 1) = rhoe;
    q(dim + 2) = 4;
    for (int di = 0; di < dim; ++di)
    {
       q(di + 1) = rhou[di];
    }

    for (int di = 0; di < dim+3; ++di)
    {
       v(di) = vec_pert[di];
    }

    mfem::Vector sacs(13);
    // create SA parameter vector
    for (int di = 0; di < 13; ++di)
    {
       sacs(di) = sa_params[di];
    }

    adept::Stack stack;
    DYNAMIC_SECTION("Jacobians w.r.t grad is correct")
    {
        // compute the jacobian
        stack.new_recording();
        mfem::Vector dSrcdq(dim+3);
        std::vector<adouble> sacs_a(sacs.Size());
        adept::set_values(sacs_a.data(), sacs.Size(), sacs.GetData());
        std::vector<adouble> q_a(sacs.Size());
        adept::set_values(q_a.data(), q.Size(), q.GetData());
        std::vector<adouble> grad_a(grad.Size());
        adept::set_values(grad_a.data(), grad.Size(), grad.GetData());
        adouble S_a = 0.000; 
        adouble d_a = 10.1;
        adouble mu_a = 1.1;
        adouble Re_a = 5000000;
        adouble src;// = mach::calcSADestruction<adouble, dim>(q_a.data(), S_a, mu_a, d_a, Re_a, sacs_a.data());
        src = mach::calcSASource<adouble,dim>(
         q_a.data(), grad_a.data(), sacs_a.data())/Re_a;
        // use negative model if turbulent viscosity is negative
        if (q_a[dim+2] < 0)
        {
            src += mach::calcSANegativeProduction<adouble,dim>(
                q_a.data(), S_a, sacs_a.data());
            src += mach::calcSANegativeDestruction<adouble,dim>(
                q_a.data(), d_a, sacs_a.data())/Re_a;
        }
        else
        {
            src += mach::calcSAProduction<adouble,dim>(
                q_a.data(), mu_a, d_a, S_a, Re_a, sacs_a.data())/Re_a;
            src += mach::calcSADestruction<adouble,dim>(
                q_a.data(), mu_a, d_a, S_a, Re_a, sacs_a.data())/Re_a;
        }

        stack.independent(q_a.data(), dim+3);
        stack.dependent(src);
        stack.jacobian(dSrcdq.GetData());

        double S = 0.000;
        double d = 10.1;
        double mu = 1.1;
        double Re = 5000000;
        mfem::Vector qp(q); mfem::Vector qm(q);
        qp.Add(delta, v); qm.Add(-delta, v);

        // get perturbed states conv vector
        //double srcp = mach::calcSADestruction<double, dim>(q.GetData(), Sp, mu, d, Re, sacs.GetData());
        //double srcm = mach::calcSADestruction<double, dim>(q.GetData(), Sm, mu, d, Re, sacs.GetData());

        double srcp = mach::calcSASource<double,dim>(
          qp.GetData(), grad.GetData(), sacs.GetData())/Re;
        if (qp(dim+2) < 0)
        {
            srcp += mach::calcSANegativeProduction<double,dim>(
                qp.GetData(), S, sacs.GetData());
            srcp += mach::calcSANegativeDestruction<double,dim>(
                qp.GetData(), d, sacs.GetData())/Re;
        }
        else
        {
            srcp += mach::calcSAProduction<double,dim>(
                qp.GetData(), mu, d, S, Re, sacs.GetData())/Re;
            srcp += mach::calcSADestruction<double,dim>(
                qp.GetData(), mu, d, S, Re, sacs.GetData())/Re;
        }

        double srcm = mach::calcSASource<double,dim>(
          qm.GetData(), grad.GetData(), sacs.GetData())/Re;
        if (qm(dim+2) < 0)
        {
            srcm += mach::calcSANegativeProduction<double,dim>(
                qm.GetData(), S, sacs.GetData());
            srcm += mach::calcSANegativeDestruction<double,dim>(
                qm.GetData(), d, sacs.GetData())/Re;
        }
        else
        {
            srcm += mach::calcSAProduction<double,dim>(
                qm.GetData(), mu, d, S, Re, sacs.GetData())/Re;
            srcm += mach::calcSADestruction<double,dim>(
                qm.GetData(), mu, d, S, Re, sacs.GetData())/Re;
        }

        // finite difference jacobian
        double jac_fd = srcp;
        jac_fd -= srcm;
        jac_fd /= 2.0 * delta;

        double jac_v = dSrcdq*v;

        std::cout << "FD Deriv: " << jac_fd  << std::endl; 
        std::cout << "AN Deriv: " << jac_v << std::endl; 
        REQUIRE(jac_v == Approx(jac_fd).margin(1e-10));

    }
} 