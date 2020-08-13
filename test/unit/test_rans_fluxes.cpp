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
        mach::calcGradJacDw<dim>(stack, Dw.GetData(), trans_jac.GetData(), 
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
            mach::calcGrad<double, dim>(Dw_plus.GetData(), trans_jac.GetData(),  
                            grad_plus.GetData());
            mach::calcGrad<double, dim>(Dw_minus.GetData(), trans_jac.GetData(),  
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