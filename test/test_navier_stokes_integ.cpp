#include <random>
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_test_data.hpp"
#include "navier_stokes_integ.hpp"
#include "navier_stokes_fluxes.hpp"

using namespace std;

TEMPLATE_TEST_CASE_SIG("ESViscousIntegrator::applyScalingJacState", "[ESViscousIntegrator]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = -1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector x(num_states - 2);
    q(0) = rho;
    q(dim + 1) = rhoe;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        x(di) = 0;
    }
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    mach::ESViscousIntegrator<dim> esviscousinteg(diff_stack, Re_num, Pr_num, mu);
    // calculate the jacobian w.r.t q
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Apply scaling jacobian w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            esviscousinteg.applyScalingJacState(di, x, q, delw, mat_vec_jac);
            //mat_vec_jac.Print();
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                esviscousinteg.applyScaling(di, x, q_plus, delw, mat_vec_plus);
                esviscousinteg.applyScaling(di, x, q_minus, delw, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int j = 0; j < num_states; j++)
                {
                    REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("ESViscousIntegrator::applyScalingJacDw", "[ESViscousIntegrator]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = -1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector v(num_states);
    mfem::Vector x(num_states - 2);
    q(0) = rho;
    q(dim + 1) = rhoe;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        x(di) = 0;
    }
    for (int di = 0; di < dim + 2; ++di)
    {
        v(di) = vec_pert[di];
    }
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    mach::ESViscousIntegrator<dim> esviscousinteg(diff_stack, Re_num, Pr_num, mu);
    // calculate the jacobian w.r.t q
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Apply scaling jacobian w.r.t Dw is correct")
        {
            std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
            for (int d = 0; d < dim; ++d)
            {
                mat_vec_jac[d].SetSize(num_states);
            }
            // get jacobian from adept
            esviscousinteg.applyScalingJacDw(di, x, q, delw, mat_vec_jac);
            // matrix perturbation reshaped into vector
            mfem::Vector v_vec(vec_pert, num_states);
            for (int d = 0; d < dim; ++d)
            {
                mfem::Vector mat_vec_jac_v(num_states);
                mat_vec_jac[d].Mult(v_vec, mat_vec_jac_v);
                // perturb one column of delw everytime
                mfem::DenseMatrix delw_plus(delw), delw_minus(delw);
                for (int s = 0; s < num_states; ++s)
                {
                    delw_plus.GetColumn(d)[s] += v(s) * delta;
                    delw_minus.GetColumn(d)[s] -= v(s) * delta;
                }
                // get finite-difference jacobian
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                esviscousinteg.applyScaling(di, x, q, delw_plus, mat_vec_plus);
                esviscousinteg.applyScaling(di, x, q, delw_minus, mat_vec_minus);
                mfem::Vector mat_vec_jac_v_fd(num_states);
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_jac_v_fd);
                mat_vec_jac_v_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int s = 0; s < num_states; s++)
                {
                    REQUIRE(mat_vec_jac_v(s) == Approx(mat_vec_jac_v_fd(s)));
                }
            } // d loop
        }     // section
    }         // di loop
} // test case

TEMPLATE_TEST_CASE_SIG("Noslip Jacobian", "[NoSlipAdiabaticWallBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = -1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector q_ref(num_states);
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    q_ref(0) = rhou[0];
    q_ref(dim + 1) = rhou[dim - 1];
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        q_ref(di + 1) = 0.0;
    }
    // random delw matrix
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::NoSlipAdiabaticWallBC<dim> noslipadiabatic(diff_stack, fec.get(), Re_num, Pr_num, q_ref, mu);
        DYNAMIC_SECTION("jacobian of no slip adiabatic wall w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            noslipadiabatic.calcFluxJacState(x, nrm, jac, q, delw, mat_vec_jac);
            //mat_vec_jac.Print();
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                noslipadiabatic.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
                noslipadiabatic.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int j = 0; j < num_states; j++)
                {
                    REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("Noslip Jacobian w.r.t Dw", "[NoSlipAdiabaticWallBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = -1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector q_ref(num_states);
    mfem::Vector nrm(dim);
    mfem::Vector v(num_states);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    q_ref(0) = rhou[0];
    q_ref(dim + 1) = rhou[dim - 1];
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        q_ref(di + 1) = 0.0;
    }
    for (int di = 0; di < dim + 2; ++di)
    {
        v(di) = vec_pert[di];
    }
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::NoSlipAdiabaticWallBC<dim> noslipadiabatic(diff_stack, fec.get(), Re_num, Pr_num, q_ref, mu);
        DYNAMIC_SECTION("jacobian of no slip adiabatic wall w.r.t Dw is correct")
        {
            std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
            for (int d = 0; d < dim; ++d)
            {
                mat_vec_jac[d].SetSize(num_states);
            }
            noslipadiabatic.calcFluxJacDw(x, nrm, jac, q, delw, mat_vec_jac);
            // loop over each state variable and check column of mat_vec_jac...
            // matrix perturbation reshaped into vector
            mfem::Vector v_vec(vec_pert, num_states);
            for (int d = 0; d < dim; ++d)
            {
                mfem::Vector mat_vec_jac_v(num_states);
                mat_vec_jac[d].Mult(v_vec, mat_vec_jac_v);
                // perturb one column of delw everytime
                mfem::DenseMatrix delw_plus(delw), delw_minus(delw);
                for (int s = 0; s < num_states; ++s)
                {
                    delw_plus.GetColumn(d)[s] += v(s) * delta;
                    delw_minus.GetColumn(d)[s] -= v(s) * delta;
                }
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                noslipadiabatic.calcFlux(x, nrm, jac, q, delw_plus, mat_vec_plus);
                noslipadiabatic.calcFlux(x, nrm, jac, q, delw_minus, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int s = 0; s < num_states; ++s)
                {
                    REQUIRE(mat_vec_jac_v(s) == Approx(mat_vec_fd(s)));
                }
            } // d loop
        }     // section
    }         // p loop
} // test case

TEMPLATE_TEST_CASE_SIG("Slip wall Jacobian states", "[ViscousSlipWallBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = 1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector q_ref(num_states);
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    q_ref(0) = rhou[0];
    q_ref(dim + 1) = rhou[dim - 1];
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        q_ref(di + 1) = 0.0;
    }
    // random delw matrix
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::ViscousSlipWallBC<dim> viscousslipwall(diff_stack, fec.get(), Re_num, Pr_num, mu);
        DYNAMIC_SECTION("jacobian of Viscous Slip Wall BC w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            viscousslipwall.calcFluxJacState(x, nrm, q, delw, mat_vec_jac);
            //mat_vec_jac.Print();
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                viscousslipwall.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
                viscousslipwall.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int j = 0; j < num_states; j++)
                {
                    REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
                }
            }
        }
    }
}

#if 0
TEMPLATE_TEST_CASE_SIG("Slip wall Jacobian Dw", "[ViscousSlipWallBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = 1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector v(num_states);
    mfem::Vector q_ref(num_states);
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    q_ref(0) = rhou[0];
    q_ref(dim + 1) = rhou[dim - 1];
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        q_ref(di + 1) = 0.0;
    }
    for (int di = 0; di < dim + 2; ++di)
    {
        v(di) = vec_pert[di];
    }
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::ViscousSlipWallBC<dim> viscousslipwall(diff_stack, fec.get(), Re_num, Pr_num, mu);
        DYNAMIC_SECTION("jacobian of Viscous Slip Wall BC w.r.t Dw is correct")
        {
            std::vector<mfem::DenseMatrix> mat_vec_jac(dim);
            for (int d = 0; d < dim; ++d)
            {
                mat_vec_jac[d].SetSize(num_states);
            }
            viscousslipwall.calcFluxJacDw(x, nrm, q, delw, mat_vec_jac);
            // loop over each state variable and check column of mat_vec_jac...
            // matrix perturbation reshaped into vector
            mfem::Vector v_vec(vec_pert, num_states);
            for (int d = 0; d < dim; ++d)
            {
                mfem::Vector mat_vec_jac_v(num_states);
                mat_vec_jac[d].Mult(v_vec, mat_vec_jac_v);
                // perturb one column of delw everytime
                mfem::DenseMatrix delw_plus(delw), delw_minus(delw);
                for (int s = 0; s < num_states; ++s)
                {
                    delw_plus.GetColumn(d)[s] += v(s) * delta;
                    delw_minus.GetColumn(d)[s] -= v(s) * delta;
                }
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                viscousslipwall.calcFlux(x, nrm, jac, q, delw_plus, mat_vec_plus);
                viscousslipwall.calcFlux(x, nrm, jac, q, delw_minus, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int s = 0; s < num_states; ++s)
                {
                    REQUIRE(mat_vec_jac_v(s) == Approx(mat_vec_fd(s)));
                }
            }
        }
    }
}
#endif
TEMPLATE_TEST_CASE_SIG("Viscous inflow Jacobian", "[ViscousInflowBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = 1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector q_in(num_states);
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    q_in(0) = rho2;        //1.2;
    q_in(dim + 1) = rhoe2; //1;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        q_in(di + 1) = rhou2[di]; //1;
    }
    // random delw matrix
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::ViscousInflowBC<dim> viscousinflow(diff_stack, fec.get(), Re_num, Pr_num, q_in, mu);
        DYNAMIC_SECTION("jacobian of viscous inflow bc w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            viscousinflow.calcFluxJacState(x, nrm, q, delw, mat_vec_jac);
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                viscousinflow.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
                viscousinflow.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int j = 0; j < num_states; j++)
                {
                    REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("Viscous outflow Jacobian", "[ViscousOutflowBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = 1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector q_out(num_states);
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    q_out(0) = rho2;        //1.2;
    q_out(dim + 1) = rhoe2; //1;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        q_out(di + 1) = rhou2[di]; //1;
    }
    // random delw matrix
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::ViscousOutflowBC<dim> viscousoutflow(diff_stack, fec.get(), Re_num, Pr_num, q_out, mu);
        DYNAMIC_SECTION("jacobian of viscous outflow bc w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            viscousoutflow.calcFluxJacState(x, nrm, q, delw, mat_vec_jac);
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                viscousoutflow.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
                viscousoutflow.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int j = 0; j < num_states; j++)
                {
                    REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("Viscous farfield Jacobian", "[ViscousFarFieldBC]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = 1;
    double jac = 1;
    // construct state vec
    mfem::Vector q(num_states);
    mfem::Vector qfs(num_states);
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    q(0) = rho;
    q(dim + 1) = rhoe;
    qfs(0) = rho2;        //1.2;
    qfs(dim + 1) = rhoe2; //1;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        qfs(di + 1) = rhou2[di]; //1;
    }
    // random delw matrix
    // spatial derivatives of entropy variables
    mfem::DenseMatrix delw(delw_data, dim + 2, dim);
    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);
    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; ++p)
    {
        fec.reset(new mfem::SBPCollection(p, dim));
        mach::ViscousFarFieldBC<dim> viscousfarfield(diff_stack, fec.get(), Re_num, Pr_num, qfs, mu);
        DYNAMIC_SECTION("jacobian of viscous farfield bc w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            viscousfarfield.calcFluxJacState(x, nrm, q, delw, mat_vec_jac);
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                viscousfarfield.calcFlux(x, nrm, jac, q_plus, delw, mat_vec_plus);
                viscousfarfield.calcFlux(x, nrm, jac, q_minus, delw, mat_vec_minus);
                mfem::Vector mat_vec_fd(num_states);
                mat_vec_fd = 0.0;
                subtract(mat_vec_plus, mat_vec_minus, mat_vec_fd);
                mat_vec_fd /= 2.0 * delta;
                // compare with explicit Jacobian
                for (int j = 0; j < num_states; j++)
                {
                    REQUIRE(mat_vec_jac(j, i) == Approx(mat_vec_fd(j)));
                }
            }
        }
    }
}

TEST_CASE("ESViscousIntegrator::AssembleElementGrad", "[ESViscousIntegrator]")
{
    using namespace mfem;
    using namespace euler_data;

    const int dim = 2; // templating is hard here because mesh constructors
    int num_state = dim + 2;
    const double Re_num = 1.0;
    const double Pr_num = 1.0;
    const double vis = -1.0; // use Sutherland's law
    adept::Stack diff_stack;
    double delta = 1e-5;

    // generate a 2 element mesh
    int num_edge = 1;
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
            res.AddDomainIntegrator(
                new mach::ESViscousIntegrator<2>(diff_stack, Re_num, Pr_num, vis));

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
                REQUIRE(jac_v(i) == Approx(jac_v_fd(i)));
            }
        }
    }
}