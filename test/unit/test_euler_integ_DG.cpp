#include <random>

#include "catch.hpp"
#include "mfem.hpp"
#include "euler_fluxes.hpp"
#include "euler_test_data.hpp"
#include "euler_integ_DG.hpp"

/// check DG integrators
TEMPLATE_TEST_CASE_SIG("Euler flux jacobian DG", "[euler_flux_jac]",
                       ((int dim), dim), 1, 2, 3)
{
    using namespace euler_data;
    double delta = 1e-5;
    mfem::Vector q(dim + 2);
    mfem::Vector flux(dim + 2);
    mfem::Vector nrm(dim);
    q(0) = rho;
    q(dim + 1) = rhoe;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
        nrm(di) = dir[di];
    }
    adept::Stack stack;
    mach::EulerDomainIntegrator<dim> eulerinteg(stack, dim + 2);

    SECTION(" Euler DG flux jacobian w.r.t state is correct.")
    {
        /// Create the perturbation vector
        mfem::Vector v(dim + 2);
        for (int i = 0; i < dim + 2; i++)
        {
            v[i] = vec_pert[i];
        }
        /// Create some intermediate variables
        mfem::Vector q_plus(q), q_minus(q);
        mfem::Vector flux_plus(dim + 2), flux_minus(dim + 2);
        mfem::Vector jac_v(dim + 2);
        mfem::DenseMatrix flux_jac(dim + 2);

        // calculate the flux jacobian
        eulerinteg.calcFluxJacState(nrm, q, flux_jac);
        flux_jac.Mult(v, jac_v);

        // calculate the plus and minus fluxes
        q_plus.Add(delta, v);
        q_minus.Add(-delta, v);
        eulerinteg.calcFlux(nrm, q_plus, flux_plus);
        eulerinteg.calcFlux(nrm, q_minus, flux_minus);

        // compare the difference
        mfem::Vector jac_v_fd(flux_plus);
        jac_v_fd -= flux_minus;
        jac_v_fd /= (2.0 * delta);
        mfem::Vector diff(jac_v);
        diff -= jac_v_fd;
        for (int i = 0; i < dim + 2; ++i)
        {
            REQUIRE(jac_v[i] == Approx(jac_v_fd[i]).margin(1e-10));
        }
    }
}

TEST_CASE("Vortex flux", "[IsentropricVortexBC]")
{
    const int dim = 2;
    using namespace euler_data;
    double delta = 1e-5;
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    mfem::Vector q(dim + 2);
    q(0) = rho;
    q(dim + 1) = rhoe;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
    }

    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);

    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    adept::Stack diff_stack;
    //diff_stack.deactivate();
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; p++)
    {
        DYNAMIC_SECTION("Jacobian of Isentropic Vortex BC flux w.r.t state is correct")
        {
            fec.reset(new mfem::DG_FECollection(p, dim));
            mach::EulerBoundaryIntegrator<dim, 1> isentropic_vortex(diff_stack, fec.get(), dim + 2, q);
            // create the perturbation vector
            mfem::Vector v(dim + 2);
            for (int i = 0; i < dim + 2; i++)
            {
                v(i) = vec_pert[i];
            }

            // get derivative information from AD functions and form product
            mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
            mfem::Vector jac_v_ad(dim + 2);
            isentropic_vortex.calcFluxJacState(x, nrm, q, jac_ad);
            jac_ad.Mult(v, jac_v_ad);

            // FD approximation
            mfem::Vector q_plus(q);
            mfem::Vector q_minus(q);
            q_plus.Add(delta, v);
            q_minus.Add(-delta, v);

            mfem::Vector flux_plus(dim + 2);
            mfem::Vector flux_minus(dim + 2);
            isentropic_vortex.calcFlux(x, nrm, q_plus, flux_plus);
            isentropic_vortex.calcFlux(x, nrm, q_minus, flux_minus);

            // finite difference jacobian
            mfem::Vector jac_v_fd(dim + 2);
            subtract(flux_plus, flux_minus, jac_v_fd);
            jac_v_fd /= 2 * delta;

            // compare
            for (int i = 0; i < dim + 2; ++i)
            {
                REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
            }
        }
    }
}

TEST_CASE("Slipwall flux", "[SlipwallBC]")
{
    const int dim = 2;
    using namespace euler_data;
    double delta = 1e-5;
    mfem::Vector nrm(dim);
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
    }
    mfem::Vector q(dim + 2);
    q(0) = rho;
    q(dim + 1) = rhoe;
    for (int di = 0; di < dim; ++di)
    {
        q(di + 1) = rhou[di];
    }

    // dummy const vector x for calcFlux - unused
    const mfem::Vector x(nrm);

    /// finite element or SBP operators
    std::unique_ptr<mfem::FiniteElementCollection> fec;
    adept::Stack diff_stack;
    //diff_stack.deactivate();
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; p++)
    {
        DYNAMIC_SECTION("Jacobian of Slipwall BC flux w.r.t state is correct")
        {
            fec.reset(new mfem::DG_FECollection(p, dim));
            mach::EulerBoundaryIntegrator<dim, 2> isentropic_vortex(diff_stack, fec.get(), dim + 2, q);
            // create the perturbation vector
            mfem::Vector v(dim + 2);
            for (int i = 0; i < dim + 2; i++)
            {
                v(i) = vec_pert[i];
            }

            // get derivative information from AD functions and form product
            mfem::DenseMatrix jac_ad(dim + 2, dim + 2);
            mfem::Vector jac_v_ad(dim + 2);
            isentropic_vortex.calcFluxJacState(x, nrm, q, jac_ad);
            jac_ad.Mult(v, jac_v_ad);

            // FD approximation
            mfem::Vector q_plus(q);
            mfem::Vector q_minus(q);
            q_plus.Add(delta, v);
            q_minus.Add(-delta, v);

            mfem::Vector flux_plus(dim + 2);
            mfem::Vector flux_minus(dim + 2);
            isentropic_vortex.calcFlux(x, nrm, q_plus, flux_plus);
            isentropic_vortex.calcFlux(x, nrm, q_minus, flux_minus);

            // finite difference jacobian
            mfem::Vector jac_v_fd(dim + 2);
            subtract(flux_plus, flux_minus, jac_v_fd);
            jac_v_fd /= 2 * delta;

            // compare
            for (int i = 0; i < dim + 2; ++i)
            {
                REQUIRE(jac_v_ad(i) == Approx(jac_v_fd(i)).margin(1e-10));
            }
        }
    }
}

TEMPLATE_TEST_CASE_SIG("Roe face-flux Jacobian", "[Roe-face]",
                       ((int dim), dim), 2)
{
    using namespace euler_data;
    // copy the data into mfem vectors for convenience
    mfem::Vector qL(dim + 2);
    mfem::Vector qR(dim + 2);
    mfem::Vector flux(dim + 2);
    mfem::Vector flux_plus(dim + 2);
    mfem::Vector flux_minus(dim + 2);
    mfem::Vector v(dim + 2);
    mfem::Vector jac_v(dim + 2);
    mfem::Vector nrm(dim);
    mfem::DenseMatrix jacL(dim + 2, 2 * (dim + 2));
    mfem::DenseMatrix jacR(dim + 2, 2 * (dim + 2));
    double delta = 1e-5;
    qL(0) = rho;
    qL(dim + 1) = rhoe;
    qR(0) = rho2;
    qR(dim + 1) = rhoe2;
    for (int di = 0; di < dim; ++di)
    {
        nrm(di) = dir[di];
        qL(di + 1) = rhou[di];
        qR(di + 1) = rhou2[di];
    }
    // create perturbation vector
    for (int di = 0; di < dim + 2; ++di)
    {
        v(di) = vec_pert[di];
    }
    // perturbed vectors
    mfem::Vector qL_plus(qL), qL_minus(qL);
    mfem::Vector qR_plus(qR), qR_minus(qR);
    adept::Stack diff_stack;
    double diss_coeff = 1.0;
    const int max_degree = 4;
    for (int p = 1; p <= max_degree; p++)
    {
        std::unique_ptr<mfem::FiniteElementCollection> fec(
            new mfem::DG_FECollection(1, dim));
        mach::EulerFaceIntegrator<dim, false> roefaceinteg(diff_stack, fec.get(), dim + 2);
        // +ve perturbation
        qL_plus.Add(delta, v);
        qR_plus.Add(delta, v);
        // -ve perturbation
        qL_minus.Add(-delta, v);
        qR_minus.Add(-delta, v);
        for (int di = 0; di < dim; ++di)
        {
            DYNAMIC_SECTION("Roe face flux Jacobian is correct w.r.t left state ")
            {
                // get perturbed states flux vector
                roefaceinteg.flux(nrm, qL_plus, qR, flux_plus);
                roefaceinteg.flux(nrm, qL_minus, qR, flux_minus);
                // compute the jacobian
                roefaceinteg.calcFluxJacState(nrm, qL, qR, jacL, jacR);
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
            DYNAMIC_SECTION("Roe face flux Jacobian is correct w.r.t right state ")
            {
                // get perturbed states flux vector
                roefaceinteg.flux(nrm, qL, qR_plus, flux_plus);
                roefaceinteg.flux(nrm, qL, qR_minus, flux_minus);
                // compute the jacobian
                roefaceinteg.calcFluxJacState(nrm, qL, qR, jacL, jacR);
                jacR.Mult(v, jac_v);
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
}