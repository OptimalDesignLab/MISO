#include <random>
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_test_data.hpp"
#include "laplace.hpp"
#include "navier_stokes_integ.hpp"
#include "navier_stokes_fluxes.hpp"
using namespace std;
template <int deg>
void uexact(const mfem::Vector &x, mfem::Vector &u)
{
    u(0) = pow(x(0), deg);
}

TEMPLATE_TEST_CASE_SIG("ViscousIntegrator::AssembleElementVector",
                       "[ViscousIntegrator]", ((int p), p), 1, 2, 3, 4)
{
    using namespace mfem;
    const int dim = 2; // templating is hard here because mesh constructors
    int num_state = 1;
    adept::Stack diff_stack;

    // generate a 2 element mesh
    int num_edge = 1;
    std::unique_ptr<Mesh> mesh(new Mesh(num_edge, num_edge, Element::TRIANGLE,
                                        true /* gen. edges */, 1.0, 1.0, true));

    std::unique_ptr<FiniteElementCollection> fec(
        new SBPCollection(p, dim));
    std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
        mesh.get(), fec.get(), num_state, Ordering::byVDIM));

    NonlinearForm res(fes.get());
    res.AddDomainIntegrator(new mach::LaplaceIntegrator<2>(diff_stack));
    res.AddDomainIntegrator(new mach::SourceIntegrator(p));
    res.AddBdrFaceIntegrator(
        new mach::LaplaceNaturalBC<2>(diff_stack, fec.get()));
    GridFunction u(fes.get()), r(fes.get());
    VectorFunctionCoefficient u0(1, uexact<p>);
    u.ProjectCoefficient(u0);
    res.Mult(u, r);
    for (int i = 0; i < r.Size(); ++i)
    {
        //std::cout << "r(i) = " << r(i) << std::endl;
        REQUIRE(r(i) == Approx(0.0).margin(euler_data::abs_tol));
    }
}

TEMPLATE_TEST_CASE_SIG("ApplyScaling", "[ViscousScaling]",
                       ((int dim), dim), 1,2 ,3)
{
    using namespace euler_data;
    double delta = 1e-5;
    int num_states = dim + 2;
    double Re_num = 1;
    double Pr_num = 1;
    double mu = 1;
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
    // random v_mat matrix
    // spatial derivatives of entropy variables
    mfem::DenseMatrix v_mat(vec_pert, dim + 2, dim);
    // Create the AD stack, and the integrator
    adept::Stack diff_stack;
    mach::ESViscousIntegrator<dim> esviscousinteg(diff_stack, Re_num, Pr_num, mu);
    // calculate the jacobian w.r.t q
    for (int di = 0; di < dim; ++di)
    {
        DYNAMIC_SECTION("Apply scaling jacobian w.r.t state is correct")
        {
            mfem::DenseMatrix mat_vec_jac(num_states);
            esviscousinteg.applyScalingJacState(di, q, v_mat, mat_vec_jac);
            //mat_vec_jac.Print();
            // loop over each state variable and check column of mat_vec_jac...
            for (int i = 0; i < num_states; ++i)
            {
                mfem::Vector q_plus(q), q_minus(q);
                mfem::Vector mat_vec_plus(num_states), mat_vec_minus(num_states);
                q_plus(i) += delta;
                q_minus(i) -= delta;
                esviscousinteg.applyScaling(di, x, q_plus, v_mat, mat_vec_plus);
                esviscousinteg.applyScaling(di, x, q_minus, v_mat, mat_vec_minus);
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
