#include <random>
#include "catch.hpp"
#include "mfem.hpp"
#include "euler_test_data.hpp"
#include "laplace.hpp"

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
    Mesh mesh(Mesh::MakeCartesian2D(num_edge, num_edge, Element::TRIANGLE,
                                    true /* gen. edges */, 1.0, 1.0, true));

    std::unique_ptr<FiniteElementCollection> fec(
        new SBPCollection(p, dim));
    std::unique_ptr<FiniteElementSpace> fes(new FiniteElementSpace(
        &mesh, fec.get(), num_state, Ordering::byVDIM));

    NonlinearForm res(fes.get());
    res.AddDomainIntegrator(new miso::LaplaceIntegrator<2>(diff_stack));
    res.AddDomainIntegrator(new miso::SourceIntegrator(p));
    res.AddBdrFaceIntegrator(
        new miso::LaplaceNaturalBC<2>(diff_stack, fec.get()));
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
