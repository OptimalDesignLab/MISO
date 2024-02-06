#ifndef MISO_SIMPLE_INTEG
#define MISO_SIMPLE_INTEG

#include "mfem.hpp"

#include "sbp_fe.hpp"
#include "utils.hpp"


namespace miso
{
    class SimpleIntegrator : public  mfem::NonlinearFormIntegrator
    {
    public:
    /// Construct an integrator for simple integration sources
    /// \note default constructor sets num_state to 1, x_chk vals to 0
    /// alpha to 1. and el_count to 0
    SimpleIntegrator()
    {
        num_states = 1;
        x_chk = 0.;
        alpha = 1.;
        el_count = 0;
    }

    /// Construct an integrator for simple integration sources
    /// \param[in] x - Vector that contains the physical node location to check orientation
    /// \param[in] num_state_vars - the number of state variables
    /// \note sets num_state to default 1 if not provided and assigns x_chk to x
    /// and el_count to 0
    SimpleIntegrator(mfem::Vector x, int num_state_vars = 1, double a = 1.)
        : num_states(num_state_vars), alpha(a)
    {
        x_chk = x;
        el_count = 0;
    }

    /// Get the contribution of this element to a functional
    /// \param[in] el - the finite element whose contribution we want
    /// \param[in] trans - defines the reference to physical element mapping
    /// \param[in] elfun - element local state function
    double GetElementEnergy(const mfem::FiniteElement &el,
                            mfem::ElementTransformation &trans,
                            const mfem::Vector &elfun) override
    {
        return 0.0;
    }

    /// Construct the element local residual
    /// \param[in] el - the finite element whose residual we want
    /// \param[in] trans - defines the reference to physical element mapping
    /// \param[in] elfun - element local state function
    /// \param[out] elvect - element local residual
    void AssembleElementVector(const mfem::FiniteElement &el,
                                mfem::ElementTransformation &trans,
                                const mfem::Vector &elfun,
                                mfem::Vector &elvect) override;

    /// Construct the element local Jacobian
    /// \param[in] el - the finite element whose Jacobian we want
    /// \param[in] trans - defines the reference to physical element mapping
    /// \param[in] elfun - element local state function
    /// \param[out] elmat - element local Jacobian
    void AssembleElementGrad(const mfem::FiniteElement &el,
                                mfem::ElementTransformation &trans,
                                const mfem::Vector &elfun,
                                mfem::DenseMatrix &elmat) override;

    /// increment count every time source function is called
    int incrementEl_count()
    {
        ++el_count;
        return el_count;
    }
    protected:
    /// number of states
    int num_states;
    /// the physical node location to check orientation
    mfem::Vector x_chk;
    /// scales the terms; can be used to move to rhs/lhs
    double alpha;
    /// counting number of elements to assign respective source terms
    int el_count;
    #ifndef MFEM_THREAD_SAFE
    /// the coordinates of node i
    mfem::Vector x_i;
    /// the source for node i
    mfem::Vector src_i;
    #endif

    /// The source function used to set specific values
    /// \param[in] x - spatial location at which to evaluate the source
    /// \param[out] src - source term evaluated at `x`
    /// \note This uses the CRTP, so it wraps a call to `calcSource` in Derived.
    void source(const mfem::Vector &x, mfem::Vector &src, int el_id)
    {
        src = 1.0;
    }

    };

    void SimpleIntegrator::AssembleElementVector(
        const mfem::FiniteElement &el,
        mfem::ElementTransformation &trans,
        const mfem::Vector &elfun,
        mfem::Vector &elvect)
    {
    using namespace mfem;
    const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
    // const IntegrationRule &ir = sbp.GetNodes();
    int num_nodes = sbp.GetDof();
    // int dim = sbp.GetDim();
    #ifdef MFEM_THREAD_SAFE
    Vector x_i, src_i;
    #endif
    src_i.SetSize(num_states);
    elvect.SetSize(num_states * num_nodes);
    DenseMatrix res(elvect.GetData(), num_nodes, num_states);
    elvect = 0.0;
    int el_id = incrementEl_count();
    for (int i = 0; i < num_nodes; ++i)
    {
        const IntegrationPoint &ip = el.GetNodes().IntPoint(i);
        trans.SetIntPoint(&ip);
        trans.Transform(ip, x_i);
        double weight = trans.Weight() * ip.weight;
        source(x_i, src_i, el_id);
        for (int n = 0; n < num_states; ++n)
        {
            //res(i, n) += weight * src_i(n);
            res(i, n) += src_i(n);
        }
    }
    res *= alpha;
    }

    void SimpleIntegrator::AssembleElementGrad(
        const mfem::FiniteElement &el,
        mfem::ElementTransformation &trans,
        const mfem::Vector &elfun,
        mfem::DenseMatrix &elmat)
    {
    using namespace mfem;
    const auto &sbp = dynamic_cast<const SBPFiniteElement &>(el);
    // const IntegrationRule &ir = sbp.GetNodes();
    int num_nodes = sbp.GetDof();
    elmat.SetSize(num_states * num_nodes);
    elmat = 0.0;
    }

} // namespace miso

#endif