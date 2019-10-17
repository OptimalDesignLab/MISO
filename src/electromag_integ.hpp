#ifndef MACH_ELECTROMAG_INTEG
#define MACH_ELECTROMAG_INTEG

#include "mfem.hpp"

#include "coefficient.hpp"
#include "solver.hpp"

namespace mach
{

/// Integrator for (\nu(u)*curl u, curl v) for Nedelec elements
class CurlCurlNLFIntegrator : public mfem::NonlinearFormIntegrator
{
public:
	/// Construct a curl curl nonlinear form integrator for Nedelec elements
   /// \param[in] diff_stack - for algorithmic differentiation
	/// \param[in] m - model describing nonlinear material parameter
   CurlCurlNLFIntegrator(StateCoefficient *m,
								 double a = 1.0)
		: model(m), alpha(a) {}

   /// Construct the element local residual
   /// \param[in] el - the finite element whose residual we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elvect - element local residual
   virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &trans,
                                      const mfem::Vector &elfun,
                                      mfem::Vector &elvect);

   /// Construct the element local Jacobian
   /// \param[in] el - the finite element whose Jacobian we want
   /// \param[in] trans - defines the reference to physical element mapping
   /// \param[in] elfun - element local state vector
   /// \param[out] elmat - element local Jacobian
   virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                    mfem::ElementTransformation &trans,
                                    const mfem::Vector &elfun,
                                    mfem::DenseMatrix &elmat);

private:
	/// material (thus mesh) dependent model describing electromagnetic behavior
	StateCoefficient *model;
   /// scales the terms; can be used to move to rhs/lhs
	double alpha;

#ifndef MFEM_THREAD_SAFE
   mfem::DenseMatrix curlshape, curlshape_dFt, M;
   mfem::Vector b_vec, temp_vec;
#endif

};

} // namespace mach

#endif