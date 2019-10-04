#ifndef ELECTROMAG_TEST_DATA
#define ELECTROMAG_TEST_DATA

#include <limits>
#include <random>
#include "mfem.hpp"

#include "coefficient.hpp"

namespace electromag_data
{
// define the random-number generator; uniform between 0 and 1
static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

// template <int dim>
void randBaselinePert(const mfem::Vector &x, mfem::Vector &u)
{
   const double scale = 0.01;
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = (1.0 + scale*uniform_rand(gen));
   }
}
// explicit instantiation of the templated function above
// template void randBaselinePert<1>(const mfem::Vector &x, mfem::Vector &u);
// template void randBaselinePert<2>(const mfem::Vector &x, mfem::Vector &u);
// template void randBaselinePert<3>(const mfem::Vector &x, mfem::Vector &u);

void randState(const mfem::Vector &x, mfem::Vector &u)
{
	// std::cout << "u size: " << u.Size() << std::endl;
   for (int i = 0; i < u.Size(); ++i)
   {
		// std::cout << i << std::endl;
      u(i) = uniform_rand(gen);
   }
}

// double randState(const mfem::Vector &x)
// {
// 	// std::cout << "u size: " << u.Size() << std::endl;
//    // for (int i = 0; i < u.Size(); ++i)
//    // {
// 	// 	// std::cout << i << std::endl;
//    return uniform_rand(gen);
//    // }
// }

/// Simple linear coefficient for testing CurlCurlNLFIntegrator
class LinearCoefficient : public mach::ExplicitStateDependentCoefficient
{
public:
	LinearCoefficient(double val = 1.0) : value(val) {}

	double Eval(mfem::ElementTransformation &trans,
					const mfem::IntegrationPoint &ip) override
	{
		return value;
	}

	double EvalStateDeriv(mfem::ElementTransformation &trans,
								 const mfem::IntegrationPoint &ip) override
	{
		return 0.0;
	}

private:
	double value;
};

/// Simple nonlinear coefficient for testing CurlCurlNLFIntegrator
class NonLinearCoefficient : public mach::ExplicitStateDependentCoefficient
{
public:
	NonLinearCoefficient(mfem::GridFunction *state_)
	 : stateGF(state_) {}

	double Eval(mfem::ElementTransformation &trans,
					const mfem::IntegrationPoint &ip) override
	{
		mfem::Vector state;
		stateGF->GetVectorValue(trans.ElementNo, ip, state);
		double state_mag = state.Norml2();
		return std::exp(-state_mag);
	}

	double EvalStateDeriv(mfem::ElementTransformation &trans,
								 const mfem::IntegrationPoint &ip) override
	{
		mfem::Vector state;
		stateGF->GetVectorValue(trans.ElementNo, ip, state);
		double state_mag = state.Norml2();
		return -std::exp(-state_mag);
	}

private:
	mfem::GridFunction *stateGF;
};

} // namespace electromag_data

#endif