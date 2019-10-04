#ifndef ELECTROMAG_TEST_DATA
#define ELECTROMAG_TEST_DATA

#include <limits>
#include <random>
#include "mfem.hpp"

namespace electromag_data
{
// define the random-number generator; uniform between 0 and 1
static std::default_random_engine gen;
static std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

// template <int dim>
// void randBaselinePert(const mfem::Vector &x, mfem::Vector &u)
// {
//    const double scale = 0.01;
//    u(0) = rho*(1.0 + scale*uniform_rand(gen));
//    u(dim+1) = rhoe*(1.0 + scale*uniform_rand(gen));
//    for (int di = 0; di < dim; ++di)
//    {
//       u(di+1) = rhou[di]*(1.0 + scale*uniform_rand(gen));
//    }
// }
// // explicit instantiation of the templated function above
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

} // namespace electromag_data

#endif