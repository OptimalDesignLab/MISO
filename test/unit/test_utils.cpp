#include <random>

#include "catch.hpp"
#include "utils.hpp"

TEST_CASE("Testing secant method", "[secant]")
{
   const double xtol = 1e-12;
   const double ftol = 1e-12;
   const int maxiter = 100;

   // First test: f(x) = e^x - 1; root = 0.0
   auto expfun = [](double x)
   {
      return exp(x) - 1;
   };
   double x = miso::secant(expfun, 0.5, M_PI, ftol, xtol, maxiter);
   std::cout << "Solution 1 = " << x << std::endl;
   REQUIRE( x == Approx(0.0).margin(xtol) );

   // Second test: f(x) = sin(e^x); one root is ln(pi)
   auto logfun = [](double x)
   {
      return sin(exp(x));
   };
   x = miso::secant(logfun, 0, M_PI, ftol, xtol, maxiter);
   std::cout << "Solution 2 = " << x << std::endl;
   REQUIRE( x == Approx(log(M_PI)).margin(xtol) );

}