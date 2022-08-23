#include <random>

#include "catch.hpp"
#include "mfem.hpp"

#include "linesearch.hpp"
#include "relaxed_newton.hpp"

namespace
{
class FunctionalOperator : public mfem::Operator
{
public:
   FunctionalOperator(std::function<void(const mfem::Vector &x,
                                         mfem::Vector &res)> calcRes,
                      std::function<void(const mfem::Vector &x,
                                         mfem::DenseMatrix &jac)> calcJac)
   : Operator(2), calcRes(std::move(calcRes)), calcJac(std::move(calcJac))
   {}

   void Mult(const mfem::Vector &x, mfem::Vector &y) const override
   {
      calcRes(x, y);
   }

   Operator & GetGradient(const mfem::Vector &x) const override
   {
      jac.SetSize(x.Size());
      calcJac(x, jac);
      return jac;
   }

private:
   std::function<void(const mfem::Vector &x, mfem::Vector &res)> calcRes;
   std::function<void(const mfem::Vector &x, mfem::DenseMatrix &jac)> calcJac;
   mutable mfem::DenseMatrix jac;
};

std::default_random_engine gen;
std::uniform_real_distribution<double> uniform_rand(-1.0,1.0);

/// Gradient for Rosenbrock function:
/// f = 100*(x_1 - x_0^2)^2 + (1-x_0)^2
auto calcRes = [](const mfem::Vector &x, mfem::Vector &res)
{
   res(0) = -400*x(0)*(x(1)-pow(x(0),2)) - 2*(1-x(0));
   res(1) = 200*(x(1)-pow(x(0),2));
};

/// Hessian for Rosenbrock function:
/// f = 100*(x_1 - x_0^2)^2 + (1-x_0)^2
auto calcJac = [](const mfem::Vector &x, mfem::DenseMatrix &jac)
{
   jac(0,0) = 1200*pow(x(0), 2) - 400*x(1) + 2;
   jac(0,1) = -400*x(0);
   jac(1,0) = -400*x(0);
   jac(1,1) = 200;
};

}  // anonymous namespace

TEST_CASE("Rosenbrock calcJac")
{
   mfem::Vector state(2);
   state(0) = uniform_rand(gen);
   state(1) = uniform_rand(gen);

   mfem::DenseMatrix jac(2);
   calcJac(state, jac);

   const double delta = 1e-5;

   mfem::Vector residual0p(2);
   state(0) += delta;
   calcRes(state, residual0p);

   mfem::Vector residual0m(2);
   state(0) -= 2*delta;
   calcRes(state, residual0m);
   state(0) += delta;

   mfem::Vector residual1p(2);
   state(1) += delta;
   calcRes(state, residual1p);

   mfem::Vector residual1m(2);
   state(1) -= 2*delta;
   calcRes(state, residual1m);
   state(1) += delta;

   mfem::DenseMatrix jac_fd(2);
   jac_fd(0,0) = (residual0p(0) - residual0m(0)) / (2*delta);
   jac_fd(0,1) = (residual0p(1) - residual0m(1)) / (2*delta);
   jac_fd(1,0) = (residual1p(0) - residual1m(0)) / (2*delta);
   jac_fd(1,1) = (residual1p(1) - residual1m(1)) / (2*delta);

   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         REQUIRE(jac(i,j) == Approx(jac_fd(i,j)));
      }
   }
}

TEST_CASE("Phi::dphi0")
{
   mfem::Vector state(2);
   state(0) = uniform_rand(gen);
   state(1) = uniform_rand(gen);

   mfem::Vector residual(2);
   calcRes(state, residual);

   mfem::DenseMatrix jac(2);
   calcJac(state, jac);

   mfem::Vector descent_dir(2);
   // descent_dir(0) = uniform_rand(gen);
   // descent_dir(1) = uniform_rand(gen);

   /// If the descent direction is the result of a Newton step, dphi(0) = -phi(0)
   mfem::DenseMatrix jac_inv(2);
   CalcInverse(jac, jac_inv);
   jac_inv.Mult(residual, descent_dir);

   auto phi = mach::Phi(calcRes, state, descent_dir, residual, jac);
   double phi0 = phi.phi0;
   double dphi0 = phi.dphi0;

   const double delta = 1e-7;
   double phip = phi(delta);
   double dphi0_fd = (phip - phi0)/delta;
   REQUIRE(dphi0 == Approx(dphi0_fd));
   std::cout.precision(20);
   std::cout << "phi0 = " << phi0 << "\n"; 
   std::cout << "dphi0 = " << dphi0 << "\n";
   std::cout << "dphi0_fd = " << dphi0_fd << "\n";
}

TEST_CASE("RelaxedNewton with BacktrackingLineSearch")
{
   FunctionalOperator oper(calcRes, calcJac);

   auto newton_opts = R"(
   {
      "linesearch": {
         "type": "backtracking",
         "mu": 1e-4,
         "rhohi": 0.9,
         "rholo": 0.1,
         "interp-order": 3
      }
   })"_json;

   mach::RelaxedNewton newton(MPI_COMM_SELF, newton_opts);
   // mfem::NewtonSolver newton(MPI_COMM_SELF);
   newton.SetPrintLevel(mfem::IterativeSolver::PrintLevel().All());
   newton.SetMaxIter(500);
   newton.SetAbsTol(1e-6);
   newton.SetRelTol(1e-6);
   newton.SetOperator(oper);

   mfem::CGSolver cg(MPI_COMM_SELF);
   // cg.SetPrintLevel(mfem::IterativeSolver::PrintLevel().All());
   newton.SetSolver(cg);

   mfem::Vector zero;
   mfem::Vector state(2);
   state(0) = 1.2;
   state(1) = 1.2;

   newton.Mult(zero, state);

   REQUIRE(newton.GetFinalNorm() == Approx(0.00010890852988421259));
   REQUIRE(newton.GetNumIterations() == 59);
   REQUIRE(state(0) == Approx(1.00001197));
   REQUIRE(state(1) == Approx(1.00002375));

   state(0) = -1.2;
   state(1) = 1.0;

   newton.Mult(zero, state);

   REQUIRE(newton.GetFinalNorm() == Approx(1.4199943164140142e-05));
   REQUIRE(newton.GetNumIterations() == 244);
   REQUIRE(state(0) == Approx(0.99999982));
   REQUIRE(state(1) == Approx(0.99999961));
}