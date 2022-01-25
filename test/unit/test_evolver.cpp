#include <random>

#include "catch.hpp"

#include "mfem_extensions.hpp"
#include "evolver.hpp"
#include "utils.hpp"

TEST_CASE("Testing RRKImplicitMidpointSolver", "[rrk]")
{
   const bool verbose = false; // set to true for some output 
   std::ostream *out = verbose ? mach::getOutStream(0) : mach::getOutStream(1);
   using namespace mfem;

   // For solving dense matrix systems
   class DenseMatrixSolver : public Solver
   {
   public:
      void SetOperator(const Operator &op) override
      {
         mat = dynamic_cast<const DenseMatrix*>(&op);
      }

      void Mult(const Vector &x, Vector &y) const override
      {
         y = x;
         DenseMatrix A(*mat);
         LinearSolve(A, y.GetData());
      }
   private:
      const DenseMatrix *mat;
   };

   // Operator for exponential-entropy ODE, Eq. (3.1) in Ranocha et al. 2020
   class ExponentialODE : public mach::EntropyConstrainedOperator
   {
   public:
      ExponentialODE() : EntropyConstrainedOperator(
         2, 0.0, TimeDependentOperator::Type::EXPLICIT), dt(0.0), x(2)
      {
         Jac.reset(new DenseMatrix(2));
         linear_solver.reset(new DenseMatrixSolver());
         newton_solver.reset(new NewtonSolver());
         newton_solver->SetRelTol(1e-14);
         newton_solver->SetAbsTol(1e-14);
         newton_solver->SetPrintLevel(-1);
         newton_solver->SetMaxIter(30);
         newton_solver->SetSolver(*linear_solver);
         newton_solver->SetOperator(*this);
         newton_solver->iterative_mode = false;
      }

      double Entropy(const Vector &x_) override
      {
         return exp(x_(0)) + exp(x_(1));
      }

      double EntropyChange(double dt, const Vector &x_,
                           const Vector &k_) override
      {
         Vector y(x_.Size());
         add(x_, dt, k_, y);
         return exp(y(0))*exp(y(1)) - exp(y(1))*exp(y(0)); // should be zero
      }

      void Mult(const Vector &k_, Vector &y_) const override
      {
         Vector x_new(x);
         x_new.Add(dt, k_);
         y_.SetSize(k_.Size());
         y_(0) = k_(0) + exp(x_new(1));
         y_(1) = k_(1) - exp(x_new(0));
      }

      Operator &GetGradient(const Vector &k_) const override
      {
         Vector x_new(x);
         x_new.Add(dt, k_);
         DenseMatrix &jacobian = static_cast<DenseMatrix&>(*Jac);
         jacobian = 0.0;
         jacobian(0,0) = 1.0;
         jacobian(0,1) = dt*exp(x_new(1));
         jacobian(1,0) = -dt*exp(x_new(0));
         jacobian(1,1) = 1.0;
         return *Jac;
      }

      void ImplicitSolve(const double dt_, const Vector &x_, Vector &k_) override
      {
         dt = dt_;
         x = x_;
         Vector zero;
         newton_solver->Mult(zero, k_);
         MFEM_ASSERT(newton_solver->GetConverged()==1, "Newton failed.\n");
      }

   private:
      double dt;
      Vector x;
      mutable std::unique_ptr<Operator> Jac;
      std::unique_ptr<NewtonSolver> newton_solver;
      std::unique_ptr<Solver> linear_solver;
   };

   std::unique_ptr<TimeDependentOperator> ode(new ExponentialODE());
   std::unique_ptr<ODESolver> solver(new mach::RRKImplicitMidpointSolver(out));
   solver->Init(*ode);

   double t_final = 5.0;
   int num_steps = 100.0;
   double dt = t_final/num_steps;
   double t = 0.0;
   Vector u0(2), u(2);
   u0(0) = 1.0;
   u0(1) = 0.5;
   u = u0;
   bool done = false;
   for (int ti = 0; !done;)
   {
      double dt_real = std::min(dt, t_final - t);
      *out << "iter " << ti << ": time = " << t << ": dt = " << dt_real
           << " (" << round(100 * t / t_final) << "% complete)" << std::endl;
      *out << "\tentropy = " << dynamic_cast<ExponentialODE &>(*ode).Entropy(u)
           << std::endl;
      solver->Step(u, t, dt_real);
      ti++;
      done = (t >= t_final - 1e-16);
   }

   // Check that solution is reasonable accurate
   auto exact_sol = [](double t, Vector &u)
   {
      const double e = std::exp(1.0);
      const double sepe = sqrt(e) + e;
      u.SetSize(2);
      u(0) = log(e + pow(e,1.5)) - log(sqrt(e) + exp(sepe*t));
      u(1) = log((sepe*exp(sepe*t))/(sqrt(e) + exp(sepe*t)));
   };



   Vector u_exact;
   exact_sol(t_final, u_exact);
   double error = sqrt( pow(u(0) - u_exact(0),2) + pow(u(1) - u_exact(1),2));
   double entropy0 = dynamic_cast<ExponentialODE&>(*ode).Entropy(u0);
   double entropy = dynamic_cast<ExponentialODE&>(*ode).Entropy(u);

   if (verbose)
   {
      std::cout << "discrete solution = " << u(0) << ": " << u(1) << std::endl;
      std::cout << "exact solution    = " << u_exact(0) << ": " << u_exact(1)
                << std::endl;
      std::cout << "terminal solution error = " << error << std::endl;
      std::cout << "entropy error = " << entropy - entropy0 << std::endl;
   }
   REQUIRE( error == Approx(0.003).margin(1e-4) );

   REQUIRE( entropy == Approx(entropy0).margin(1e-12) );
}