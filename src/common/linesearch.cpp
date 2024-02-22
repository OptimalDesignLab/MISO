#include <functional>

#include "mfem.hpp"

#include "linesearch.hpp"

namespace
{
/// create quadratic interpolant to phi(0), dphi(0), and phi(alpha) and return
/// its analytical minimum
double quadratic_interp(double phi0,
                        double dphi0,
                        double phi_alpha,
                        double alpha)
{
   auto c = phi0;
   auto b = dphi0;
   auto a = (phi_alpha - b * alpha - c) / (pow(alpha, 2));

   return -b / (2 * a);
}

/// create cubic interpolant to phi(0), dphi(0), phi(alpha1), and phi(alpha2)
/// and return its analytical minimum
double cubic_interp(double phi0,
                    double dphi0,
                    double phi_alpha1,
                    double alpha1,
                    double phi_alpha2,
                    double alpha2)
{
   auto denom = pow(alpha1, 2) * pow(alpha2, 2) * (alpha2 - alpha1);
   auto a = (pow(alpha1, 2) * (phi_alpha2 - phi0 - dphi0 * alpha2) -
             pow(alpha2, 2) * (phi_alpha1 - phi0 - dphi0 * alpha1)) /
            denom;

   auto b = (-pow(alpha1, 3) * (phi_alpha2 - phi0 - dphi0 * alpha2) +
             pow(alpha2, 3) * (phi_alpha1 - phi0 - dphi0 * alpha1)) /
            denom;

   if (abs(a) < std::numeric_limits<double>::epsilon())
   {
      return dphi0 / (2 * b);
   }
   else
   {
      // discriminant
      auto d = std::max(pow(b, 2) - 3 * a * dphi0, 0.0);
      // quadratic equation root
      return (-b + sqrt(d)) / (3 * a);
   }
}

}  // namespace

namespace mach
{
double BacktrackingLineSearch::search(const std::function<double(double)> &phi,
                                      double phi0,
                                      double dphi0,
                                      double alpha)
{
   auto alpha1 = alpha;
   auto alpha2 = alpha;
   auto phi2 = phi(alpha2);
   auto phi1 = phi2;

   int iter = 0;
   // std::cout << "linesearch iter 0: phi(0) = " << phi0
   //           << ", dphi(0)/dalpha = " << dphi0 << "\n";
   while (phi2 > phi0 + mu * alpha2 * dphi0)
   {
      iter += 1;
      // std::cout << "linesearch iter " << iter << ": alpha = " << alpha2
      //           << ", phi(alpha) = " << phi2 << "\n";
      if (iter > max_iter)
      {
         // std::cout << "Max iterations reached!\n";
         // return alpha2;

         // return 0.0 indicating the linesearch failed to improve,
         // so then Newton will terminate
         return 0.0;
      }

      double alpha_tmp = 0.0;
      if (iter == 1 || interp_order == 2)
      {
         alpha_tmp = quadratic_interp(phi0, dphi0, phi2, alpha2);
      }
      else
      {
         alpha_tmp = cubic_interp(phi0, dphi0, phi1, alpha1, phi2, alpha2);
      }
      alpha1 = alpha2;
      phi1 = phi2;

      alpha_tmp = std::min(alpha_tmp, alpha2 * rho_hi);
      alpha2 = std::max(alpha_tmp, alpha2 * rho_lo);

      phi2 = phi(alpha2);
   }
   // std::cout << "Solved backtrack linesearch (alpha_star = " << alpha2
   //           << ", phi(alpha_star) = " << phi2 << ") in " << iter
   //           << " iterations!\n";
   return alpha2;
}

Phi::Phi(std::function<void(const mfem::Vector &x, mfem::Vector &res)> calcRes,
         const mfem::Vector &state,
         const mfem::Vector &descent_dir,
         mfem::Vector &residual,
         mfem::Operator &jac)
 : calcRes(std::move(calcRes)),
   state(state),
   descent_dir(descent_dir),
   scratch(state.Size()),
   residual(residual),
   phi0(residual.Norml2()),
   dphi0(
       [&]()
       {
          jac.Mult(descent_dir, scratch);
          return -(scratch * residual) / phi0;
          //  return -phi0;
       }())
{ }

double Phi::operator()(double alpha)
{
   scratch = 0.0;
   add(state, -alpha, descent_dir, scratch);

   calcRes(scratch, residual);
   return residual.Norml2();
}

}  // namespace mach
