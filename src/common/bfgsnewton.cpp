#include "bfgsnewton.hpp"
#include "utils.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace mfem
{

BFGSNewtonSolver::BFGSNewtonSolver(double a_init, double a_max, double cc1,
                                   double cc2, double max)
{
   alpha_init = a_init;
   alpha_max = a_max;
   c1 = cc1;
   c2 = cc2;
   print_level = 1;
   abs_tol = 1e-12;
   rel_tol = 1e-6;
   max_iter = max;
   zoom_max_iter = 50;
}

void BFGSNewtonSolver::SetOperator(const Operator &op)
{
   oper = dynamic_cast<const LinearOptimizer*>(&op);
}

void BFGSNewtonSolver::Mult(Vector &x, Vector &opt)
{
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");

   std::cout << "Beginning of BFGS Newton..." << '\n';
   numvar = x.Size();
   opt.SetSize(numvar);
   Vector c(numvar);
   // initialize the hessian inverse as the identity matrix
   DenseMatrix ident(numvar);
   DenseMatrix s(numvar,1);
   DenseMatrix y(numvar,1);
   B.SetSize(numvar);
   jac.SetSize(numvar);
   jac_new.SetSize(numvar);
   
   // initialize the hessian approximation
   for (int i = 0; i < numvar; i++)
   {
      B(i,i) = 1.0;
      ident(i,i) = 1.0;
   }
   int it;
   double norm0, norm_goal;

   norm0 = norm = dynamic_cast<const LinearOptimizer*>(oper)->GetEnergy(x);
   norm_goal = std::max(rel_tol*norm, abs_tol);
   cout << "norm goal is " << norm_goal << '\n';
   cout << "initial objective value is " << norm0 <<'\n';

   // initialize the jacobian
   oper->Mult(x,jac);
   
   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++)
   {
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0)
      {
         mfem::out << "BFGS optimization iteration " << setw(2) << it
                   << " : J = " << norm;
         if (it > 0)
         {
            mfem::out << ", J/J_0 = " << norm/norm0;
            mfem::out << " . jac norm is " << jac.Norml2();
         }
         mfem::out<<'\n';
      }

      
      if (norm <= norm_goal)
      {
         converged = 1;
         break;
      }
      
      if (it >= max_iter)
      {
         converged = 0;
         break;
      }

      // compute the direction c = B * (-deriv) 
      B.Mult(jac, c);
      c.Neg();
      // compute step size
      double c_scale = ComputeStepSize(x,c,norm);
      cout << "step size is " <<  c_scale << ". ";
      if (c_scale == 0.0)
      {
         converged = 0;
         break;
      }
      c *= c_scale;
      // update the state
      x += c;

      // update objective new value and derivative
      norm = dynamic_cast<const LinearOptimizer*>(oper)->GetEnergy(x);
      cout << "new objective value is " << norm << '\n';

      // update hessian
      oper->Mult(x,jac_new);
      UpdateHessianInverse(c,jac,jac_new,ident,B);
      // update jac
      jac = jac_new;
   }
   opt = x;
   final_iter = it;
   final_norm = norm;
}

void BFGSNewtonSolver::UpdateHessianInverse(const Vector &s, const Vector &jac,
                                      const Vector &jac_new,const DenseMatrix &I,
                                      DenseMatrix &H)
{
   Vector y(jac_new);
   y -= jac;

   double rho = 1./(y * s);
   
   DenseMatrix s_mat(numvar,1);
   s_mat.SetCol(0,s);

   DenseMatrix y_mat(numvar,1);
   y_mat.SetCol(0,y);

   DenseMatrix sy_mat(numvar);
   DenseMatrix ys_mat(numvar);
   DenseMatrix ss_mat(numvar);

   ::MultABt(s_mat,y_mat,sy_mat);
   ::MultABt(y_mat,s_mat,ys_mat);
   ::MultABt(s_mat,s_mat,ss_mat);

   sy_mat *= (-rho);
   ys_mat *= (-rho);
   ss_mat *= rho;

   sy_mat += I;
   ys_mat += I;

   DenseMatrix syh(numvar);

   ::Mult(sy_mat,H,syh);
   ::Mult(syh,ys_mat,H);

   H += ss_mat;
}

double BFGSNewtonSolver::ComputeStepSize(const Vector &x, const Vector &c,
                                         const double norm0)
{
   Vector jac_aux(x.Size());
   Vector x_new(x.Size());
   double phi_init = norm0;
   double dphi_init = jac*c; // should be negative
   MFEM_ASSERT(dphi_init < 0.0,
      "BFGS Newton::ComputeStepSize(): wrong searching direction.\n");

   double phi_old = phi_init;
   double dphi_old = dphi_init;
   double alpha_old = 0.0;

   double alpha_new = alpha_init;
   double phi_new;
   double dphi_new;
   double quad_coeff;
   for (int iter = 0; true; iter++)
   {
      // evalueate the new function value
      add(x,alpha_new,c,x_new);
      phi_new = dynamic_cast<const LinearOptimizer*>(oper)->GetEnergy(x_new);
      // check if the step violates the sdc,
      // or when i > 0, new phi is greater than the old, then zoom
      if ( (phi_new > phi_init+c1*alpha_new*dphi_init) || 
           ((iter > 0) && (phi_new >= phi_old)) )
      {
         return Zoom(alpha_old,alpha_new,phi_old,phi_init,dphi_init,x,c);
      }


      // get new gradient
      oper->Mult(x_new,jac_aux);
      dphi_new = c * jac_aux;

      // if curvature condition is satisfied
      if (fabs(dphi_new) <= -c2*dphi_init)
      {
         if (c2 > 1e-6)
         {
            norm = phi_new; 
            return alpha_new; 
         }
         // c2 < 1e-6, this is not quite often
      }


      // curvature condition is not satisfied,
      // and phi_new < phi_old
      if (dphi_new >= 0)
      {
         return Zoom(alpha_new,alpha_old,phi_new,phi_init,dphi_init,x,c);
      }

      // update variables
      quad_coeff = alpha_new - alpha_old;
      quad_coeff = ( (phi_new - phi_old) - dphi_new*quad_coeff) / (quad_coeff*quad_coeff);
      alpha_old = alpha_new;
      phi_old = phi_new;
      dphi_old = dphi_new;

      // update step size
      if (quad_coeff > 0.0)
      {
         alpha_new = alpha_old - 0.5 * dphi_old / quad_coeff;
         if (alpha_new < alpha_old || alpha_new > alpha_max )
         {
            alpha_new = std::min(2.0*alpha_old,alpha_max);
         }
      }
      else
      {
         alpha_new = std::min(2.0*alpha_old,alpha_max);
      }
   } // end of iteration
   throw MachException("ComputeStepSize(): fail to meet the strong wolfe condition.\n");
}

double BFGSNewtonSolver::Zoom(double alpha_low, double alpha_hi, double phi_low,
                              double phi_0, double dphi_0, const Vector &x,
                              const Vector &c)
{
   double alpha_new;
   double phi_new;
   double dphi_new;
   Vector jac_aux(x.Size());
   Vector x_new(x.Size());
   for (int j = 0; j < zoom_max_iter; j++)
   {
      alpha_new = (alpha_low + alpha_hi) / 2.0;
      add(x,alpha_new,c,x_new);
      phi_new = dynamic_cast<const LinearOptimizer*>(oper)->GetEnergy(x_new);

      // the SDC condition is not met
      if ( phi_new > phi_0 + c1 * alpha_new * dphi_0 || phi_new > phi_low )
      {
         alpha_hi = alpha_new;
      }
      // the SDC condition is met
      else
      {
         oper->Mult(x_new,jac_aux);
         dphi_new = c * jac_aux;
         // check curvature condition
         if (fabs(dphi_new) < -c2*dphi_0) 
         {
            norm = phi_new;
            return alpha_new;
         }
         if (dphi_new * (alpha_hi - alpha_low) >= 0.0)
         {
            alpha_hi = alpha_low;
         }
         alpha_low = alpha_new;
         phi_low = alpha_new;
      }

   }
   throw MachException("Zoom(): fail to find a step size.\n");
}

// double BFGSNewtonSolver::Zoom(double alpha_low, double alpha_hi,
//                               double phi_low, double phi_hi,
//                               double dphi_low,double dphi_hi,
//                               bool deriv_hi, const Vector &x,
//                               const Vector &c)
// {
//    Vector jac_aux(x.Size());
//    double phi_new, dphi_new, alpha_new;
//    for (int j = 0; j < max_iter; j++)
//    {
//       alpha_new = InterpStep(alpha_low,alpha_hi,phi_low,phi_hi,
//                              dphi_low,dphi_hi,deriv_hi);
//       phi_new = oper->GetEnergy(x+alpha_new*c);
      
//       // if the new location violates the SDC
//       if ( (phi_new > phi_init+c1*alpha_new*dphi_init) || 
//            (phi_new > phi_low) )
//       {
//          alpha_hi = alpha_new;
//          phi_hi = phi_new;
//          dphi_hi = 0.0;
//          deriv_hi = false;
//       }
//       // SDC is satisfied
//       else
//       {
//          oper->Mult(x+alpha_new*c,jac_aux);
//          dphi_new = c * jac_aux;

//          // curvature condition is satisfied
//          if (fabs(dphi_new) <= -c2*dphi_init)
//          {
//             return alpha_new;
//          }
//          // minimum locates within [alpha_low, alpha_hi]
//          else if( dphi_new * (alpha_hi - alpha_low) >= 0.)
//          {
//             alpha_hi = alpha_low;
//             phi_hi = phi_low;
//             dphi_hi = dphi_low;
//             deriv_hi = true;
//          }

//          // new low position is alpha_new
//          alpha_low = alpha_new;
//          phi_low = phi_new;
//          dphi_low = dphi_new;
//       }
//    } // iteration j

//    if (phi_new < phi_init+c1*alpha_new*dphi_init)
//    {
//       cout << "WARNING in Zoom(): "
//            << "step found, but curvature condition not met" << '\n';
//       return alpha_new;
//    }
//    throw MachException("Zoom(): max iteration reached.");
// }

// double BFGSNewtonSolver::InterpStep(const double &alpha_low, const double &alpha_hi,
//                                     const double &f_low, const double &f_hi,
//                                     const double &df_low, const double &df_hi,
//                                     const bool &deriv_hi)
// {
//    // 0.5 * (alpha_low + alpha_hi)
//    return QuadraticStep();
//    if (!deriv_hi) { return QuadraticStep(alpha_low,alpha_hi,f_low,f_hi,df_low); }

//    // derivative availabe at alphi_hi, try cubic interpolation
//    double dalpha = alpha_hi - alpha_low;
//    double a = 6.0 * (f_low - f_hi) + 3.0 * (df_low + df_hi) *dalpha;
//    double b = 6.0 * (f_hi - f_low) - 2.0 * (2.0 * df_low +df_hi) * dalpha;
//    double c = df_low * dalpha;

//    // check the discriminant; if negative, resort to quadratic fit
//    double det = pow(b,2.0) - 4.0*a*c;
//    if (det < 0.0) { throw MachException("InterpStep: det is negative?\n"); }

//    // if the 3rd order is small, then do quadratic
//    if (fabs(a) < 1e-10) { return QuadraticStep(alpha_low,alpha_hi,f_low,f_hi,df_low); }

//    //calculate the two extre num
//    double x1 = (-b + sqrt(det))/(2.0*a);
//    double x2 = (-b - sqrt(det))/(2.0*a);
//    x1 = alpha_low + x1 * (alpha_hi - alpha_low);
//    x2 = alpha_low + x2 * (alpha_hi - alpha_low);

//    // get steps' range
//    double min_alpha = std::min(alpha_hi, alpha_low);
//    double max_alpha = std::max(alpha_hi, alpha_low);
//    double step;


//    bool x1_inrange = (x1 >= min_alpha) && (x1 <= max_alpha);
//    bool x2_inrange = (x2 >= min_alpha) && (x2 <= max_alpha);

//    // alpha_min <= x1 <= alpha_max
//    if ( x1_inrange )
//    {  
//       // also alpha_min <= x2 <= alpha_max
//       if ( x2_inrange )
//       {
//          throw MachException("InterpStep: found x2 in range?\n");
//       }
//       // only x1 in range
//       else
//       {
//          step = x1;
//       }
//    }
//    // only alpha_min <= x2 <= alpha_max, but x1 not
//    else if ( x2_inrange )
//    {
//       throw MachException("InterpStep: found x2 in range?\n");
//    }
//    // nor x1 or x1 in range
//    else
//    {
//       throw MachException("InterStep: x1 is not in range?\n");
//    }
//    return step;
// }


// double BFGSNewtonSolver::QuadraticStep(const double &alpha_low, const double &alpha_hi
//                                        const double &f_low, const double &f_hi,
//                                        const double &df_low)
// {
//    double dalpha = alpha_hi - alpha_low;
//    double step = alpha_low - 0.5*df_low*dalpha*dalpha / (f_hi - f_low - df_low*dalpha);

//    double min_alpha = std::min(alpha_hi,alpha_low);
//    double max_alpha = std::max(alpha_hi,alpha_low);

//    if ((step < min_alpha) || (step > max_alpha))
//    {
//       throw MachException("something wrong in zoom.");
//    }

//    if ( (step - min_alpha) < 1e-2*(max_alpha - minalpha) )
//    {
//       step = 0.5 * (alpha_low + alpha_hi);
//    }
//    return step;
// }



} // end of namespace mfem
