#include <iostream>

#include "mfem_extensions.hpp"
#include "evolver.hpp"
#include "utils.hpp"

using namespace mfem;
using namespace std;
using namespace mach;

namespace mach
{

void PseudoTransientSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
}

void PseudoTransientSolver::Step(Vector &x, double &t, double &dt)
{
   f->SetTime(t + dt);
   k.SetSize(x.Size(), mem_type); 
   f->ImplicitSolve(dt, x, k);
   x.Add(dt, k);
   t += dt;
}

void RRKImplicitMidpointSolver::Init(TimeDependentOperator &_f)
{
   ODESolver::Init(_f);
}

void RRKImplicitMidpointSolver::Step(Vector &x, double &t, double &dt)
{
   EntropyConstrainedOperator *f_ode =
       dynamic_cast<EntropyConstrainedOperator *>(f);
   f_ode->SetTime(t + dt/2);
   k.SetSize(x.Size(), mem_type);
   f_ode->ImplicitSolve(dt, dt/2, x, k);

   // Set-up and solve the scalar nonlinear problem for the relaxation gamma
   //cout << "x size is " << x.Size() << '\n';
   //cout << "x is empty? == " << x.GetMemory().Empty() << '\n';
   double delta_entropy = f_ode->EntropyChange(dt/2, x, k);
   //double delta_entropy = f_ode->EntropyChange(dt, x, k);
   *out << "delta_entropy is " << delta_entropy << '\n';
   double entropy_old = f_ode->Entropy(x);
   *out << "old entropy is " << entropy_old << '\n';
   mfem::Vector x_new(x.Size());
   //cout << "x_new size is " << x_new.Size() << '\n';
   auto entropyFun = [&](double gamma)
   {
      *out << "In lambda function: " << std::setprecision(14); 
      add(x, gamma*dt, k, x_new);
      double entropy = f_ode->Entropy(x_new);
      *out << "gamma = " << gamma << ": ";
      *out << "residual = " << entropy - entropy_old + gamma*dt*delta_entropy << endl;
      //cout << "new entropy is " << entropy << '\n';
      return entropy - entropy_old + gamma*dt*delta_entropy;
   };
   // TODO: tolerances and maxiter should be provided in some other way
   const double ftol = 1e-12;
   const double xtol = 1e-12;
   const int maxiter = 30;
   //double gamma = bisection(entropyFun, 0.50, 1.5, ftol, xtol, maxiter);
   double gamma = secant(entropyFun, 0.99, 1.01, ftol, xtol, maxiter);
   *out << "\tgamma = " << gamma << endl;
   x.Add(gamma*dt, k);
   t += gamma*dt;
}

// namespace
// {

// /// find the roots of a cubic defined by a + bx + cx^2 + dx^3 
// void cubicFormula(double a, double b, double c, double d, Vector roots)
// {

//    double p = -c / (3*d);
//    double q = std::pow(p, 3) + (c*b-3*a*d)/(6*std::pow(d,2));
//    double r = b / (3*d);

//    double disc = q*q + std::pow((r-p*p),3);
//    if (disc > 0)
//    {
//       double x = std::cbrt(q + std::sqrt(disc))
//                + std::cbrt(q + std::sqrt(disc))
//                + p;
      
//    }
//    else if (disc == 0)
//    {

//    }
//    else if (disc < 0)
//    {

//    }
// }

/**
% goldenSec
%
% Purpose: finds a local mimimum using the golden section search
%
% Pre: assumes the function is C1 continuous
%
% Input:
%   a0, b0 = the intial interval of uncertainty
%   delta = the desired tolerance on the interval of uncertainty
%   func = a handle to the function to be optimized
%   dfdx = a handle to the function derivative (not needed in general)
%   start, offset = the intervals of uncertainty are drawn as lines,
%      starting at the vertical height start, and each successive line
%      is offset from the last
%
function [a,b,fa,fb,dF,n] = goldenSec(a0,b0,delta,func,dfdx,start,offset,...
    plotInt)
​
% assumes that a0 < b0; so check and flip if necessary
if (a0 > b0) 
    temp = a0;
    a0 = b0;
    b0 = temp;
end
    
% plot the intial interval of uncertainity
if (plotInt ~= 0)
    line([a0 b0],[start start],'color','k','linewidth',1);
    line([a0 a0],[(start-offset/3) (start+offset/3)],'color','k','linewidth',1);
    line([b0 b0],[(start-offset/3) (start+offset/3)],'color','k','linewidth',1);
end
​
% set tau, the golden ratio
tau = 2.0/(1.0 + sqrt(5.0));
​
% initialize the two intervals
Intv1 = [a0, (b0-a0)*tau + a0]';
Intv2 = [b0 - (b0-a0)*tau, b0]';
​
% evaluate the function at the interval end points
F1 = zeros(2,1);
F2 = zeros(2,1);
F1(1) = func(Intv1(1));
F1(2) = func(Intv1(2));
F2(1) = func(Intv2(1));
F2(2) = func(Intv2(2));
​
% get the interval of uncertainty tolerance, and set the number of
% iterations
tol = delta/(b0 - a0);
n = 0; 
plottol = (1/log(tau))*log(tol)/2;
dfdx0 = max(dfdx(a0),dfdx(b0));
dF(1) = abs((F2(2) - F1(1))/(Intv2(2) - Intv1(1)))/abs(dfdx0);
%while ( dF(n+1) > tol )
while ((Intv2(2) - Intv1(1)) > tol)
    n = n+1;
    if (F1(2) < F2(1))
        % iterval 2 contains the lowest interior evaluation point
        Intv1(1) = Intv2(1);
        F1(1) = F2(1);
        Intv2(1) = Intv1(2);
        F2(1) = F1(2);
        % get new interval 1 end point, and function eval
        Intv1(2) = Intv1(1) + (Intv2(2) - Intv1(1))*tau;
        F1(2) = func(Intv1(2));
    else
        % iterval 1 contains the lowest interior evaluation point
        Intv2(2) = Intv1(2);
        F2(2) = F1(2);
        Intv1(2) = Intv2(1);
        F1(2) = F2(1);
        % get new interval 2 start point, and function eval
        Intv2(1) = Intv2(2) - (Intv2(2) - Intv1(1))*tau;
        F2(1) = func(Intv2(1));
    end % if
    
    % plot the intial interval of uncertainity
    if ( (n < plottol) && (plotInt ~= 0) )
        start = start-offset;
        line([Intv1(1) Intv2(2)],[start start],'color','k','linewidth',1);
        line([Intv1(1) Intv1(1)],[(start-offset/3) (start+offset/3)],'color',...
            'k','linewidth',1);
        line([Intv2(2) Intv2(2)],[(start-offset/3) (start+offset/3)],'color',...
            'k','linewidth',1);
    end % if
    
    dF(n+1) = abs((F2(2) - F1(1))/(Intv2(2) - Intv1(1)))/abs(dfdx0);
    
end % while
     
a = Intv1(1);
b = Intv2(2);
fa = F1(1);
fb = F2(2);
        
return;
*/

// }

void RelaxedNewton::SetOperator(const Operator &op)
{
   NewtonSolver::SetOperator(op);

   rkp1.SetSize(width);
   xkp1.SetSize(width);
}

void RelaxedNewton::Mult(const Vector &b, Vector &x) const
{
   MFEM_ASSERT(energy != NULL, "the energy operator is not set "
                               "(use SetEnergyOperator).");
   
   NewtonSolver::Mult(b, x);
}

double RelaxedNewton::ComputeScalingFactor(const Vector &x,
                                           const Vector &b) const
{
   double alpha = 1.0;
   if (first_iter)
   {
      alpha = 0.1;
      first_iter = false;
   }
   double beta = 0.5;

   add(x, -alpha, c, xkp1);

   // double g0 = energy->GetEnergy(x) + *load*x;
   // double galpha = energy->GetEnergy(xkp1) + *load*xkp1;

   double g0 = Norm(r);
   oper->Mult(xkp1, rkp1);
   double galpha = Norm(rkp1);

   std::cout << "g0: " << g0 << "\n";
   std::cout << "g1: " << galpha << "\n";
   while (galpha > g0)
   {
      alpha *= beta;
      add(x, -alpha, c, xkp1);

      // galpha = energy->GetEnergy(xkp1) + *load*xkp1;
      oper->Mult(xkp1, rkp1);
      galpha = Norm(rkp1);
      std::cout << "alpha = " << alpha << "\n";
      std::cout << "galpha: " << galpha << "\n";
   }

   return alpha;
}

void RelaxedNewton::SetEnergyOperator(const NonlinearForm &op)
{
   energy = &op;
}

} // namespace mach