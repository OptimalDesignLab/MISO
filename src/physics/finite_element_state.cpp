#include "finite_element_state.hpp"

namespace miso
{
double calcLpNorm(const FiniteElementState &state, const double p)
{
   if (state.gridFunc().VectorDim() == 1)
   {
      mfem::ConstantCoefficient zero(0.0);
      // return state.gridFunc().ComputeLpError(p, zero);
      return calcLpError(state, zero, p);
   }
   else
   {
      mfem::Vector zero(state.gridFunc().VectorDim());
      zero = 0.0;
      mfem::VectorConstantCoefficient zerovec(zero);
      // return state.gridFunc().ComputeLpError(p, zerovec);
      return calcLpError(state, zerovec, p);
   }
}

double calcL2Error(const FiniteElementState &state, mfem::Coefficient &ex_sol)
{
   double loc_error = 0.0;

   const char *name = state.coll().Name();
   const auto &space = state.space();
   const auto &gf = state.gridFunc();
   mfem::Array<int> vdofs;
   mfem::Vector shape;

   // sum up the L2 error over all states
   for (int i = 0; i < space.GetNE(); i++)
   {
      const auto *fe = space.GetFE(i);
      const auto *ir = [&]()
      {
         if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
         {
            return &(fe->GetNodes());
         }
         else
         {
            int intorder = 2 * fe->GetOrder() + 3;
            return &(mfem::IntRules.Get(fe->GetGeomType(), intorder));
         }
      }();
      int fdof = fe->GetDof();
      auto *T = space.GetElementTransformation(i);
      shape.SetSize(fdof);
      space.GetElementVDofs(i, vdofs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const auto &ip = ir->IntPoint(j);
         fe->CalcShape(ip, shape);
         double a = 0;
         for (int k = 0; k < fdof; k++)
         {
            if (vdofs[k] >= 0)
            {
               a += gf(vdofs[k]) * shape(k);
            }
            else
            {
               a -= gf(-1 - vdofs[k]) * shape(k);
            }
         }
         T->SetIntPoint(&ip);
         a -= ex_sol.Eval(*T, ip);
         loc_error += ip.weight * T->Weight() * a * a;
      }
   }

   // negative quadrature weights may cause the error to be negative
   if (loc_error < 0.0)
   {
      loc_error = -sqrt(-loc_error);
   }
   else
   {
      loc_error = sqrt(loc_error);
   }
   return mfem::GlobalLpNorm(2, loc_error, state.comm());
}

double calcL2Error(const FiniteElementState &state,
                   mfem::VectorCoefficient &ex_sol,
                   const int entry)
{
   double loc_error = 0.0;

   const char *name = state.coll().Name();
   const auto &space = state.space();
   const auto &gf = state.gridFunc();
   mfem::DenseMatrix vals;
   mfem::DenseMatrix exact_vals;
   mfem::Vector loc_errs;

   if (entry < 0)
   {
      // sum up the L2 error over all states
      for (int i = 0; i < space.GetNE(); i++)
      {
         const auto *fe = space.GetFE(i);
         const auto *ir = [&]()
         {
            if ((strncmp(name, "SBP", 3) == 0) ||
                (strncmp(name, "DSBP", 4) == 0))
            {
               return &(fe->GetNodes());
            }
            else
            {
               int intorder = 2 * fe->GetOrder() + 3;
               return &(mfem::IntRules.Get(fe->GetGeomType(), intorder));
            }
         }();
         auto *T = space.GetElementTransformation(i);
         gf.GetVectorValues(*T, *ir, vals);
         ex_sol.Eval(exact_vals, *T, *ir);
         vals -= exact_vals;
         loc_errs.SetSize(vals.Width());
         vals.Norm2(loc_errs);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const auto &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            loc_error += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
         }
      }
   }
   else
   {
      // calculate the L2 error for component index `entry`
      for (int i = 0; i < space.GetNE(); i++)
      {
         const auto *fe = space.GetFE(i);
         const auto *ir = [&]()
         {
            if ((strncmp(name, "SBP", 3) == 0) ||
                (strncmp(name, "DSBP", 4) == 0))
            {
               return &(fe->GetNodes());
            }
            else
            {
               int intorder = 2 * fe->GetOrder() + 3;
               return &(mfem::IntRules.Get(fe->GetGeomType(), intorder));
            }
         }();
         auto *T = space.GetElementTransformation(i);
         gf.GetVectorValues(*T, *ir, vals);
         ex_sol.Eval(exact_vals, *T, *ir);
         vals -= exact_vals;
         loc_errs.SetSize(vals.Width());
         vals.GetRow(entry, loc_errs);
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            const auto &ip = ir->IntPoint(j);
            T->SetIntPoint(&ip);
            loc_error += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
         }
      }
   }
   // negative quadrature weights may cause the error to be negative
   if (loc_error < 0.0)
   {
      loc_error = -sqrt(-loc_error);
   }
   else
   {
      loc_error = sqrt(loc_error);
   }
   return mfem::GlobalLpNorm(2, loc_error, state.comm());
}

double calcLpError(const FiniteElementState &state,
                   mfem::Coefficient &exsol,
                   const double p)
{
   double loc_error = 0.0;

   const char *name = state.coll().Name();
   const auto &space = state.space();
   const auto &gf = state.gridFunc();
   mfem::Vector vals;

   for (int i = 0; i < space.GetNE(); i++)
   {
      const auto *fe = space.GetFE(i);
      const auto *ir = [&]()
      {
         if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
         {
            return &(fe->GetNodes());
         }
         else
         {
            int intorder = 2 * fe->GetOrder() + 3;
            return &(mfem::IntRules.Get(fe->GetGeomType(), intorder));
         }
      }();
      gf.GetValues(i, *ir, vals);
      auto *T = space.GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const auto &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double ip_error = fabs(vals(j) - exsol.Eval(*T, ip));
         if (p < mfem::infinity())
         {
            ip_error = pow(ip_error, p);
            loc_error += ip.weight * T->Weight() * ip_error;
         }
         else
         {
            loc_error = std::max(loc_error, ip_error);
         }
      }
   }

   if (p < mfem::infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (loc_error < 0.0)
      {
         loc_error = -pow(-loc_error, 1.0 / p);
      }
      else
      {
         loc_error = pow(loc_error, 1.0 / p);
      }
   }

   return mfem::GlobalLpNorm(p, loc_error, state.comm());
}

double calcLpError(const FiniteElementState &state,
                   mfem::VectorCoefficient &ex_sol,
                   const double p,
                   const int entry)
{
   double loc_error = 0.0;

   const char *name = state.coll().Name();
   const auto &space = state.space();
   const auto &gf = state.gridFunc();

   mfem::DenseMatrix vals;
   mfem::DenseMatrix exact_vals;
   mfem::Vector loc_errs;

   for (int i = 0; i < space.GetNE(); i++)
   {
      const auto *fe = space.GetFE(i);
      const auto *ir = [&]()
      {
         if ((strncmp(name, "SBP", 3) == 0) || (strncmp(name, "DSBP", 4) == 0))
         {
            return &(fe->GetNodes());
         }
         else
         {
            int intorder = 2 * fe->GetOrder() + 3;
            return &(mfem::IntRules.Get(fe->GetGeomType(), intorder));
         }
      }();
      auto *T = space.GetElementTransformation(i);
      gf.GetVectorValues(*T, *ir, vals);
      ex_sol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      if (entry < 0)
      {
         // compute the lengths of the errors at the integration points
         // thus the vector norm is rotationally invariant
         vals.Norm2(loc_errs);
      }
      else
      {
         vals.GetRow(entry, loc_errs);
      }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const auto &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double ip_error = loc_errs(j);
         if (p < mfem::infinity())
         {
            ip_error = pow(ip_error, p);
            loc_error += ip.weight * T->Weight() * ip_error;
         }
         else
         {
            loc_error = std::max(loc_error, ip_error);
         }
      }
   }

   if (p < mfem::infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (loc_error < 0.0)
      {
         loc_error = -pow(-loc_error, 1.0 / p);
      }
      else
      {
         loc_error = pow(loc_error, 1.0 / p);
      }
   }

   return mfem::GlobalLpNorm(p, loc_error, state.comm());
}

}  // namespace miso
