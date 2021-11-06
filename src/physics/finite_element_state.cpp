#include "finite_element_state.hpp"

namespace mach
{
double norm(const FiniteElementState &state, const double p)
{
   if (state.space().GetVDim() == 1)
   {
      mfem::ConstantCoefficient zero(0.0);
      return state.gridFunc().ComputeLpError(p, zero);
   }
   else
   {
      mfem::Vector zero(state.space().GetVDim());
      zero = 0.0;
      mfem::VectorConstantCoefficient zerovec(zero);
      return state.gridFunc().ComputeLpError(p, zerovec);
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

   // from MFEM: negative quadrature weights may cause the error to be negative
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
                   int entry)
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
   // from MFEM: negative quadrature weights may cause the error to be negative
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

double error(const FiniteElementState &state,
             mfem::Coefficient &ex_sol,
             double p)
{
   return calcL2Error(state, ex_sol);
}
double error(const FiniteElementState &state,
             mfem::VectorCoefficient &ex_sol,
             int entry,
             double p)
{
   return calcL2Error(state, ex_sol, entry);
}

}  // namespace mach
