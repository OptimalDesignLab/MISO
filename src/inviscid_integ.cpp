#include "inviscid_integ.hpp"
#include "sbp_fe.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

void InviscidIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &Trans,
    const Vector &elfun, Vector &elvect)
{
   // This should be in a try/catch, but that creates other issues
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, fluxi, dxidx;
   DenseMatrix adjJ_i, elflux, elres;
#endif
	elvect.SetSize(num_states*num_nodes);
   ui.SetSize(num_states);
   adjJ_i.SetSize(dim);
   dxidx.SetSize(dim);
   elflux.SetSize(num_states, num_nodes);
   elres.SetSize(num_states, num_nodes);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);
   
   elres = 0.0;
   for (int di = 0; di < dim; ++di)
   {
      // get the flux at all the nodes      
      for (int i = 0; i < num_nodes; ++i)
      {
         Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
         CalcAdjugate(Trans.Jacobian(), adjJ_i);
         adjJ_i.GetRow(di, dxidx);
         u.GetRow(i, ui);
         elflux.GetColumnReference(i, fluxi);
         flux(dxidx.GetData(), ui.GetData(), fluxi.GetData());
      }
      sbp.multWeakOperator(di, elflux, elres, true);
   }
   // This is necessary because data in elvect is expected to be ordered `byNODES`
   res.Transpose(elres);
   res *= alpha;
}

void DyadicFluxIntegrator::AssembleElementVector(
    const FiniteElement &el,
    ElementTransformation &Trans,
    const Vector &elfun, Vector &elvect)
{
   // This should be in a try/catch, but that creates other issues
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui, uj, fluxij;
   DenseMatrix adjJ_i, adjJ_j;
#endif
	elvect.SetSize(num_states*num_nodes);
   fluxij.SetSize(num_states);
   adjJ_i.SetSize(dim);
   adjJ_j.SetSize(dim);
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

	elvect = 0.0;
   for (int i = 0; i < num_nodes; ++i)
   {
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      CalcAdjugate(Trans.Jacobian(), adjJ_i);
      u.GetRow(i,ui);
		for (int j = i+1; j < num_nodes; ++j)
		{
         Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
         CalcAdjugate(Trans.Jacobian(), adjJ_j);
         u.GetRow(j, uj);
			for (int di = 0; di < dim; ++di)
			{
            // TODO: we should add state_offset to ui and uj, and eqn_offset to
            // fluxij because the flux function may not know about other states 
            // and equations.
				flux(di, ui.GetData(), uj.GetData(), fluxij.GetData());
            double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
            Sij *= alpha;
            for (int n = 0; n < num_states; ++n)
            {
               res(i,n) += Sij*fluxij(n);
               res(j,n) -= Sij*fluxij(n);
            }
			} // di loop
      } // j node loop
   } // i node loop
}

// void DyadicFluxIntegrator::AssembleElementGrad(
//    const FiniteElement &el,
//    ElementTransformation &Trans,
//    const Vector &elfun, DenseMatrix &elmat)
// {
//    // This should be in a try/catch, but that creates other issues
//    SBPFiniteElement &sbp = dynamic_cast<SBPFiniteElement&>(el);


// 	elmat = 0.0;
//    for (int i = 0; i < num_nodes; ++i)
//    {
//       Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
//       CalcAdjugate(Trans.Jacobian(), adjJ_i);
//       u.GetColumnReference(i, ui);
//       res.GetColumnReference(i, resi);

// 		for (int j = i+1; j < num_nodes; ++j)
// 		{
//          Trans.SetIntPoint(&el.GetNodes().IntPoint(j));
//          CalcAdjugate(Trans.Jacobian(), adjJ_j);
//          u.GetColumnReference(j, uj);
//          res.GetColumnReference(j, resj);

// 			for (int di = 0; di < dim; ++di)
// 			{
//             stack.new_recording();
//             adept::set_values(ui_ad, num_states, ui.GetData());
//             adept::set_values(uj_ad, num_states, uj.GetData());
// 				flux_fun<adouble>(di, ui_ad, uj_ad, flux_ad);
//             double Sij = sbp.getSkewEntry(di, i, j, adjJ_i, adjJ_j);
//             // The following is for if we start using Adept arrays
//             // double Sij = sbp.getSkewEntry(di, i, j, dxidx(i,__,__),
//             //                              dxidx(j,__,__));
//             Sij*flux_jac
//             resi.Add(Sij, flux);
//             resj.Add(-Sij, flux);
// 			} // di loop
//       } // j node loop
//    } // i node loop

// }

void LPSIntegrator::AssembleElementVector(const mfem::FiniteElement &el,
                                          mfem::ElementTransformation &Trans,
                                          const mfem::Vector &elfun,
                                          mfem::Vector &elvect)
{
   const SBPFiniteElement &sbp = dynamic_cast<const SBPFiniteElement&>(el);
   int num_nodes = sbp.GetDof();
   int dim = sbp.GetDim();
#ifdef MFEM_THREAD_SAFE
   Vector ui;
   DenseMatrix adjJt, w, Pw;
#endif
	elvect.SetSize(num_states*num_nodes);
   ui.SetSize(num_states);
   adjJt.SetSize(dim);
   w.SetSize(num_states, num_nodes);
   Pw.SetSize(num_states, num_nodes);
   Vector wi, Pwi;
   DenseMatrix u(elfun.GetData(), num_nodes, num_states);
   DenseMatrix res(elvect.GetData(), num_nodes, num_states);

   // Step 1: convert from working variables (this may be the identity)
   for (int i = 0; i < num_nodes; ++i)
   {
      u.GetRow(i,ui);
      w.GetColumnReference(i, wi);
      convertVars(ui.GetData(), wi.GetData());
   }
   // Step 2: apply the projection operator to w
   sbp.multProjOperator(w, Pw, false);
   // Step 3: apply scaling matrix at each node and diagonal norm
   for (int i = 0; i < num_nodes; ++i)
   {
      Trans.SetIntPoint(&el.GetNodes().IntPoint(i));
      CalcAdjugateTranspose(Trans.Jacobian(), adjJt);
      u.GetRow(i,ui);
      Pw.GetColumnReference(i, Pwi);
      w.GetColumnReference(i, wi);
      applyScaling(adjJt.GetData(), ui.GetData(), Pwi.GetData(), wi.GetData());
      wi *= lps_coeff;
   }
   sbp.multNormMatrix(w, w);
   // Step 4: apply the transposed projection operator to H*A*P*w
   sbp.multProjOperator(w, Pw, true);
   // This is necessary because data in elvect is expected to be ordered `byNODES`
   res.Transpose(Pw);
   res *= alpha;
}

} // namespace mach