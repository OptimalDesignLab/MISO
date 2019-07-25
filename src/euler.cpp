#include "euler.hpp"
#include "euler_fluxes.hpp"
#include "sbp_fe.hpp"
#include "diag_mass_integ.hpp"
#include "inviscid_integ.hpp"
#include "evolver.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

EulerSolver::EulerSolver(const string &opt_file_name,
                         unique_ptr<mfem::Mesh> smesh, int dim)
   : AbstractSolver(opt_file_name, move(smesh))
{
   // set the finite-element space and create (but do not initialize) the
   // state GridFunction
   num_state = dim + 2;
   fes.reset(new SpaceType(mesh.get(), fec.get(), num_state, Ordering::byVDIM));
   u.reset(new GridFunType(fes.get()));
#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
        << fes->GlobalTrueVSize() << endl;
#else 
   cout << "Number of finite element unknowns: "
        << fes->GetTrueVSize() << endl;
#endif

   // set up the mass matrix
   mass.reset(new BilinearFormType(fes.get()));
   mass->AddDomainIntegrator(new DiagMassIntegrator(num_state));
   mass->Assemble();
   mass->Finalize();

   // set up the spatial semi-linear form
   // TODO: should decide between one-point and two-point fluxes using options
   double alpha = 1.0;
   res.reset(new NonlinearFormType(fes.get()));
   
   res->AddDomainIntegrator(new DyadicFluxIntegrator(diff_stack,
                                                     calcIsmailRoeFlux<double,2>,
                                                     num_state, alpha));

   //res->AddDomainIntegrator(new InviscidIntegrator(diff_stack,
   //                                                calcEulerFlux<double,2>,
   //                                                num_state, alpha));

   // add the LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].get<double>();
   res->AddDomainIntegrator(new LPSIntegrator(diff_stack,
                                              calcEntropyVars<double,2>,
                                              applyLPSScaling<double,2>,
                                              num_state, alpha, lps_coeff));

   // define the time-dependent operator
#ifdef MFEM_USE_MPI
   // The parallel bilinear forms return a pointer that this solver owns
   mass_matrix.reset(mass->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#endif
   evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));

}

double EulerSolver::calcStepSize(double cfl) const
{
   double (*calcSpect)(const double *dir, const double *q);
   if (num_dim == 1)
   {
      calcSpect = calcSpectralRadius<double,1>;
   }
   else if (num_dim == 2)
   {
      calcSpect = calcSpectralRadius<double,2>;
   }
   else
   {
      calcSpect = calcSpectralRadius<double,3>;
   }

   double dt_local = 1e100;
   Vector xi(num_dim);
   Vector dxij(num_dim);
   Vector ui, dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(num_dim);
   for (int k = 0; k < fes->GetNE(); k++)
   {
      // get the element, its transformation, and the state values on element
      const FiniteElement *fe = fes->GetFE(k);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *trans = fes->GetElementTransformation(k);
      u->GetVectorValues(*trans, *ir, uk);
      for (int i = 0; i < fe->GetDof(); ++i)
      {
         trans->SetIntPoint(&fe->GetNodes().IntPoint(i));
         trans->Transform(fe->GetNodes().IntPoint(i), xi);
         CalcAdjugateTranspose(trans->Jacobian(), adjJt);
         uk.GetColumnReference(i, ui);
         for (int j = 0; j < fe->GetDof(); ++j)
         {
            if (j == i) continue;
            trans->Transform(fe->GetNodes().IntPoint(j), dxij);
            dxij -= xi;
            double dx = dxij.Norml2();
            dt_local = min(dt_local, cfl*dx*dx/calcSpect(dxij, ui)); // extra dx is to normalize dxij
         }
      }
   }
   double dt_min;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&dt_local, &dt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
#else
   dt_min = dt_local;
#endif   
   return dt_min;
}

} // namespace mach