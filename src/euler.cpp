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
template <int dim>
void EntStableLPSIntegrator<dim>::convertVars(const mfem::Vector &q,
                                              mfem::Vector &w)
{
   calcEntropyVars<double,dim>(q.GetData(), w.GetData());
}

template <int dim>
void EntStableLPSIntegrator<dim>::applyScaling(const mfem::DenseMatrix &adjJ,
                                               const mfem::Vector &q,
                                               const mfem::Vector &vec,
                                               mfem::Vector &mat_vec)
{
   applyLPSScaling<double,dim>(adjJ.GetData(), q.GetData(), vec.GetData(),
                               mat_vec.GetData());
}
<<<<<<< HEAD
=======

void IsentropicVortexBC::calcFlux(const mfem::Vector &x,
                                  const mfem::Vector &dir,
                                  const mfem::Vector &q,
                                  mfem::Vector &flux_vec)
{
   calcIsentropicVortexFlux<double>(x.GetData(), dir.GetData(), q.GetData(),
                                    flux_vec.GetData());
}

template <int dim>
void SlipWallBC<dim>::calcFlux(const mfem::Vector &x,
                               const mfem::Vector &dir,
                               const mfem::Vector &q,
                               mfem::Vector &flux_vec)
{
   calcSlipWallFlux<double,dim>(x.GetData(), dir.GetData(), q.GetData(),
                                flux_vec.GetData());
}
>>>>>>> 0533f0400e75c14d84989afefec364393b99405c

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

   res->AddDomainIntegrator(new IsmailRoeIntegrator<2>(diff_stack, alpha));

   //res->AddDomainIntegrator(new EulerIntegrator<2>(diff_stack, alpha));

   // add the LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].get<double>();
   res->AddDomainIntegrator(new EntStableLPSIntegrator<2>(diff_stack, alpha,
                                                          lps_coeff));

   // boundary face integrators are handled in their own function
   addBoundaryIntegrators(alpha, dim);

   // define the time-dependent operator
#ifdef MFEM_USE_MPI
   // The parallel bilinear forms return a pointer that this solver owns
   mass_matrix.reset(mass->ParallelAssemble());
#else
   mass_matrix.reset(new MatrixType(mass->SpMat()));
#endif
   evolver.reset(new NonlinearEvolver(*mass_matrix, *res, -1.0));

}

void EulerSolver::addBoundaryIntegrators(double alpha, int dim)
{
   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size());
   int idx = 0;
   if (bcs.find("vortex") != bcs.end())
   { // isentropic vortex BC
      vector<int> tmp = bcs["vortex"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new IsentropicVortexBC(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      switch (dim)
      {
         case 1:
            res->AddBdrFaceIntegrator(
                new SlipWallBC<1>(diff_stack, fec.get(), alpha),
                bndry_marker[idx]);
            break;
         case 2:
            res->AddBdrFaceIntegrator(
               new SlipWallBC<2>(diff_stack, fec.get(), alpha),
               bndry_marker[idx]);
            break;
         case 3:
            res->AddBdrFaceIntegrator(
               new SlipWallBC<3>(diff_stack, fec.get(), alpha),
               bndry_marker[idx]);
            break;
      }
      idx++;
   }
   for (int k = 0; k < bndry_marker.size(); ++k)
   {
      cout << "boundary_marker[" << k << "]: ";
      for (int i = 0; i < bndry_marker[k].Size(); ++i)
      {
         cout << bndry_marker[k][i] << " ";
      }
      cout << endl;
   }
}

double EulerSolver::calcResidualNorm()
{
   GridFunType r(fes.get());
   res->Mult(*u, r);  // TODO: option to recompute only if necessary
   double res_norm = r*r;
#ifdef MFEM_USE_MPI
   double loc_norm = res_norm;
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#endif
   res_norm = sqrt(res_norm);
   return res_norm;
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



#if 0
<<<<<<< HEAD
template<int dim>
void EulerSolver::calcEulerFluxJacQ(const mfem::Vector &dir,
                                    const mfem::Vector &q,
                                    mfem::DenseMatrix &jac)
{
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   diff_stack.new_recording();
   std::vector<adouble> flux_a(q.Size());
   calcEulerFlux<adouble, dim>(dir_a, q_a, flux_a);
   diff_stack.independent(q_a.data(), q.Size());
   diff_stack.dependent(flux_a.data(), q.Size());
   diff_stack.jacobian(jac.GetData());
}

template<int dim>
void EulerSolver::calcEulerFluxJacDir(const mfem::Vector &dir,
                                    const mfem::Vector &q,
                                    mfem::DenseMatrix &jac)
{
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   diff_stack.new_recording();
   std::vector<adouble> flux_a(q.Size());
   calcEulerFlux<adouble, dim>(dir_a, q_a, flux_a);
   diff_stack.independent(dir_a.data(), dir.Size());
   diff_stack.dependent(flux_a.data(), q.Size());
   diff_stack.jacobian(jac.GetData());
}
=======
>>>>>>> 0533f0400e75c14d84989afefec364393b99405c

template <int dim>
inline void EulerSolver::IsmailRoeFlux(int di, const mfem::Vector &qL,
                                           const mfem::Vector &qR,
                                           mfem::Vector &flux)
{
   calcIsmailRoeFlux<double,dim>(di, qL.GetData(), qR.GetData(),
                                 flux.GetData());
}

template <int dim>
void EulerSolver::calcSlipWallFluxJacQ(const mfem::Vector &x,
                                       const mfem::Vector &dir,
                                       const mfem::Vector &q,
                                       mfem::DenseMatrix &Jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   diff_stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                          flux_a.data());
   diff_stack.independent(q_a.data(), q.Size());
   diff_stack.dependent(flux_a.data(), q.Size());
   diff_stack.jacobian(Jac.GetData());
}

template <int dim>
void EulerSolver::calcSlipWallFluxJacDir(const mfem::Vector &x,
                                         const mfem::Vector &dir,
                                         const mfem::Vector &q,
                                         mfem::DenseMatrix &Jac)
{
   // create containers for active double objects for each input
   std::vector<adouble> x_a(x.Size());
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());
   // initialize active double containers with data from inputs
   adept::set_values(x_a.data(), x.Size(), x.GetData());
   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());
   // start new stack recording
   diff_stack.new_recording();
   // create container for active double flux output
   std::vector<adouble> flux_a(q.Size());
   calcSlipWallFlux<adouble, dim>(x_a.data(), dir_a.data(), q_a.data(),
                                          flux_a.data());
   diff_stack.independent(dir_a.data(), dir.Size());
   diff_stack.dependent(flux_a.data(), q.Size());
   diff_stack.jacobian(Jac.GetData());
}

<<<<<<< HEAD
template <int dim>
inline void EulerSolver::calcSpectralRadius(const mfem::Vector &dir,
					    const mfem::Vector &q)
{
   calcSpectralRadius(dir.GetData(), q.GetData());
}

template <int dim>
static void EulerSolver::calcSpectralRadiusJacDir(const mfem::Vector &dir,
					   const mfem::Vector &q,
					   mfem::DenseMatrix &Jac)
{
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());

   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());

   diff_stack.new_recording();

   adouble sr = calcSpectralRadius<adouble, dim>(dir_a.data(), q_a.data());

   diff_stack.independent(dir_a.data(), dir.Size());
   diff_stack.dependent(sr.data());
   diff_stack.jacobian(Jac.GetData());
}

template <int dim>
static void EulerSolver::calcSpectralRadiusJacQ(const mfem::Vector &dir,
                                           const mfem::Vector &q,
                                           mfem::DenseMatrix &Jac)
{
   std::vector<adouble> dir_a(dir.Size());
   std::vector<adouble> q_a(q.Size());

   adept::set_values(dir_a.data(), dir.Size(), dir.GetData());
   adept::set_values(q_a.data(), q.Size(), q.GetData());

   diff_stack.new_recording();

   adouble sr = calcSpectralRadius<adouble, dim>(dir_a.data(), q_a.data());

   diff_stack.independent(q_a.data(), q.Size());
   diff_stack.dependent(sr.data());
   diff_stack.jacobian(Jac.GetData());
}

=======
template<int dim>
void EulerSolver::calcIsmailRoeJacQ(int di, const mfem::Vector &qL, 
                                    const mfem::Vector &qR,
                                    mfem::DenseMatrix &jac)
{
   // import stack and adouble from adept
   using adept::adouble;
   // vector of active input variables
   std::vector<adouble> qL_a(qL.Size())
   std::vector<adouble> qR_a(qR.Size());
   // initialize adouble inputs
   adept::set_values(qL_a.Data(),qL.Size(),qL.GetData());
   adept::set_values(qR_a.Data(),qR.Size(),qR.GetData());
   // start recording
   diff_stack.new_recording();
   // create vector of active output variables
   std::vector<adouble> a_flux(flux.Size());
   // run algorithm
   calcIsmailRoeFlux<adouble,dim>(di,&qL_a[0],&qR_a[0],&flux_a);
   // identify independent and dependent variables
   diff_stack.independent(qL_a.Data(),qL.Size());
   diff_stack.independent(qR_a.Data(),qR.Size());
   diff_stack.dependent(flux_a.Data(),flux.Size());
   // compute and store jacobian in jac ?
   diff_stack.jacobian_reverse(jac.GetData());
}
>>>>>>> 0533f0400e75c14d84989afefec364393b99405c
#endif

} // namespace mach
