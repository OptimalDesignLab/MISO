#include <memory>
#include "euler.hpp"
//#include "euler_fluxes.hpp"
#include "sbp_fe.hpp"
#include "diag_mass_integ.hpp"
#include "euler_integ.hpp"
#include "evolver.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

EulerSolver::EulerSolver(const string &opt_file_name,
                         unique_ptr<mfem::Mesh> smesh, int dim)
    : AbstractSolver(opt_file_name, move(smesh))
{
   // define free-stream parameters; may or may not be used, depending on case
   mach_fs = options["flow-param"]["mach"].get<double>();
   aoa_fs = options["flow-param"]["aoa"].get<double>()*M_PI/180;

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
   // add the integrators based on if discretization is continuous or discrete
   if (options["space-dis"]["basis-type"].get<string>() == "dsbp")
   {
      res->AddInteriorFaceIntegrator(new InterfaceIntegrator<2>(diff_stack,
                                                                fec.get(),
                                                                alpha));
   }
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
   // add the output functional QoIs 
   addOutputs(dim);
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
          new IsentropicVortexBC<2>(diff_stack, fec.get(), alpha),
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

void EulerSolver::addOutputs(int dim)
{
   auto &fun = options["outputs"];
   output_bndry_marker.resize(fun.size());
   int idx = 0;
   if (fun.find("drag") != fun.end())
   { // force on the specified boundaries
      mfem::Vector drag_dir(dim);
      vector<int> tmp = fun["drag"].get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
      output.emplace("drag", fes.get());  
      switch (dim)
      {
      case 1:
         drag_dir(0) = 1.0;
         output.at("drag").AddBdrFaceIntegrator(
             new PressureForce<1>(diff_stack, fec.get(), drag_dir),
                                  output_bndry_marker[idx]);
         break;
      case 2:
         drag_dir(0) = cos(aoa_fs);
         drag_dir(1) = sin(aoa_fs);
         output.at("drag").AddBdrFaceIntegrator(
             new PressureForce<2>(diff_stack, fec.get(), drag_dir),
                                  output_bndry_marker[idx]);
         break;
      case 3:
         drag_dir(1) = cos(aoa_fs);
         drag_dir(2) = sin(aoa_fs);
         output.at("drag").AddBdrFaceIntegrator(
             new PressureForce<3>(diff_stack, fec.get(), drag_dir),
                                  output_bndry_marker[idx]);
         break;
      }
      idx++;
   }
   if (fun.find("lift") != fun.end())
   { // force on the specified boundaries
      mfem::Vector lift_dir(dim);
      vector<int> tmp = fun["lift"].get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
      output.emplace("lift", fes.get());  
      switch (dim)
      {
      case 1:
         lift_dir(0) = 0.0;
         output.at("lift").AddBdrFaceIntegrator(
             new PressureForce<1>(diff_stack, fec.get(), lift_dir),
                                  output_bndry_marker[idx]);
         break;
      case 2:
         lift_dir(0) = -sin(aoa_fs);
         lift_dir(1) = cos(aoa_fs);
         output.at("lift").AddBdrFaceIntegrator(
             new PressureForce<2>(diff_stack, fec.get(), lift_dir),
                                  output_bndry_marker[idx]);
         break;
      case 3:
         lift_dir(1) = -sin(aoa_fs);
         lift_dir(2) = cos(aoa_fs);
         output.at("lift").AddBdrFaceIntegrator(
             new PressureForce<3>(diff_stack, fec.get(), lift_dir),
                                  output_bndry_marker[idx]);
         break;
      }
      idx++;
   }
}

double EulerSolver::calcResidualNorm()
{
   GridFunType r(fes.get());
   double res_norm;
#ifdef MFEM_USE_MPI
   HypreParVector *U = u->GetTrueDofs();
   res->Mult(*U, r);   
   double loc_norm = r*r;
   MPI_Allreduce(&loc_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   res->Mult(*u, r);
   res_norm = r*r;
#endif
   res_norm = sqrt(res_norm);
   return res_norm;
}

double EulerSolver::calcStepSize(double cfl) const
{
   double (*calcSpect)(const double *dir, const double *q);
   if (num_dim == 1)
   {
      calcSpect = calcSpectralRadius<double, 1>;
   }
   else if (num_dim == 2)
   {
      calcSpect = calcSpectralRadius<double, 2>;
   }
   else
   {
      calcSpect = calcSpectralRadius<double, 3>;
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
            if (j == i)
               continue;
            trans->Transform(fe->GetNodes().IntPoint(j), dxij);
            dxij -= xi;
            double dx = dxij.Norml2();
            dt_local = min(dt_local, cfl * dx * dx / calcSpect(dxij, ui)); // extra dx is to normalize dxij
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
