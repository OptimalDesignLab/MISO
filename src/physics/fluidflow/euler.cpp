#include <memory>

#include "sbp_fe.hpp"
#include "euler.hpp"
//#include "euler_fluxes.hpp"
#include "euler_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

template <int dim>
EulerSolver<dim>::EulerSolver(const string &opt_file_name,
                              unique_ptr<mfem::Mesh> smesh)
    : AbstractSolver(opt_file_name, move(smesh))
{
   // define free-stream parameters; may or may not be used, depending on case
   mach_fs = options["flow-param"]["mach"].template get<double>();
   aoa_fs = options["flow-param"]["aoa"].template get<double>()*M_PI/180;
   iroll = options["flow-param"]["roll-axis"].template get<int>();
   ipitch = options["flow-param"]["pitch-axis"].template get<int>();
   if (iroll == ipitch)
   {
      throw MachException("iroll and ipitch must be distinct dimensions!");
   }
   if ( (iroll < 0) || (iroll > 2) )
   {
      throw MachException("iroll axis must be between 0 and 2!");
   }
   if ( (ipitch < 0) || (ipitch > 2) )
   {
      throw MachException("ipitch axis must be between 0 and 2!");
   }
}

template <int dim>
void EulerSolver<dim>::addVolumeIntegrators(double alpha)
{
   // TODO: if statement when using entropy variables as state variables

   // TODO: should decide between one-point and two-point fluxes using options
   res->AddDomainIntegrator(
       new IsmailRoeIntegrator<dim, false>(diff_stack, alpha));
   //res->AddDomainIntegrator(new EulerIntegrator<dim>(diff_stack, alpha));

   // add the LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].template get<double>();
   res->AddDomainIntegrator(
       new EntStableLPSIntegrator<dim, false>(diff_stack, alpha, lps_coeff));
}

template <int dim>
void EulerSolver<dim>::addBoundaryIntegrators(double alpha)
{
   auto &bcs = options["bcs"];
   int idx = 0;
   if (bcs.find("vortex") != bcs.end())
   { // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException("EulerSolver::addBoundaryIntegrators(alpha)\n"
                             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> tmp = bcs["vortex"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new IsentropicVortexBC<dim, false>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
             new SlipWallBC<dim, false>(diff_stack, fec.get(), alpha),
             bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("far-field") != bcs.end())
   { 
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim+2);
      getFreeStreamState(qfar);
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
          new FarFieldBC<dim, false>(diff_stack, fec.get(), qfar, alpha),
          bndry_marker[idx]);
      idx++;
   }
}

template <int dim>
void EulerSolver<dim>::addInterfaceIntegrators(double alpha)
{
   // add the integrators based on if discretization is continuous or discrete
   if (options["space-dis"]["basis-type"].template get<string>() == "dsbp")
   {
      res->AddInteriorFaceIntegrator(
          new InterfaceIntegrator<dim, false>(diff_stack, fec.get(), alpha));
   }
}

template <int dim>
void EulerSolver<dim>::addOutputs()
{
   auto &fun = options["outputs"];
   int idx = 0;
   if (fun.find("drag") != fun.end())
   { 
      // drag on the specified boundaries
      vector<int> tmp = fun["drag"].template get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
      output.emplace("drag", fes.get());
      mfem::Vector drag_dir(dim);
      drag_dir = 0.0;
      if (dim == 1)
      {
         drag_dir(0) = 1.0;
      }
      else 
      {
         drag_dir(iroll) = cos(aoa_fs);
         drag_dir(ipitch) = sin(aoa_fs);
      }
      output.at("drag").AddBdrFaceIntegrator(
          new PressureForce<dim>(diff_stack, fec.get(), drag_dir),
          output_bndry_marker[idx]);
      idx++;
   }
   if (fun.find("lift") != fun.end())
   { 
      // lift on the specified boundaries
      vector<int> tmp = fun["lift"].template get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
      output.emplace("lift", fes.get());
      mfem::Vector lift_dir(dim);
      lift_dir = 0.0;
      if (dim == 1)
      {
         lift_dir(0) = 0.0;
      }
      else
      {
         lift_dir(iroll) = -sin(aoa_fs);
         lift_dir(ipitch) = cos(aoa_fs);
      }
      output.at("lift").AddBdrFaceIntegrator(
          new PressureForce<dim>(diff_stack, fec.get(), lift_dir),
          output_bndry_marker[idx]);
      idx++;
   }
}

template <int dim>
double EulerSolver<dim>::calcStepSize(double cfl) const
{
   double (*calcSpect)(const double *dir, const double *q);
   calcSpect = calcSpectralRadius<double, dim>;
   double dt_local = 1e100;
   Vector xi(dim);
   Vector dxij(dim);
   Vector ui, dxidx;
   DenseMatrix uk;
   DenseMatrix adjJt(dim);
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

template <int dim>
void EulerSolver<dim>::getFreeStreamState(mfem::Vector &q_ref) 
{
   q_ref = 0.0;
   q_ref(0) = 1.0;
   if (dim == 1)
   {
      q_ref(1) = q_ref(0)*mach_fs; // ignore angle of attack
   }
   else
   {
      q_ref(iroll+1) = q_ref(0)*mach_fs*cos(aoa_fs);
      q_ref(ipitch+1) = q_ref(0)*mach_fs*sin(aoa_fs);
   }
   q_ref(dim+1) = 1/(euler::gamma*euler::gami) + 0.5*mach_fs*mach_fs;
}

// explicit instantiation
template class EulerSolver<1>;
template class EulerSolver<2>;
template class EulerSolver<3>;

} // namespace mach
