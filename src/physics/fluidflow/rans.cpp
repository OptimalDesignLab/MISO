#include <memory>

#include "rans.hpp"
#include "rans_integ.hpp"

using namespace mfem;
using namespace std;
#if 0
namespace mach
{

template <int dim, bool entvar>
RANavierStokesSolver<dim, entvar>::RANavierStokesSolver(const string &opt_file_name,
                                            unique_ptr<mfem::Mesh> smesh)
    : NavierStokesSolver<dim, entvar>(opt_file_name, move(smesh))
{
   if (entvar)
   {
      throw MachException("Entropy variables not implemented for RANS!");
   }

   // define free-stream parameters; may or may not be used, depending on case
   chi_fs = this->options["flow-param"]["chi"].template get<double>();
   mu = this->options["flow-param"]["mu"].template get<double>();
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   // add Navier Stokes integrators
   cout << "Inside RANS add volume integrators" << endl;
   this->res->AddDomainIntegrator(new IsmailRoeIntegrator<dim, entvar>( //Inviscid term
       this->diff_stack, alpha));
   this->res->AddDomainIntegrator(new SAViscousIntegrator<dim>( //SAViscousIntegrator
       this->diff_stack, re_fs, pr_fs, mu, alpha));
   // now add RANS integrators
   // combine these, so we compute vorticity once
   this->res->AddDomainIntegrator(new SASourceIntegrator<dim>(
       this->diff_stack, alpha));
   // add LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].template get<double>();
   this->res->AddDomainIntegrator(new EntStableLPSIntegrator<dim, entvar>(
       this->diff_stack, alpha, lps_coeff));

}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
{
   // add base class integrators
   auto &bcs = this->options["bcs"];
   double mu = this->options["flow-param"]["mu"].template get<double>();
   int idx = 0;
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new SAViscousSlipWallBC<dim>(this->diff_stack, this->fec.get(),
                                     re_fs, pr_fs, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("no-slip-adiabatic") != bcs.end())
   {
      vector<int> tmp = bcs["no-slip-adiabatic"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // reference state needed by penalty flux
      Vector q_ref(dim+3);
      this->getFreeStreamState(q_ref);
      // Add the adiabatic flux BC
      this->res->AddBdrFaceIntegrator(
          new SANoSlipAdiabaticWallBC<dim>(
              this->diff_stack, this->fec.get(), re_fs, pr_fs, q_ref, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-inflow") != bcs.end())
   {
      throw MachException("Viscous inflow bc not implemented!");
   }
   if (bcs.find("viscous-outflow") != bcs.end())
   {
      throw MachException("Viscous outflow bc not implemented!");
   }
   if (bcs.find("far-field") != bcs.end())
   { 
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim+3);
      this->getFreeStreamState(qfar);
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new SAFarFieldBC<dim, entvar>(this->diff_stack, this->fec.get(), qfar,
                              alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-shock") != bcs.end())
   { 
      throw MachException("Viscous shock bc not implemented!");
   }
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::getFreeStreamState(mfem::Vector &q_ref) 
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
   q_ref(dim+2) = chi_fs*(mu/q_ref(0));
}

// explicit instantiation
template class RANavierStokesSolver<1>;
template class RANavierStokesSolver<2>;
template class RANavierStokesSolver<3>;

}//namespace mach

#endif