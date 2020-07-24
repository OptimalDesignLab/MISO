#include <memory>

#include "rans.hpp"
#include "rans_integ.hpp"

using namespace mfem;
using namespace std;

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
   vector<double> sa = this->options["flow-param"]["sa-consts"].template get<vector<double>>();
   sacs.SetData(sa.data());

}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   // add Navier Stokes integrators
   cout << "Inside RANS add volume integrators" << endl;
   this->res->AddDomainIntegrator(new SAInviscidIntegrator<dim, entvar>( //Inviscid term
       this->diff_stack, alpha));
   // this->res->AddDomainIntegrator(new SAViscousIntegrator<dim>( //SAViscousIntegrator
   //     this->diff_stack, re_fs, pr_fs, mu, alpha));
   // now add RANS integrators
   // this->res->AddDomainIntegrator(new SASourceIntegrator<dim>(
   //     this->diff_stack, sacs, mu, alpha));
   // add LPS stabilization
   // double lps_coeff = this->options["space-dis"]["lps-coeff"].template get<double>();
   // this->res->AddDomainIntegrator(new EntStableLPSIntegrator<dim, entvar>(
   //     this->diff_stack, alpha, lps_coeff));

}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
{
   // add base class integrators
   auto &bcs = this->options["bcs"];
   double mu = this->options["flow-param"]["mu"].template get<double>();

   int idx = 0;
#if 0
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

   if (bcs.find("rans-inflow") != bcs.end())
   {
      vector<int> tmp = bcs["rans-inflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the in-flow state needed by the integrator
      Vector q_in(dim+2);
      getViscousInflowState(q_in);
      // Add the rans-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new RANSInflowBC<dim>(this->diff_stack, this->fec.get(),
                                   re_fs, pr_fs, q_in, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("rans-outflow") != bcs.end())
   {
      vector<int> tmp = bcs["rans-outflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the out-flow state needed by the integrator
      Vector q_out(dim+2);
      getRANSOutflowState(q_out);
      // Add the rans-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new RANSOutflowBC<dim>(this->diff_stack, this->fec.get(),
                                   re_fs, pr_fs, q_out, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
#endif
   if (bcs.find("far-field") != bcs.end())
   { 
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim+3);
      this->getFreeStreamState(qfar);
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new SAFarFieldBC<dim>(this->diff_stack, this->fec.get(), qfar,
                              alpha),
          this->bndry_marker[idx]);
      idx++;
   }
#if 0
   if (bcs.find("viscous-shock") != bcs.end())
   { 
      throw MachException("Viscous shock bc not implemented!");
   }
#endif

}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::getRANSInflowState(Vector &q_in)
{
   vector<double> tmp = this->options["flow-param"]
                                     ["inflow-state"]
                                         .template get<vector<double>>();
   if (tmp.size() != dim+3)
   {
      throw MachException("inflow-state option has wrong number of entries"
                          " for problem dimension!");
   }
   for (int i = 0; i < dim+3; ++i)
   {
      q_in(i) = tmp[i];
   }
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::getRANSOutflowState(Vector &q_out)
{
   vector<double> tmp = this->options["flow-param"]
                                     ["outflow-state"]
                                         .template get<vector<double>>();
   if (tmp.size() != dim+3)
   {
      throw MachException("outflow-state option has wrong number of entries"
                          " for problem dimension!");
   }
   for (int i = 0; i < dim+3; ++i)
   {
      q_out(i) = tmp[i];
   }
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::getFreeStreamState(mfem::Vector &q_ref) 
{
   q_ref = 0.0;
   q_ref(0) = 1.0;
   if (dim == 1)
   {
      q_ref(1) = q_ref(0)*this->mach_fs; // ignore angle of attack
   }
   else
   {
      q_ref(this->iroll+1) = q_ref(0)*this->mach_fs*cos(this->aoa_fs);
      q_ref(this->ipitch+1) = q_ref(0)*this->mach_fs*sin(this->aoa_fs);
   }
   q_ref(dim+1) = 1/(euler::gamma*euler::gami) + 0.5*this->mach_fs*this->mach_fs;
   q_ref(dim+2) = this->chi_fs*(mu/q_ref(0));
}

// explicit instantiation
template class RANavierStokesSolver<1>;
template class RANavierStokesSolver<2>;
template class RANavierStokesSolver<3>;

}//namespace mach

