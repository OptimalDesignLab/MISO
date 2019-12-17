#include <memory>
#include "navier_stokes.hpp"
#include "navier_stokes_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

template <int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const string &opt_file_name,
                                            unique_ptr<mfem::Mesh> smesh)
    : EulerSolver<dim>(opt_file_name, move(smesh))
{
   // define NS-related parameters; may or may not be used, depending on case
   re_fs = this->options["flow-param"]["Re"].template get<double>();
   pr_fs = this->options["flow-param"]["Pr"].template get<double>();

   // Note: the viscous terms are added to the semi-linear form `res` via
   // virtual function calls to addVolumeIntegrators(), addBoundaryIntegrators,
   // and addInterfaceIntegrators(). 

   // add the output functional QoIs 
   //addOutputs(dim);
}

template <int dim>
void NavierStokesSolver<dim>::addVolumeIntegrators(double alpha)
{
   // add base class integrators
   EulerSolver<dim>::addVolumeIntegrators(alpha);
   cout << "Inside NS add volume integrators" << endl;
   // now add NS integrators
   double mu = this->options["flow-param"]["mu"].template get<double>();
   this->res->AddDomainIntegrator(new ESViscousIntegrator<dim>(
       this->diff_stack, re_fs, pr_fs, mu, alpha));
}

template <int dim>
void NavierStokesSolver<dim>::addBoundaryIntegrators(double alpha)
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
          new ViscousSlipWallBC<dim>(this->diff_stack, this->fec.get(),
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
      Vector q_ref(dim+2);
      this->getFreeStreamState(q_ref);
      // Add the adiabatic flux BC
      this->res->AddBdrFaceIntegrator(
          new NoSlipAdiabaticWallBC<dim>(this->diff_stack, this->fec.get(),
                                         re_fs, pr_fs, q_ref, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-inflow") != bcs.end())
   {
      vector<int> tmp = bcs["viscous-inflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the in-flow state needed by the integrator
      Vector q_in(dim+2);
      getViscousInflowState(q_in);
      // Add the viscous-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new ViscousInflowBC<dim>(this->diff_stack, this->fec.get(),
                                   re_fs, pr_fs, q_in, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-outflow") != bcs.end())
   {
      vector<int> tmp = bcs["viscous-outflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the out-flow state needed by the integrator
      Vector q_out(dim+2);
      getViscousOutflowState(q_out);
      // Add the viscous-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new ViscousOutflowBC<dim>(this->diff_stack, this->fec.get(),
                                   re_fs, pr_fs, q_out, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
}

template <int dim>
void NavierStokesSolver<dim>::addInterfaceIntegrators(double alpha)
{
}

template <int dim>
void NavierStokesSolver<dim>::getViscousInflowState(Vector &q_in)
{
   vector<double> tmp = this->options["flow-param"]
                                     ["inflow-state"]
                                         .template get<vector<double>>();
   if (tmp.size() != dim+2)
   {
      throw MachException("inflow-state option has wrong number of entries"
                          " for problem dimension!");
   }
   for (int i = 0; i < dim+2; ++i)
   {
      q_in(i) = tmp[i];
   }
}

template <int dim>
void NavierStokesSolver<dim>::getViscousOutflowState(Vector &q_out)
{
   vector<double> tmp = this->options["flow-param"]
                                     ["outflow-state"]
                                         .template get<vector<double>>();
   if (tmp.size() != dim+2)
   {
      throw MachException("outflow-state option has wrong number of entries"
                          " for problem dimension!");
   }
   for (int i = 0; i < dim+2; ++i)
   {
      q_out(i) = tmp[i];
   }
}

// explicit instantiation
//template class NavierStokesSolver<1>;
template class NavierStokesSolver<2>;
//template class NavierStokesSolver<3>;


} // namespace mach 

