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
   re_fs = this->options["flow-param"]["Re"].get<double>();
   pr_fs = this->options["flow-param"]["Pr"].get<double>();

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
   // now add NS integrators
   this->res->AddDomainIntegrator(new ESViscousIntegrator<dim>(
       this->diff_stack, re_fs, pr_fs, alpha));
}

template <int dim>
void NavierStokesSolver<dim>::addBoundaryIntegrators(double alpha)
{
   // add base class integrators
   EulerSolver<dim>::addBoundaryIntegrators(alpha);
   auto &bcs = this->options["bcs"];
   //bndry_marker.resize(bcs.size());
   int idx = 0;
   if (bcs.find("no-slip-adiabatic") != bcs.end())
   {
      vector<int> tmp = bcs["no-slip-adiabatic"].get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // reference state needed by penalty flux
      Vector q_ref(dim+2);
      this->getFreeStreamState(q_ref);
      // Add the adiabatic flux BC
      this->res->AddBdrFaceIntegrator(
          new NoSlipAdiabaticWallBC<dim>(this->diff_stack, this->fec.get(),
                                         re_fs, pr_fs, q_ref, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
}

template <int dim>
void NavierStokesSolver<dim>::addInterfaceIntegrators(double alpha)
{
}

// explicit instantiation
template class NavierStokesSolver<1>;
template class NavierStokesSolver<2>;
template class NavierStokesSolver<3>;


} // namespace mach 

