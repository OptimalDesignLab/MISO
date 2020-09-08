#include <memory>
#include <random>

#include "rans.hpp"
#include "rans_integ.hpp"

using namespace mfem;
using namespace std;

namespace mach
{

template <int dim, bool entvar>
RANavierStokesSolver<dim, entvar>::RANavierStokesSolver(const nlohmann::json &json_options,
                                             unique_ptr<mfem::Mesh> smesh)
    : NavierStokesSolver<dim, entvar>(json_options, move(smesh))
{
   if (entvar)
   {
      throw MachException("Entropy variables not implemented for RANS!");
   }

   // define free-stream parameters; may or may not be used, depending on case
   chi_fs = this->options["flow-param"]["chi"].template get<double>();
   mu = this->options["flow-param"]["mu"].template get<double>();
   vector<double> sa = this->options["flow-param"]["sa-consts"].template get<vector<double>>();
   sacs.SetSize(13);
   sacs = sa.data();
   getDistanceFunction();
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   // add Navier Stokes integrators
   cout << "Inside RANS add volume integrators" << endl;
   this->res->AddDomainIntegrator(new SAInviscidIntegrator<dim, entvar>( //Inviscid term
       this->diff_stack, alpha));
   this->res->AddDomainIntegrator(new SAViscousIntegrator<dim>( //SAViscousIntegrator
       this->diff_stack, this->re_fs, this->pr_fs, sacs, mu, alpha));
   // now add RANS integrators
   this->res->AddDomainIntegrator(new SASourceIntegrator<dim>(
       this->diff_stack, *dist, this->re_fs, sacs, mu, -alpha)); 
   // add LPS stabilization
   double lps_coeff = this->options["space-dis"]["lps-coeff"].template get<double>();
   this->res->AddDomainIntegrator(new SALPSIntegrator<dim, entvar>(
       this->diff_stack, alpha, lps_coeff));

}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
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
                                     this->re_fs, this->pr_fs, sacs, mu, alpha),
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
              this->diff_stack, this->fec.get(), this->re_fs, 
                                    this->pr_fs, sacs, q_ref, mu, alpha),
                                    this->bndry_marker[idx]);
      idx++;
   }
#if 0
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
          new SAFarFieldBC<dim>(this->diff_stack, this->fec.get(), 
                              this->re_fs, this->pr_fs, qfar, alpha),
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
   q_ref(dim+2) = q_ref(0)*this->chi_fs*mu;
}

static void pert(const Vector &x, Vector& p);

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::iterationHook(int iter, 
                                                      double t, double dt) 
{
   this->checkJacobian(pert);
   this->printSolution("rans_wall_last", 0);
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::getDistanceFunction()
{
   std::string wall_type = 
      this->options["wall-func"]["type"].template get<std::string>();
   H1_FECollection *dfec = new H1_FECollection(1, dim);
   FiniteElementSpace *dfes = new FiniteElementSpace(this->mesh.get(), dfec);
   dist.reset(new GridFunction(dfes));

   if (wall_type == "const")
   { 
      double val = this->options["wall-func"]["val"].template get<double>();
      ConstantCoefficient wall_coeff(val);
      dist->ProjectCoefficient(wall_coeff);
   }
   if (wall_type == "y-dist")
   {
      double offset = 
         this->options["mesh"]["offset"].template get<double>();
      auto walldist = [offset](const Vector &x)
      {
         if(x(1) == 0.0)
            return 0.25*offset;
         else
            return x(1); 
      };
      FunctionCoefficient wall_coeff(walldist);
      dist->ProjectCoefficient(wall_coeff);
   }   
}

std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(-1.0,1.0);

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      p(i) = normal_rand(gen);
   }
}

// explicit instantiation
template class RANavierStokesSolver<1>;
template class RANavierStokesSolver<2>;
template class RANavierStokesSolver<3>;

}//namespace mach

