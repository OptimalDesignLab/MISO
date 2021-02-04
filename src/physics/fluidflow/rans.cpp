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
   this->start_up = this->options["time-dis"]["start-up"].template get<bool>();
   vector<double> sa = this->options["flow-param"]["sa-consts"].template get<vector<double>>();
   sacs.SetSize(13);
   sacs = sa.data();
   getDistanceFunction();
}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   vector<double> srcs = this->options["flow-param"]["sa-srcs"].template get<vector<double>>();
   Vector q_ref(dim+3);
   this->getFreeStreamState(q_ref);
   double d0 = getZeroDistance();
   bool mms = this->options["flow-param"]["sa-mms"].template get<bool>();

   // MMS option
   if (mms)
   {
      if (dim != 2)
      {
         throw MachException("SA MMS problem only available for 2D!");
      }
      this->res->AddDomainIntegrator(
          new SAMMSIntegrator(this->re_fs, this->pr_fs));
      this->res->AddDomainIntegrator(new SASourceIntegrator<dim>(
       this->diff_stack, *dist, this->re_fs, sacs, mu, alpha, srcs[0], srcs[1], d0)); 
   }
   else
   {
      
      this->res->AddDomainIntegrator(new SAInviscidIntegrator<dim, entvar>( //Inviscid term
            this->diff_stack, alpha));
      this->res->AddDomainIntegrator(new SAViscousIntegrator<dim>( //SAViscousIntegrator
            this->diff_stack, this->re_fs, this->pr_fs, sacs, mu, alpha));
      // now add RANS integrators
      this->res->AddDomainIntegrator(new SASourceIntegrator<dim>(
            this->diff_stack, *dist, this->re_fs, sacs, mu, alpha, srcs[0], srcs[1], d0)); 
      // add LPS stabilization
      double lps_coeff = this->options["space-dis"]["lps-coeff"].template get<double>();
      this->res->AddDomainIntegrator(new SALPSIntegrator<dim, entvar>(
            this->diff_stack, alpha, lps_coeff));
   }

}

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
{
   auto &bcs = this->options["bcs"];
   double mu = this->options["flow-param"]["mu"].template get<double>();

   int idx = 0;

   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      
      Vector q_ref(dim+3);
      this->getFreeStreamState(q_ref);
      this->res->AddBdrFaceIntegrator(
          new SAViscousSlipWallBC<dim>(this->diff_stack, this->fec.get(),
                                     this->re_fs, this->pr_fs, sacs, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }

   if (bcs.find("no-slip-adiabatic") != bcs.end())
   { // no-slip-wall boundary condition
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
   q_ref(dim+2) = this->chi_fs*mu;
}

static void pert(const Vector &x, Vector& p);

template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::iterationHook(int iter, 
                                                      double t, double dt) 
{
   string file = this->options["file-names"].template get<std::string>();
   stringstream filename;
   filename << file <<"_last";
   int num_rans = this->res->Height();
   Array<int> disable_ns(num_rans);
   //disable_ns = NULL;
   //this->res->SetEssentialTrueDofs(disable_ns);

   this->checkJacobian(pert);
   this->printSolution(filename.str(), 0);
   
   stringstream solname;
	solname << file <<"_last.gf";
   ofstream ssol(solname.str()); ssol.precision(16);
	this->u->Save(ssol);

   stringstream rsolname;
	rsolname << file <<"_last_res.gf";
   ofstream rsol(rsolname.str()); rsol.precision(16);
	GridFunction r(this->u->FESpace());
   this->res->Mult(*this->u, r);
   r.Save(rsol);
   double res_norm = this->calcResidualNorm();

   cout << "Iter "<<iter<<" Residual Norm: "<<res_norm<<endl;

   // if(this->start_up)
   //    cout << "Start-Up Phase"<<endl;

   // // disable updates to NS equation to help SA converge on its own
   // if(iter > 1 && res_norm/this->res_norm0 > 1e-9 && sa_conv == false)
   // {
   //    cout << "Navier Stokes Disabled"<<endl;
   //    this->start_up = true;
   //    for(int s = 0; s < num_rans; s++)
   //    {
   //       if((s+1)%(dim+3) == 0) 
   //          disable_ns[s] = s-1;
   //       else
   //          disable_ns[s] = s;
   //    }
   // }
   // else if (iter <= 1)
   //    disable_ns = NULL;
   // else
   // {
   //    disable_ns = NULL;
   //    sa_conv = true;
   //    this->start_up = false;
   // }
   // this->res->SetEssentialTrueDofs(disable_ns);
}


template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::terminalHook(int iter, double t_final)
{
   // double entropy = ent->GetEnergy(*u);
   // entropylog << t_final << ' ' << entropy << endl;
   // entropylog.close();

   string file = this->options["file-names"].template get<std::string>();
   stringstream filename;
   filename << file <<"_final";
   stringstream solname;
	solname << file <<"_final.gf";
   ofstream ssol(solname.str()); ssol.precision(30);
	this->u->Save(ssol);
}


template <int dim, bool entvar>
void RANavierStokesSolver<dim, entvar>::getDistanceFunction()
{
   std::string wall_type = 
      this->options["wall-func"]["type"].template get<std::string>();
   int fe_order = this->options["space-dis"]["degree"].template get<int>();
   H1_FECollection *dfec = new H1_FECollection(fe_order, dim);
   FiniteElementSpace *dfes = new FiniteElementSpace(this->mesh.get(), dfec);
   dist.reset(new GridFunction(dfes));

   if (wall_type == "const") // uniform d value
   { 
      double val = this->options["wall-func"]["val"].template get<double>();
      ConstantCoefficient wall_coeff(val);
      dist->ProjectCoefficient(wall_coeff);
   }
   if (wall_type == "y-dist") // y distance from the origin
   {
      auto walldist = [](const Vector &x)
      {
         // if(x(1) == 0.0)
         //    return 0.25*offset; //// not going to do it this way
         // else
            return x(1); 
      };
      FunctionCoefficient wall_coeff(walldist);
      dist->ProjectCoefficient(wall_coeff);
   }
   if (wall_type == "true") // true computed distance function
   {
      // assemble wall attribute vector
      auto &bcs = this->options["bcs"];
      vector<int> tmp = bcs["no-slip-adiabatic"].template get<vector<int>>();
      vector<int> tmp2 = bcs["slip-wall"].template get<vector<int>>();
      for(int i = 0; i < tmp.size(); i++)
      {
         tmp[i] = tmp[i] + tmp2[i];
      }
      mfem::Array<int> wall_marker; 
      wall_marker.SetSize(tmp.size(), 0);
      wall_marker.Assign(tmp.data());

      // create surface object
      Surface<dim> surf(*this->mesh, wall_marker);

      // project surface function
      auto walldist = [&surf](const Vector &x)
      {
         return surf.calcDistance(x); 
      };
      FunctionCoefficient wall_coeff(walldist);
      dist->ProjectCoefficient(wall_coeff);
   } 
   ///TODO: Differentiate true wall function
}

template <int dim, bool entvar>
double RANavierStokesSolver<dim, entvar>::getZeroDistance()
{
   std::string wall_type = 
      this->options["wall-func"]["type"].template get<std::string>();
   // H1_FECollection *dfec = new H1_FECollection(1, dim);
   // FiniteElementSpace *dfes = new FiniteElementSpace(this->mesh.get(), dfec);
   // unique_ptr<GridFunction> distcheck;
   // distcheck.reset(new GridFunction(dfes));

   double d0 = 0.0;

   if (wall_type == "const") // uniform d value
   { 
      // doesn't make any difference
      double val = this->options["wall-func"]["val"].template get<double>();
      d0 = val;
   }
   else
   {
      // this should probably work for any distance function 
      double work = 1e15;
      // exhaustive search of the mesh for the smallest d value
      for(int i = 0; i < dist->Size(); i++)
      {
         if(dist->Elem(i) < work && dist->Elem(i) > 1e-15)
         {
            work = dist->Elem(i);
         }
      }      

      // half the smallest computed distance from the wall
      d0 = work/2;
   } 
   ///TODO: Add option for proper wall distance function 

   cout << "At-wall distance: "<<d0<<endl;
   return d0;
}

std::default_random_engine gen_ns(std::random_device{}());
std::uniform_real_distribution<double> normal_rand_ns(-1.0,1.0);

// perturbation function used to check the jacobian in each iteration
void pert(const Vector &x, Vector& p)
{
   p.SetSize(5);
   for (int i = 0; i < 5; i++)
   {
      p(i) = normal_rand_ns(gen_ns);
   }
}

// explicit instantiation
template class RANavierStokesSolver<1>;
template class RANavierStokesSolver<2>;
template class RANavierStokesSolver<3>;

}//namespace mach

