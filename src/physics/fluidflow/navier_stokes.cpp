#include <cmath>
#include <memory>

#include "functional_output.hpp"
#include "navier_stokes_integ.hpp"

#include "navier_stokes.hpp"

using namespace mfem;
using namespace std;

namespace miso
{
template <int dim, bool entvar>
NavierStokesSolver<dim, entvar>::NavierStokesSolver(
    const nlohmann::json &json_options,
    unique_ptr<mfem::Mesh> smesh,
    MPI_Comm comm)
 : EulerSolver<dim, entvar>(json_options, move(smesh), comm)
{
   // define NS-related parameters; may or may not be used, depending on case
   re_fs = this->options["flow-param"]["Re"].template get<double>();
   pr_fs = this->options["flow-param"]["Pr"].template get<double>();

   // Note: the viscous terms are added to the semi-linear form `res` via
   // virtual function calls to addVolumeIntegrators(), addBoundaryIntegrators,
   // and addInterfaceIntegrators().
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   // add base class integrators
   EulerSolver<dim, entvar>::addResVolumeIntegrators(alpha);
   cout << "Inside NS add volume integrators" << endl;
   // now add NS integrators
   auto mu = this->options["flow-param"]["mu"].template get<double>();
   this->res->AddDomainIntegrator(
       new ESViscousIntegrator<dim>(this->diff_stack, re_fs, pr_fs, mu, alpha));
   if (this->options["flow-param"]["viscous-mms"].template get<bool>())
   {
      if (dim != 2)
      {
         this->res->AddDomainIntegrator(
            new NavierStokesMMSIntegrator(re_fs, pr_fs, -1., 3));
      }
      else
      {
         this->res->AddDomainIntegrator(
          new NavierStokesMMSIntegrator(re_fs, pr_fs));
      }

   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
{
   auto &bcs = this->options["bcs"];
   auto mu = this->options["flow-param"]["mu"].template get<double>();
   int idx = 0;
   if (bcs.find("slip-wall") != bcs.end())
   {  // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new ViscousSlipWallBC<dim>(
              this->diff_stack, this->fec.get(), re_fs, pr_fs, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("no-slip-adiabatic") != bcs.end())
   {
      vector<int> tmp = bcs["no-slip-adiabatic"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // reference state needed by penalty flux
      Vector q_ref(dim + 2);
      this->getFreeStreamState(q_ref);
      // Add the adiabatic flux BC
      this->res->AddBdrFaceIntegrator(
          new NoSlipAdiabaticWallBC<dim>(this->diff_stack,
                                         this->fec.get(),
                                         re_fs,
                                         pr_fs,
                                         q_ref,
                                         mu,
                                         alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-inflow") != bcs.end())
   {
      vector<int> tmp = bcs["viscous-inflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the in-flow state needed by the integrator
      Vector q_in(dim + 2);
      getViscousInflowState(q_in);
      // Add the viscous-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new ViscousInflowBC<dim>(
              this->diff_stack, this->fec.get(), re_fs, pr_fs, q_in, mu, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-outflow") != bcs.end())
   {
      vector<int> tmp = bcs["viscous-outflow"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      // get the out-flow state needed by the integrator
      Vector q_out(dim + 2);
      getViscousOutflowState(q_out);
      // Add the viscous-inflow integrator
      this->res->AddBdrFaceIntegrator(
          new ViscousOutflowBC<dim>(this->diff_stack,
                                    this->fec.get(),
                                    re_fs,
                                    pr_fs,
                                    q_out,
                                    mu,
                                    alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("far-field") != bcs.end())
   {
      // far-field boundary conditions
      vector<int> tmp = bcs["far-field"].template get<vector<int>>();
      mfem::Vector qfar(dim + 2);
      this->getFreeStreamState(qfar);
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(
          new FarFieldBC<dim, entvar>(
              this->diff_stack, this->fec.get(), qfar, alpha),
          this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-shock") != bcs.end())
   {
      // viscous shock boundary conditions
      vector<int> tmp = bcs["viscous-shock"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      this->res->AddBdrFaceIntegrator(new ViscousExactBC<dim>(this->diff_stack,
                                                              this->fec.get(),
                                                              re_fs,
                                                              pr_fs,
                                                              shockExact,
                                                              mu,
                                                              alpha),
                                      this->bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("viscous-mms") != bcs.end())
   {
      // viscous MMS boundary conditions
      vector<int> tmp = bcs["viscous-mms"].template get<vector<int>>();
      this->bndry_marker[idx].SetSize(tmp.size(), 0);
      this->bndry_marker[idx].Assign(tmp.data());
      auto exactbc = [](const Vector &x, Vector &u)
      { viscousMMSExact<double>(dim, x.GetData(), u.GetData()); };
      this->res->AddBdrFaceIntegrator(new ViscousExactBC<dim>(this->diff_stack,
                                                              this->fec.get(),
                                                              re_fs,
                                                              pr_fs,
                                                              exactbc,
                                                              mu,
                                                              alpha),
                                      this->bndry_marker[idx]);
      idx++;
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addResInterfaceIntegrators(double alpha)
{
   // add base class integrators
   EulerSolver<dim, entvar>::addResInterfaceIntegrators(alpha);
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::addOutput(const std::string &fun,
                                                const nlohmann::json &options)
{
   auto mu = this->options["flow-param"]["mu"].template get<double>();
   Vector q_ref(dim + 2);
   this->getFreeStreamState(q_ref);
   if (fun == "drag")
   {
      // drag on the specified boundaries
      auto bdrs = options["boundaries"].template get<vector<int>>();

      mfem::Vector drag_dir(dim);
      drag_dir = 0.0;
      if (dim == 1)
      {
         drag_dir(0) = 1.0;
      }
      else
      {
         drag_dir(this->iroll) = cos(this->aoa_fs);
         drag_dir(this->ipitch) = sin(this->aoa_fs);
      }
      drag_dir *= 1.0 / pow(this->miso_fs, 2.0);  // to get non-dimensional Cd

      FunctionalOutput out(*(this->fes), this->res_fields);
      out.addOutputBdrFaceIntegrator(new SurfaceForce<dim>(this->diff_stack,
                                                           this->fec.get(),
                                                           dim + 2,
                                                           re_fs,
                                                           pr_fs,
                                                           q_ref,
                                                           drag_dir,
                                                           mu),
                                     std::move(bdrs));
      this->outputs.emplace(fun, std::move(out));
   }
   else if (fun == "lift")
   {
      // lift on the specified boundaries
      auto bdrs = options["boundaries"].template get<vector<int>>();

      mfem::Vector lift_dir(dim);
      lift_dir = 0.0;
      if (dim == 1)
      {
         lift_dir(0) = 0.0;
      }
      else
      {
         lift_dir(this->iroll) = -sin(this->aoa_fs);
         lift_dir(this->ipitch) = cos(this->aoa_fs);
      }
      lift_dir *= 1.0 / pow(this->miso_fs, 2.0);  // to get non-dimensional Cl

      FunctionalOutput out(*(this->fes), this->res_fields);
      out.addOutputBdrFaceIntegrator(new SurfaceForce<dim>(this->diff_stack,
                                                           this->fec.get(),
                                                           dim + 2,
                                                           re_fs,
                                                           pr_fs,
                                                           q_ref,
                                                           lift_dir,
                                                           mu),
                                     std::move(bdrs));
      this->outputs.emplace(fun, std::move(out));
   }
   else if (fun == "entropy")
   {
      // integral of entropy over the entire volume domain
      FunctionalOutput out(*(this->fes), this->res_fields);
      out.addOutputDomainIntegrator(
          new EntropyIntegrator<dim, entvar>(this->diff_stack));
      this->outputs.emplace(fun, std::move(out));
   }
   else
   {
      throw MISOException("Output with name " + fun +
                          " not supported by "
                          "NavierStokesSolver!\n");
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::getViscousInflowState(Vector &q_in)
{
   vector<double> tmp = this->options["flow-param"]["inflow-state"]
                            .template get<vector<double>>();
   if (tmp.size() != dim + 2)
   {
      throw MISOException(
          "inflow-state option has wrong number of entries"
          " for problem dimension!");
   }
   for (int i = 0; i < dim + 2; ++i)
   {
      q_in(i) = tmp[i];
   }
}

template <int dim, bool entvar>
void NavierStokesSolver<dim, entvar>::getViscousOutflowState(Vector &q_out)
{
   vector<double> tmp = this->options["flow-param"]["outflow-state"]
                            .template get<vector<double>>();
   if (tmp.size() != dim + 2)
   {
      throw MISOException(
          "outflow-state option has wrong number of entries"
          " for problem dimension!");
   }
   for (int i = 0; i < dim + 2; ++i)
   {
      q_out(i) = tmp[i];
   }
}

// explicit instantiation
template class NavierStokesSolver<1>;
template class NavierStokesSolver<2>;
template class NavierStokesSolver<3>;

double shockEquation(double Re, double Ma, double v)
{
   double vf = (2.0 + euler::gami * Ma * Ma) / ((euler::gamma + 1) * Ma * Ma);
   double alpha = (8 * euler::gamma) / (3 * (euler::gamma + 1) * Re * Ma);
   // double r = (1 + vf) / (1 - vf);
   double a = abs((v - 1) * (v - vf));
   double b = (1 + vf) / (1 - vf);
   double c = abs((v - 1) / (v - vf));
   return 0.5 * alpha * (log(a) + b * log(c));
}

// Exact solution
void shockExact(const mfem::Vector &x, mfem::Vector &u)
{
   double Re = 10.0;  // !!!!! Values from options file are ignored
   double Ma = 2.5;
   double vf = (2.0 + euler::gami * Ma * Ma) / ((euler::gamma + 1) * Ma * Ma);
   double ftol = 1e-10;
   double xtol = 1e-10;
   int maxiter = 50;
   double v = NAN;
   if (x(0) < -1.25)
   {
      v = 1.0;
   }
   else if (x(0) > 0.4)
   {
      v = vf;
   }
   else
   {
      // define a lambda function for equation (7.5)
      auto func = [&](double vroot)
      { return x(0) - shockEquation(Re, Ma, vroot); };
      v = bisection(func, 1.0000001 * vf, 0.9999999, ftol, xtol, maxiter);
   }
   double vel = v * Ma;
   u(0) = Ma / vel;  // rho*u = M_L
   u(1) = Ma;
   u(2) = 0.0;
   u(3) = 0.5 * (euler::gamma + 1) * vf * u(0) * Ma * Ma /
              (euler::gamma * euler::gami) +
          euler::gami * Ma * vel / (2 * euler::gamma);
}

// MMS Exact solution
void viscousMMSExact(int dim, const mfem::Vector &x, mfem::Vector &u)
{  std::cout << "we are here \n";
   u.SetSize(dim+2);
   switch(dim)
   {
      case 3:
         {
            const double r_0 = 1.0;
            const double r_xyz = 1.0;
            const double u_0 = 0.0;
            const double v_0 = 0.0;
            const double w_0 = 0.0;
            const double T_0 = 1.0;

            u(0) = r_0 + r_0*0.1*sin(2*r_xyz*M_PI*x(0))*sin(2*r_xyz*M_PI*x(1))*sin(2*r_xyz*M_PI*x(2));
            u(1) = u_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.));
            u(2) = v_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.));
            u(3) = w_0*((pow(x(0),3)/3. - pow(x(0),2)/2.) + (pow(x(1),3)/3. - pow(x(1),2)/2.) + (pow(x(2),3)/3. - pow(x(2),2)/2.));
            double T = T_0;
            double p = u(0) * T;
            u(4) = p/euler::gami + 0.5 * u(0) * (u(1)*u(1) + u(2)*u(2) + u(3)*u(3));
            break;
         }
      default:
         {
            const double rho0 = 1.0;
            const double rhop = 0.05;
            const double U0 = 0.5;
            const double Up = 0.05;
            const double T0 = 1.0;
            const double Tp = 0.05;

            u(0) = rho0 + rhop * pow(sin(M_PI * x(0)), 2) * sin(M_PI * x(1));
            u(1) = 4.0 * U0 * x(1) * (1.0 - x(1)) +
                  Up * sin(2 * M_PI * x(1)) * pow(sin(M_PI * x(0)), 2);
            u(2) = -Up * pow(sin(2 * M_PI * x(0)), 2) * sin(M_PI * x(1));
            double T = T0 + Tp * (pow(x(0), 4) - 2 * pow(x(0), 3) + pow(x(0), 2) +
                                 pow(x(1), 4) - 2 * pow(x(1), 3) + pow(x(1), 2));
            double p = u(0) * T;  // T is nondimensionalized by 1/(R*a_infty^2)
            u(3) = p / euler::gami + 0.5 * u(0) * (u(1) * u(1) + u(2) * u(2));
            u(1) *= u(0);
            u(2) *= u(0);
            break;
         }
   }
}

}  // namespace miso
