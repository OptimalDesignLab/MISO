#include <memory>

#include "sbp_fe.hpp"
#include "euler.hpp"
#include "euler_integ.hpp"
#include "diag_mass_integ.hpp"
#include "euler_sens_integ.hpp"
#include "gauss_hermite.hpp"

using namespace mfem;
using namespace std;


namespace mach
{

template <int dim, bool entvar>
EulerSolver<dim, entvar>::EulerSolver(const string &opt_file_name,
                              unique_ptr<mfem::Mesh> smesh)
    : AbstractSolver(opt_file_name, move(smesh))
{
   if (entvar)
   {
      *out << "The state variables are the entropy variables." << endl;
   }
   else
   {
      *out << "The state variables are the conservative variables." << endl;
   }
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

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::constructForms()
{
   res.reset(new NonlinearFormType(fes.get()));
   if ( (entvar) && (!options["time-dis"]["steady"].get<bool>()) )
   {
      nonlinear_mass.reset(new NonlinearFormType(fes.get()));
      mass.reset();
   }
   else
   {
      mass.reset(new BilinearFormType(fes.get()));
      nonlinear_mass.reset();
   }
   ent.reset(new NonlinearFormType(fes.get()));
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addMassIntegrators(double alpha)
{
   if (options["time-dis"]["steady"].get<bool>()) {
      mass->AddDomainIntegrator(new DiagMassIntegrator(num_state, true));
      //AbstractSolver::addMassIntegrators(alpha);
   }
   else {
      AbstractSolver::addMassIntegrators(alpha);
   }
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addNonlinearMassIntegrators(double alpha)
{
   nonlinear_mass->AddDomainIntegrator(
       new MassIntegrator<dim, entvar>(diff_stack, alpha));
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addResVolumeIntegrators(double alpha)
{
   // TODO: if statement when using entropy variables as state variables

   // TODO: should decide between one-point and two-point fluxes using options
   res->AddDomainIntegrator(
       new IsmailRoeIntegrator<dim, entvar>(diff_stack, alpha));
   //res->AddDomainIntegrator(new EulerIntegrator<dim>(diff_stack, alpha));

   // add the LPS stabilization
   double lps_coeff = options["space-dis"]["lps-coeff"].template get<double>();
   res->AddDomainIntegrator(
       new EntStableLPSIntegrator<dim, entvar>(diff_stack, alpha, lps_coeff));
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addResBoundaryIntegrators(double alpha)
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
          new IsentropicVortexBC<dim, entvar>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res->AddBdrFaceIntegrator(
             new SlipWallBC<dim, entvar>(diff_stack, fec.get(), alpha),
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
          new FarFieldBC<dim, entvar>(diff_stack, fec.get(), qfar, alpha),
          bndry_marker[idx]);
      idx++;
   }
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addResInterfaceIntegrators(double alpha)
{
   // add the integrators based on if discretization is continuous or discrete
   if (options["space-dis"]["basis-type"].template get<string>() == "dsbp")
   {
      double diss_coeff = options["space-dis"]["iface-coeff"].template get<double>();
      res->AddInteriorFaceIntegrator(
          new InterfaceIntegrator<dim, entvar>(diff_stack, diss_coeff,
                                               fec.get(), alpha));
   }
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addEntVolumeIntegrators()
{
   ent->AddDomainIntegrator(new EntropyIntegrator<dim, entvar>(diff_stack));
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::initialHook() 
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm();
   }
   // TODO: this should only be output if necessary
   double entropy = ent->GetEnergy(*u);
   cout << "before time stepping, entropy is "<< entropy << endl;
   remove("entropylog.txt");
   entropylog.open("entropylog.txt", fstream::app);
   entropylog << setprecision(14);
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::iterationHook(int iter, double t, double dt)
{
   double entropy = ent->GetEnergy(*u);
   entropylog << t << ' ' << entropy << endl;
}

template <int dim, bool entvar>
bool EulerSolver<dim, entvar>::iterationExit(int iter, double t, double t_final,
                                             double dt)
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // use tolerance options for Newton's method
      double norm = calcResidualNorm();
      if (norm <= options["time-dis"]["steady-abstol"].get<double>())
         return true;
      if (norm <= res_norm0 *
                      options["time-dis"]["steady-reltol"].get<double>())
         return true;
      return false;
   }
   else
   {
      return AbstractSolver::iterationExit(iter, t, t_final, dt);
   }
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::terminalHook(int iter, double t_final)
{
   double entropy = ent->GetEnergy(*u);
   entropylog << t_final << ' ' << entropy << endl;
   entropylog.close();
}

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::addOutputs()
{
   output.clear();
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
      drag_dir *= 1.0/pow(mach_fs, 2.0); // to get non-dimensional Cd
      output.at("drag").AddBdrFaceIntegrator(
          new PressureForce<dim, entvar>(diff_stack, fec.get(), drag_dir),
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
      lift_dir *= 1.0/pow(mach_fs, 2.0); // to get non-dimensional Cl
      output.at("lift").AddBdrFaceIntegrator(
          new PressureForce<dim, entvar>(diff_stack, fec.get(), lift_dir),
          output_bndry_marker[idx]);
      idx++;
   }
   if (fun.find("entropy") != fun.end())
   {
      // integral of entropy over the entire volume domain
      output.emplace("entropy", fes.get());
      output.at("entropy").AddDomainIntegrator(
         new EntropyIntegrator<dim, entvar>(diff_stack));
   }
}

template <int dim, bool entvar>
double EulerSolver<dim, entvar>::calcStepSize(int iter, double t,
                                              double t_final,
                                              double dt_old) const
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm();
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return max(dt, dt_old);
   }
   if (!options["time-dis"]["const-cfl"].get<bool>())
   {
      return options["time-dis"]["dt"].get<double>();
   }
   // Otherwise, use a constant CFL condition
   double cfl = options["time-dis"]["cfl"].get<double>();
   Vector q(dim+2);
   auto calcSpect = [&q](const double* dir, const double* u)
   {
      if (entvar)
      {
         calcConservativeVars<double, dim>(u, q);
         return calcSpectralRadius<double, dim>(dir, q);
      }
      else
      {
         return calcSpectralRadius<double, dim>(dir, u);
      }   
   };
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

template <int dim, bool entvar>
void EulerSolver<dim, entvar>::getFreeStreamState(mfem::Vector &q_ref) 
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

template <int dim, bool entvar>
double EulerSolver<dim, entvar>::calcConservativeVarsL2Error(
   void (*u_exact)(const mfem::Vector &, mfem::Vector &), int entry)
{
   // This lambda function computes the error at a node
   // Beware: this is not particularly efficient, given the conditionals
   // Also **NOT thread safe!**
   Vector qdiscrete(dim+2), qexact(dim+2); // define here to avoid reallocation
   auto node_error = [&](const Vector &discrete, const Vector &exact) -> double
   {
      if (entvar)
      {
         calcConservativeVars<double, dim>(discrete.GetData(),
                                           qdiscrete.GetData());
         calcConservativeVars<double, dim>(exact.GetData(), qexact.GetData());
      }
      else
      {
         qdiscrete = discrete;
         qexact = exact;
      }
      double err = 0.0;
      if (entry < 0)
      {
         for (int i = 0; i < dim+2; ++i)
         {
            double dq = qdiscrete(i) - qexact(i);
            err += dq*dq;
         }
      }
      else
      {
         err = qdiscrete(entry) - qexact(entry);
         err = err*err;  
      }
      return err;
   };

   VectorFunctionCoefficient exsol(num_state, u_exact);
   DenseMatrix vals, exact_vals;
   Vector u_j, exsol_j;
   double loc_norm = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule *ir = &(fe->GetNodes());
      ElementTransformation *T = fes->GetElementTransformation(i);
      u->GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         vals.GetColumnReference(j, u_j);
         exact_vals.GetColumnReference(j, exsol_j);
         loc_norm += ip.weight * T->Weight() * node_error(u_j, exsol_j);
      }
   }
   double norm;
#ifdef MFEM_USE_MPI
   MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
#else
   norm = loc_norm;
#endif
   if (norm < 0.0) // This was copied from mfem...should not happen for us
   {
      return -sqrt(-norm);
   }
   return sqrt(norm);
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::convertToEntvar(mfem::Vector &state)
{
   if (entvar)
   {
      return ;
   }
   else
   {
      int num_nodes, offset;
      Array<int> vdofs(num_state);
      Vector el_con, el_ent;
      const FiniteElement *fe;
      for (int i = 0; i < fes->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         num_nodes = fe->GetDof();
         for (int j = 0; j < num_nodes; j++)
         {
            offset = i * num_nodes * num_state + j * num_state;
            for (int k = 0; k < num_state; k++)
            {
               vdofs[k] = offset + k;
            }
            u->GetSubVector(vdofs, el_con);
            calcEntropyVars<double, dim>(el_con.GetData(), el_ent.GetData());
            state.SetSubVector(vdofs, el_ent);
         }
      }
   }
}

template<int dim, bool entvar>
double EulerSolver<dim, entvar>::getParamSens()
{
   double sens = 0;

   // compute the adjoint 
	adj = NULL;
   string drags = "drag";
	solveForAdjoint(drags);
	
   j_mesh_sens.reset(new NonlinearFormType(fes.get()));

	// start by adding/computing the output partial
	/// NOTE: Should eventually support different outputs
	auto &fun = options["outputs"];
   int idx = 0;
   if (fun.find("drag") != fun.end())
   { 
      // drag on the specified boundaries
      vector<int> tmp = fun["drag"].template get<vector<int>>();
      output_bndry_marker[idx].SetSize(tmp.size(), 0);
      output_bndry_marker[idx].Assign(tmp.data());
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
      drag_dir *= 1.0/pow(mach_fs, 2.0); // to get non-dimensional Cd
      j_mesh_sens->AddBdrFaceIntegrator(
          new PressureForceDiff<dim, entvar>(diff_stack, *u, *adj,
                        drag_dir, mach_fs, aoa_fs), output_bndry_marker[idx]);
      idx++;
   }
   // if (fun.find("lift") != fun.end())
   // { 
   //    // lift on the specified boundaries
   //    vector<int> tmp = fun["lift"].template get<vector<int>>();
   //    output_bndry_marker[idx].SetSize(tmp.size(), 0);
   //    output_bndry_marker[idx].Assign(tmp.data());
   //    mfem::Vector lift_dir(dim);
   //    lift_dir = 0.0;
   //    if (dim == 1)
   //    {
   //       lift_dir(0) = 0.0;
   //    }
   //    else
   //    {
   //       lift_dir(iroll) = -sin(aoa_fs);
   //       lift_dir(ipitch) = cos(aoa_fs);
   //    }
   //    lift_dir *= 1.0/pow(mach_fs, 2.0); // to get non-dimensional Cl
   //    j_mesh_sens.AddBdrFaceIntegrator(
   //        new mach::PressureForceDiff<dim, entvar>(diff_stack, u, adj,
   //                      lift_dir, mach_fs, aoa_fs), output_bndry_marker[idx]);
   //    idx++;
   // }
	sens += j_mesh_sens->GetEnergy(*u);

	// get residual sensitivities
	res_mesh_sens.reset(new NonlinearFormType(fes.get()));
   res_mesh_sens_l.reset(new LinearFormType(fes.get()));

	/// add integrators R = [M + (dt/2)K]dudt + Ku + b = 0 (only need FarFieldBC)
	auto &bcs = options["bcs"];
	bndry_marker.resize(bcs.size());
	idx = 0;
   mfem::Vector qfar(dim+2); //mfem::Vector w_far(4);
   getFreeStreamState(qfar);
	if (bcs.find("far-field") != bcs.end())
	{ 
 	    vector<int> tmp = bcs["far-field"].get<vector<int>>();
 	    bndry_marker[idx].SetSize(tmp.size(), 0);
 	    bndry_marker[idx].Assign(tmp.data());
 	    res_mesh_sens_l->AddBdrFaceIntegrator(
			new FarFieldBCDiff<dim, entvar>(diff_stack, *u, *adj,
                        qfar, mach_fs, aoa_fs), bndry_marker[idx]);
 	    idx++;
	}
	/// Compute the derivatives and accumulate the result
	res_mesh_sens_l->Assemble();
	sens -= *adj * *res_mesh_sens_l;

	return sens;
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::verifyParamSens()
{
	std::cout << "Verifying Drag Sensitivity to Mach Number..." << std::endl;
	double delta = 1e-7;
	double delta_cd = 1e-5;
   string drags = "drag";
	double dJdX_fd_v = -calcOutput(drags)/delta;
	double dJdX_cd_v;
	double dJdX_a = getParamSens();
    // extract mesh nodes and get their finite-element space

   // compute finite difference approximation
   mach_fs += delta;
   std::cout << "Solving Forward Step..." << std::endl;
   constructMesh(nullptr);
	initDerived();
   constructLinearSolver(options["lin-solver"]);
	constructNewtonSolver();
	constructEvolver();
   HYPRE_ClearAllErrors();
   Vector qfar(4);
   getFreeStreamState(qfar);
	setInitialCondition(qfar);
   solveForState();
   std::cout << "Solver Done" << std::endl;
   dJdX_fd_v += calcOutput(drags)/delta;

   std::cout << "Mach Number Sensitivity (FD Only):  " << std::endl;
    std::cout << "Finite Difference:  " << dJdX_fd_v << std::endl;
    std::cout << "Analytic: 		  " << dJdX_a << std::endl;
	std::cout << "FD Relative: 		  " << (dJdX_a-dJdX_fd_v)/dJdX_a << std::endl;
    std::cout << "FD Absolute: 		  " << dJdX_a - dJdX_fd_v << std::endl;

	//central difference approximation
	std::cout << "Solving CD Backward Step..." << std::endl;
	mach_fs -= delta; mach_fs -= delta_cd;
	constructMesh(nullptr);
	initDerived();
   constructLinearSolver(options["lin-solver"]);
	constructNewtonSolver();
	constructEvolver();
   HYPRE_ClearAllErrors();
   getFreeStreamState(qfar);
	setInitialCondition(qfar);
   solveForState();
   std::cout << "Solver Done" << std::endl;
   dJdX_cd_v = -calcOutput(drags)/(2*delta_cd);

	std::cout << "Solving CD Forward Step..." << std::endl;
   mach_fs += 2*delta_cd;
	constructMesh(nullptr);
	initDerived();
   constructLinearSolver(options["lin-solver"]);
	constructNewtonSolver();
	constructEvolver();
   HYPRE_ClearAllErrors();
   getFreeStreamState(qfar);
	setInitialCondition(qfar);
   solveForState();
   std::cout << "Solver Done" << std::endl;
   dJdX_cd_v += calcOutput(drags)/(2*delta_cd);

	std::cout << "Mach Number Sensitivity:  " << std::endl;
    std::cout << "Finite Difference:  " << dJdX_fd_v << std::endl;
	std::cout << "Central Difference: " << dJdX_cd_v << std::endl;
    std::cout << "Analytic: 		  " << dJdX_a << std::endl;
	std::cout << "FD Relative: 		  " << (dJdX_a-dJdX_fd_v)/dJdX_a << std::endl;
    std::cout << "FD Absolute: 		  " << dJdX_a - dJdX_fd_v << std::endl;
	std::cout << "CD Relative: 		  " << (dJdX_a-dJdX_cd_v)/dJdX_a << std::endl;
    std::cout << "CD Absolute: 		  " << dJdX_a - dJdX_cd_v << std::endl;
}

template<int dim, bool entvar>
void EulerSolver<dim, entvar>::calcStatistics()
{
   std::string type = options["statistics"]["type"].get<std::string>();
  
   //get uncertain parameters
   auto &params = options["statistics"]["param"];
   auto &outputs = options["outputs"];
   double pmean; double pstdv;
   double mean; double stdev; int order;
   if (params.find("mach") != params.end())
   {
      auto tmp = params["mach"].get<vector<double>>();
      pmean = tmp[0]; pstdv = tmp[1];
   }
   Vector qfar(dim+2);

   //get scheme
   if (type == "collocation")
   {
      order = options["statistics"]["order"].get<int>();
      Vector abs(order); Vector wt(order); Vector pts(order); Vector eval(order); Vector meansq(order);
      switch (order)
      {
         case 1:
            abs = gho1_x; wt = gho1_w;
            break;
         case 2:
            abs = gho2_x; wt = gho2_w;
            break;
         case 3:
            abs = gho3_x; wt = gho3_w;
            break;
         case 4:
            abs = gho4_x; wt = gho4_w;
            break;
         case 5:
            abs = gho5_x; wt = gho5_w;
            break;
         case 6:
            abs = gho6_x; wt = gho6_w;
            break;
         default:
            mfem_error("Gauss-Hermite collocation is currently only supported for 1 <= order <= 6");
            break;
      }

      //compute realizations (1D)
      for(int i = 0; i < order; i++)
      {
         if (params.find("mach") != params.end())
         {
            mach_fs = pmean + sqrt(2.0)*pstdv*abs(i);
         }

	      initDerived();
         constructLinearSolver(options["lin-solver"]);
	      constructNewtonSolver();
	      constructEvolver();
         HYPRE_ClearAllErrors();
         getFreeStreamState(qfar);
	      setInitialCondition(qfar);
         solveForState();
         std::cout << "Solver Done" << std::endl;
         if (outputs.find("drag") != outputs.end())
         {
            string drags = "drag";
            eval(i) = calcOutput(drags);
         }
         meansq(i) = eval(i)*eval(i);
      }

      //print realizations
      stringstream evalname;
      evalname << "realization_outs_"<<pmean<<"_"<<pstdv<<"_o"<<order<<".txt";
      std::ofstream evalfile(evalname.str());
      evalfile.precision(18);
      eval.Print(evalfile);

      mean = eval*wt;
      stdev = sqrt(meansq*wt - mean*mean);

      cout << "Stochastic Collocation Order "<<order<<endl;
   }
   else if (type == "MM1")
   {
      //compute realization at param mean
      if (params.find("mach") != params.end())
      {
         mach_fs = pmean;
      }
	   initDerived();
      constructLinearSolver(options["lin-solver"]);
	   constructNewtonSolver();
	   constructEvolver();
      HYPRE_ClearAllErrors();
      getFreeStreamState(qfar);
	   setInitialCondition(qfar);
      solveForState();
      std::cout << "Solver Done" << std::endl;
      double d1 = getParamSens();

      //compute mean
      if (outputs.find("drag") != outputs.end())
      {
         string drags = "drag";
         mean = calcOutput(drags);
      }

      //compute standard deviation
      stdev = pstdv*d1; //sqrt(pstdv*d1);?

      cout << "Moment Method Order 1"<<endl;
   }

   cout << "Mean: "<<mean<<endl;
   cout << "Standard Deviation: "<<stdev<<endl;

   //write to file
   stringstream statname;
   statname << "euler_stats_"<<type<<"_"<<pmean<<"_"<<pstdv;
   if(type == "collocation")
      statname << "_o"<<order;
   statname <<".txt";
   std::ofstream statfile(statname.str());
   statfile.precision(18);
   statfile << mean << "\n" << stdev;
}

// explicit instantiation
template class EulerSolver<1, true>;
template class EulerSolver<1, false>;
template class EulerSolver<2, true>;
template class EulerSolver<2, false>;
template class EulerSolver<3, true>;
template class EulerSolver<3, false>;

} // namespace mach
