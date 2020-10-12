#include <fstream>

#include "thermal.hpp"
#include "evolver.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

ThermalSolver::ThermalSolver(const nlohmann::json &options,
                             std::unique_ptr<mfem::Mesh> smesh,
                             MPI_Comm comm)
   : AbstractSolver(options, move(smesh), comm)
{
   int dim = getMesh()->Dimension();
   int order = options["space-dis"]["degree"].get<int>();

   /// Create the H(Div) finite element collection for the representation the
   /// magnetic flux density field in the thermal solver
   h_div_coll.reset(new RT_FECollection(order, dim));
   /// Create the H(Div) finite element space
   h_div_space.reset(new SpaceType(mesh.get(), h_div_coll.get()));
   /// Create magnetic flux grid function
   mag_field.reset(new GridFunType(h_div_space.get()));
}

void ThermalSolver::initDerived()
{
   AbstractSolver::initDerived();

   mesh->ReorientTetMesh();
   /// Override, only use 1st order H elements
   int fe_order = 1;// options["space-dis"]["degree"].get<int>();

   th_exact.reset(new GridFunType(fes.get()));

   *out << "Number of finite element unknowns: "
      << fes->GlobalTrueVSize() << endl;

   *out << "Constructing Material Coefficients..." << std::endl;

   int dim = getMesh()->Dimension();
   int order = options["space-dis"]["degree"].get<int>();

   /// Create the H(Div) finite element collection for the representation the
   /// magnetic flux density field in the thermal solver
   h_div_coll.reset(new RT_FECollection(order, dim));
   /// Create the H(Div) finite element space
   h_div_space.reset(new SpaceType(mesh.get(), h_div_coll.get()));
   /// Create magnetic flux grid function
   mag_field.reset(new GridFunType(h_div_space.get()));
      

   /// TODO: REPLACE WITH DOMAIN BASED TEMPERATURE MAXIMA ARRAY
   rhoa = options["rho-agg"].get<double>();

   /// assemble max temp array
   max.SetSize(fes->GetMesh()->attributes.Size()+1);
   for (auto& component : options["components"])
   {
      double mat_max = component["max-temp"].get<double>();
      int attrib = component["attr"].get<int>();
      max(attrib) = mat_max;
   }

   /// pass through aggregation parameters for functional
   // func.reset(new AggregateIntegrator(fes.get(), rhoa, max));
   // /// pass through aggregation parameters for functional
   // does not include dJdu calculation, need AddOutputs for that
   if (rhoa != 0)
   {
      funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
   }
   else
   {
      funct.reset(new TempIntegrator(fes.get()));
   }
}


std::vector<GridFunType*> ThermalSolver::getFields(void)
{
   return {u.get(), mag_field.get()};
}


void ThermalSolver::addOutputs()
{
   auto &fun = options["outputs"];
   int idx = 0;
   if (fun.find("temp-agg") != fun.end())
   {
      rhoa = options["rho-agg"].template get<double>();
      //double max = options["max-temp"].template get<double>();
      output.emplace("temp-agg", fes.get());
      /// assemble max temp array
      max.SetSize(fes->GetMesh()->attributes.Size()+1);
      for (auto& component : options["components"])
      {
         double mat_max = component["max-temp"].template get<double>();
         int attrib = component["attr"].template get<int>();
         max(attrib) = mat_max;
      }
      
      // call the second constructor of the aggregate integrator
      if(rhoa != 0)
      {
         output.at("temp-agg").AddDomainIntegrator(
         new AggregateIntegrator(fes.get(), rhoa, max, u.get()));
      }
      else
      {
         auto &bcs = options["bcs"];
         int idx = 0;
         bndry_marker.resize(bcs.size());
         if (bcs.find("outflux") != bcs.end())
         { // outward flux bc
            vector<int> tmp = bcs["outflux"].get<vector<int>>();
            bndry_marker[idx].SetSize(tmp.size(), 0);
            bndry_marker[idx].Assign(tmp.data());
            output.at("temp-agg").AddBdrFaceIntegrator(
               new TempIntegrator(fes.get(), u.get()), bndry_marker[idx]);
            idx++;
         }
         //output.at("temp-agg").AddDomainIntegrator(
         //new TempIntegrator(fes.get(), u.get()));
      }
         idx++; 
   }
}

void ThermalSolver::solveUnsteady(ParGridFunction &state)
{
   double t = 0.0;
   double agg;
   double gerror = 0;
   evolver->SetTime(t);
   ode_solver->Init(*evolver);

   int precision = 8;
   {
      ofstream osol("motor_heat_init.gf");
      osol.precision(precision);
      u->Save(osol);
   }
   // {
   //     ofstream sol_ofs("motor_heat_init.vtk");
   //     sol_ofs.precision(14);
   //     mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
   //     u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
   //     sol_ofs.close();
   // }

   bool done = false;
   double t_final = options["time-dis"]["t-final"].get<double>();
   double dt = options["time-dis"]["dt"].get<double>();

   // compute functional for first step, testing purposes
   if (rhoa != 0)
   {
      agg = funca->GetIEAggregate(u.get());

      cout << "aggregated temp constraint = " << agg << endl;

   // 	compare to actual max, ASSUMING UNIFORM CONSTRAINT
   // 	gerror = (u->Max()/max(1) - agg)/(u->Max()/max(1));
      
   }
   else
   {
      agg = funct->GetTemp(u.get());
   }

   for (int ti = 0; !done;)
   {
      // if (options["time-dis"]["const-cfl"].get<bool>())
      // {
      //     dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
      // }
      double dt_real = min(dt, t_final - t);
      dt_real_ = dt_real;
      //if (ti % 100 == 0)
      {
         cout << "iter " << ti << ": time = " << t << ": dt = " << dt_real
            << " (" << round(100 * t / t_final) << "% complete)" << endl;
      }
      HypreParVector *TV = u->GetTrueDofs();
      ode_solver->Step(*TV, t, dt_real);
      *u = *TV;

      // compute functional
      if (rhoa != 0)
      {
         agg = funca->GetIEAggregate(u.get());
         cout << "aggregated temp constraint = " << agg << endl;
      }
      else
      {
         agg = funct->GetTemp(u.get());
      }

      // evolver->updateParameters();

      ti++;

      done = (t >= t_final - 1e-8 * dt);
   }

   if (rhoa != 0)
   {
      cout << "aggregated constraint error at initial state = " << gerror << endl;
   }

   {
      ofstream osol("motor_heat.gf");
      osol.precision(precision);
      u->Save(osol);
   }
   
      
   // sol_ofs.precision(14);
   // mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
   // u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
}

// void ThermalSolver::setStaticMembers()
// {
// 	temp_0 = options["init-temp"].get<double>();
// }

void ThermalSolver::constructDensityCoeff()
{
   rho.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      std::unique_ptr<mfem::Coefficient> rho_coeff;
      std::string material = component["material"].get<std::string>();
      std::cout << material << '\n';
      {
         auto rho_val = materials[material]["rho"].get<double>();
         rho_coeff.reset(new ConstantCoefficient(rho_val));
      }
      // int attrib = component["attr"].get<int>();
      rho->addCoefficient(component["attr"].get<int>(), move(rho_coeff));
   }
}

void ThermalSolver::constructHeatCoeff()
{
   cv.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      std::unique_ptr<mfem::Coefficient> cv_coeff;
      std::string material = component["material"].get<std::string>();
      std::cout << material << '\n';
      {
         auto cv_val = materials[material]["cv"].get<double>();
         cv_coeff.reset(new ConstantCoefficient(cv_val));
      }
      cv->addCoefficient(component["attr"].get<int>(), move(cv_coeff));
   }
}

void ThermalSolver::constructMassCoeff()
{
   rho_cv.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      int attr = component.value("attr", -1);

      std::string material = component["material"].get<std::string>();
      auto cv_val = materials[material]["cv"].get<double>();
      auto rho_val = materials[material]["rho"].get<double>();

      if (-1 != attr)
      {
         std::unique_ptr<mfem::Coefficient> temp_coeff;
         temp_coeff.reset(new ConstantCoefficient(cv_val*rho_val));
         rho_cv->addCoefficient(attr, move(temp_coeff));
      }
      else
      {
         auto attrs = component["attrs"].get<std::vector<int>>();
         for (auto& attribute : attrs)
         {
            std::unique_ptr<mfem::Coefficient> temp_coeff;
            temp_coeff.reset(new ConstantCoefficient(cv_val*rho_val));
            rho_cv->addCoefficient(attribute, move(temp_coeff));
         }
      }
   }
}

void ThermalSolver::constructConductivity()
{
   kappa.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      int attr = component.value("attr", -1);

      std::string material = component["material"].get<std::string>();
      auto kappa_val = materials[material]["kappa"].get<double>();

      if (-1 != attr)
      {
         std::unique_ptr<mfem::Coefficient> temp_coeff;
         temp_coeff.reset(new ConstantCoefficient(kappa_val));
         kappa->addCoefficient(attr, move(temp_coeff));
      }
      else
      {
         auto attrs = component["attrs"].get<std::vector<int>>();
         for (auto& attribute : attrs)
         {
            std::unique_ptr<mfem::Coefficient> temp_coeff;
            temp_coeff.reset(new ConstantCoefficient(kappa_val));
            kappa->addCoefficient(attribute, move(temp_coeff));
         }
      }
   }
}

void ThermalSolver::constructJoule()
{
   i2sigmainv.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      int attr = component.value("attr", -1);

      std::string material = component["material"].get<std::string>();

      /// todo use grid function?
      auto current = options["motor-opts"]["current"].get<double>();

      double sigma = materials[material].value("sigma", 0.0);

      if (-1 != attr)
      {
         if (sigma > 1e-12)
         {
            std::unique_ptr<mfem::Coefficient> temp_coeff;
            temp_coeff.reset(new ConstantCoefficient(current*current/sigma));
            i2sigmainv->addCoefficient(attr, move(temp_coeff));
         }
      }
      else
      {
         auto attrs = component["attrs"].get<std::vector<int>>();
         for (auto& attribute : attrs)
         {
            if (sigma > 1e-12)
            {
               std::unique_ptr<mfem::Coefficient> temp_coeff;
               temp_coeff.reset(new ConstantCoefficient(current*current/sigma));
               i2sigmainv->addCoefficient(attribute, move(temp_coeff));
            }
         }
      }
   }
}

void ThermalSolver::constructCore()
{
   coreloss.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      std::string material = component["material"].get<std::string>();

      /// check for each of these values --- if they do not exist they take
      /// the value of zero
      double rho_val = materials[material].value("rho", 0.0);
      double alpha = materials[material].value("alpha", 0.0);
      double freq = options["motor-opts"].value("frequency", 0.0);
      double kh = materials[material].value("kh", 0.0);
      double ke = materials[material].value("ke", 0.0);

      /// make sure that there is a coefficient
      double params = rho_val + alpha + freq + kh + ke;

      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         if (params > 1e-12)
         {
            std::unique_ptr<mfem::Coefficient> temp_coeff;
            // temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
            // 														kh, ke, mag_field));
            temp_coeff.reset(new ConstantCoefficient(0.0));
            coreloss->addCoefficient(attr, move(temp_coeff));		
         }
      }
      else
      {
         auto attrs = component["attrs"].get<std::vector<int>>();
         for (auto& attribute : attrs)
         {
            if (params > 1e-12)
            {
               std::unique_ptr<mfem::Coefficient> temp_coeff;
               // temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
               // 														kh, ke, mag_field));
               temp_coeff.reset(new ConstantCoefficient(0.0));
               coreloss->addCoefficient(attribute, move(temp_coeff));
            }
         }
      }
   }
}

// double ThermalSolver::calcL2Error(
//     double (*u_exact)(const Vector &), int entry)
// {
// 	// TODO: need to generalize to parallel
// 	FunctionCoefficient exsol(u_exact);
// 	th_exact->ProjectCoefficient(exsol);

	
// 	// sol_ofs.precision(14);
// 	// th_exact->SaveVTK(sol_ofs, "Analytic", options["space-dis"]["degree"].get<int>() + 1);
// 	// sol_ofs.close();

// 	return u->ComputeL2Error(exsol);
// }

void ThermalSolver::solveUnsteadyAdjoint(const std::string &fun)
{
   // only solve for state at end time
   double time_beg, time_end;
   if (0==rank)
   {
      time_beg = MPI_Wtime();
   }

   // add the dJdu output, do this now to precompute max temperature and 
   // certain values for the functional so that we don't need it at every call
   addOutputs();

   // Step 0: allocate the adjoint variable
   adj.reset(new GridFunType(fes.get()));

   // Step 1: get the right-hand side vector, dJdu, and make an appropriate
   // alias to it, the state, and the adjoint
   std::unique_ptr<GridFunType> dJdu(new GridFunType(fes.get()));
   HypreParVector *state = u->GetTrueDofs();
   HypreParVector *dJ = dJdu->GetTrueDofs();
   HypreParVector *adjoint = adj->GetTrueDofs();
   double energy = output.at(fun).GetEnergy(*u);
   output.at(fun).Mult(*state, *dJ);
   cout << "Last Functional Output: " << energy << endl;

   // // Step 2: get the last time step's Jacobian
   // HypreParMatrix *jac = evolver->GetOperator();
   // //TransposeOperator jac_trans = TransposeOperator(jac);
   // HypreParMatrix *jac_trans = jac->Transpose();

   // Step 2: get the last time step's Jacobian and transpose it
   Operator *jac = &evolver->GetGradient(*state);
   TransposeOperator jac_trans = TransposeOperator(jac);

   // Step 3: Solve the adjoint problem
   *out << "Solving adjoint problem:\n"
         << "\tsolver: HypreGMRES\n"
         << "\tprec. : Euclid ILU" << endl;
   prec.reset(new HypreEuclid(fes->GetComm()));
   double tol = options["adj-solver"]["rel-tol"].get<double>();
   int maxiter = options["adj-solver"]["max-iter"].get<int>();
   int ptl = options["adj-solver"]["print-lvl"].get<int>();
   solver.reset(new HypreGMRES(fes->GetComm()));
   solver->SetOperator(jac_trans);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetTol(tol);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetMaxIter(maxiter);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPrintLevel(ptl);
   dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver *>(prec.get()));
   solver->Mult(*dJ, *adjoint);
   adjoint->Set(dt_real_, *adjoint);
   adj->SetFromTrueDofs(*adjoint);
   if (0==rank)
   {
      time_end = MPI_Wtime();
      cout << "Time for solving adjoint is " << (time_end - time_beg) << endl;
   }

   {
      ofstream sol_ofs_adj("motor_heat_adj.vtk");
      sol_ofs_adj.precision(14);
      mesh->PrintVTK(sol_ofs_adj, options["space-dis"]["degree"].get<int>());
      adj->SaveVTK(sol_ofs_adj, "Adjoint", options["space-dis"]["degree"].get<int>());
      sol_ofs_adj.close();
   }
}

void ThermalSolver::constructCoefficients()
{
   // constructDensityCoeff();
   constructMassCoeff();
   // constructHeatCoeff();
   constructConductivity();
   constructJoule();
   constructCore();
}

void ThermalSolver::constructForms()
{
   mass.reset(new BilinearFormType(fes.get()));
   res.reset(new ParNonlinearForm(fes.get()));
   load.reset(new LinearFormType(fes.get()));
}

void ThermalSolver::addMassIntegrators(double alpha)
{
   mass->AddDomainIntegrator(new MassIntegrator(*rho_cv));
}

void ThermalSolver::addResVolumeIntegrators(double alpha)
{
   res->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
}

void ThermalSolver::addLoadVolumeIntegrators(double alpha)
{
   auto load_lf = dynamic_cast<ParLinearForm*>(load.get());
   /// add joule heating term
   load_lf->AddDomainIntegrator(new DomainLFIntegrator(*i2sigmainv));
   /// add iron loss heating terms
   load_lf->AddDomainIntegrator(new DomainLFIntegrator(*coreloss));
}

void ThermalSolver::addLoadBoundaryIntegrators(double alpha)
{
   auto load_lf = dynamic_cast<ParLinearForm*>(load.get());

   //determine type of flux function
   if(options["outflux-type"].template get<string>() == "test")
   {
      flux_coeff.reset(new VectorFunctionCoefficient(3, testFluxFunc));
   }
   else
      throw MachException("Specified flux function not supported!\n");

   auto &bcs = options["bcs"];
   bndry_marker.resize(bcs.size());
   int idx = 0;
   if (bcs.find("outflux") != bcs.end())
   { // outward flux bc
      vector<int> tmp = bcs["outflux"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      load_lf->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(
                                             *flux_coeff), bndry_marker[idx]);
      idx++;
   }
}

void ThermalSolver::constructEvolver()
{
   Array<int> ess_bdr;
   auto &bcs = options["bcs"];
   /// if any boundaries are marked as essential in the options file use that
   if (bcs.find("essential") != bcs.end())
   {
      auto tmp = bcs["essential"].get<vector<int>>();
      ess_bdr.SetSize(tmp.size(), 0);
      ess_bdr.Assign(tmp.data());
   }
   /// otherwise mark all attributes as nonessential
   else
   {
      ess_bdr.SetSize(mesh->bdr_attributes.Max());
      ess_bdr = 0;
   }

   evolver.reset(new ThermalEvolver(ess_bdr, mass.get(), res.get(), load.get(), *out,
                                    0.0, flux_coeff.get()));
   evolver->SetLinearSolver(solver.get());
   //if (newton_solver == nullptr)
   //   constructNewtonSolver();
   evolver->SetNewtonSolver(newton_solver.get());
}

void ThermalSolver::testFluxFunc(const Vector &x, double time, Vector &y)
{
   y.SetSize(3);
   //use constant in time for now

   //assuming centered coordinate system, will offset
   // double th;// = atan(x(1)/x(0));

   if (x(0) > .5)
   {
      y(0) = 1;
   }
   else
   {
      y(0) = -(M_PI/2)*exp(-M_PI*M_PI*time/4);
   }

   y(1) = 0;
   y(2) = 0;
   
}

ThermalEvolver::ThermalEvolver(Array<int> ess_bdr, BilinearFormType *mass,
                               ParNonlinearForm *res,
                               Vector *load,
                               std::ostream &outstream,
                               double start_time,
                               mfem::VectorCoefficient *_flux_coeff)
   : MachEvolver(ess_bdr, nullptr, mass, res, nullptr, load, nullptr,
                 outstream, start_time),
   flux_coeff(_flux_coeff), work(height)
{ };

void ThermalEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   flux_coeff->SetTime(t);
   LinearFormType *load_lf = dynamic_cast<LinearFormType*>(load);
   if (load_lf)
      load_lf->Assemble();
   else
      throw MachException("Couldn't cast load to LinearFormType!\n");
   work.SetSize(x.Size());
   res->Mult(x, work);
   work += *load;
   mass_solver.Mult(work, y);
   y *= -1.0;
}

void ThermalEvolver::ImplicitSolve(const double dt, const Vector &x,
                                   Vector &k)
{
   /// re-assemble time dependent load vector
   flux_coeff->SetTime(t);
   // dynamic_cast<LinearFormType*>(load)->Assemble();
   LinearFormType *load_lf = dynamic_cast<LinearFormType*>(load);
   if (load_lf)
      load_lf->Assemble();
   else
      throw MachException("Couldn't cast load to LinearFormType!\n");

   setOperParameters(dt, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver

   // set iterative mode to false so k is only zeroed once, but set back after
   auto iter_mode = newton->iterative_mode;
   newton->iterative_mode = false;
   newton->Mult(zero, k);
   newton->iterative_mode = iter_mode;
   MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
}

} // namespace mach
