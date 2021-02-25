#include <fstream>
#include <unordered_set>

#include "mfem.hpp"

#include "evolver.hpp"
#include "thermal.hpp"
#include "mach_load.hpp"
#include "mach_linearform.hpp"

using namespace std;
using namespace mfem;

namespace
{

void test_flux_func(const Vector &x, double time, Vector &y)
{
   y.SetSize(3);

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

} // anonymous namespace

namespace mach
{

ThermalSolver::ThermalSolver(const nlohmann::json &options,
                             std::unique_ptr<mfem::Mesh> smesh,
                             MPI_Comm comm)
   : AbstractSolver(options, move(smesh), comm)
{
	// mesh->ReorientTetMesh();   
}

void ThermalSolver::initDerived()
{
   AbstractSolver::initDerived();
}

ThermalSolver::~ThermalSolver() = default;

std::vector<GridFunType*> ThermalSolver::getFields(void)
{
   return {u.get()};
}


void ThermalSolver::addOutputs()
{
   auto &fun = options["outputs"];
   int idx = 0;
   if (fun.find("new-agg") != fun.end())
   {
      fractional_output.emplace("new-agg",
         std::initializer_list<std::string>{"agg-num", "agg_denom"});

      output.emplace("agg-num", fes.get());
      output.emplace("agg-denom", fes.get());

      /// assemble max temp array
      Vector max(fes->GetMesh()->attributes.Size());
      double default_max = options["problem-opts"].value("max-temp", 1e6);
      for (auto& component : options["components"])
      {
         auto material = component["material"].get<std::string>();
         auto mat_max = materials[material].value("max-temp", default_max);

         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            max[attr - 1] = mat_max;
         }
         else
         {
            auto attrs = component["attrs"].get<std::vector<int>>();
            for (auto& attribute : attrs)
            {
               max[attribute - 1] = mat_max;
            }
         }
         
      }
      
      /// use rho = 10 for a default if rho not given in options
      double rhoa = options["problem-opts"].value("rho-agg", 10.0);
      output.at("agg-num").AddDomainIntegrator(
         new AggregateIntegratorNumerator(rhoa, max));

      output.at("agg-denom").AddDomainIntegrator(
         new AggregateIntegratorDenominator(rhoa, max));

   }
   if (fun.find("agg") != fun.end())
   {
      output.emplace("agg", fes.get());
      
      /// assemble max temp array
      Vector max(fes->GetMesh()->attributes.Size());
      double default_max = options["problem-opts"].value("max-temp", 1e6);
      for (auto& component : options["components"])
      {
         auto material = component["material"].get<std::string>();
         auto mat_max = materials[material].value("max-temp", default_max);

         int attr = component.value("attr", -1);
         if (-1 != attr)
         {
            max[attr - 1] = mat_max;
         }
         else
         {
            auto attrs = component["attrs"].get<std::vector<int>>();
            for (auto& attribute : attrs)
            {
               max[attribute - 1] = mat_max;
            }
         }
         
      }
      
      /// use rho = 10 for a default if rho not given
      double rhoa = options["problem-opts"].value("rho-agg", 10.0);
      output.at("agg").AddDomainIntegrator(
         new AggregateIntegrator(fes.get(), rhoa, max, u.get()));
   }
   if (fun.find("temp") != fun.end())
   {
      output.emplace("temp", fes.get());

      auto &bcs = options["bcs"];
      int idx = 0;
      bndry_marker.resize(bcs.size());
      if (bcs.find("outflux") != bcs.end())
      { // outward flux bc
         vector<int> tmp = bcs["outflux"].get<vector<int>>();
         bndry_marker[idx].SetSize(tmp.size(), 0);
         bndry_marker[idx].Assign(tmp.data());
         output.at("temp").AddBdrFaceIntegrator(
            new TempIntegrator(fes.get(), u.get()), bndry_marker[idx]);
         idx++;
      }
      //output.at("temp-agg").AddDomainIntegrator(
      //new TempIntegrator(fes.get(), u.get()));
   }
   idx++; 
}

void ThermalSolver::initialHook(const ParGridFunction &state) 
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // res_norm0 is used to compute the time step in PTC
      res_norm0 = calcResidualNorm(state);
   }
}

bool ThermalSolver::iterationExit(int iter, double t, double t_final,
                                  double dt,
                                  const ParGridFunction &state) const
{
   if (options["time-dis"]["steady"].get<bool>())
   {
      // use tolerance options for Newton's method
      double norm = calcResidualNorm(state);
      std::cout << "res norm: " << norm << "\n";
      if (norm <= options["time-dis"]["steady-abstol"].get<double>())
         return true;
      if (norm <= res_norm0 *
                      options["time-dis"]["steady-reltol"].get<double>())
         return true;
      return false;
   }
   else
   {
      return AbstractSolver::iterationExit(iter, t, t_final, dt, state);
   }
}

double ThermalSolver::calcStepSize(int iter, 
                                   double t,
                                   double t_final,
                                   double dt_old,
                                   const ParGridFunction &state) const
{
   if (options["time-dis"]["steady"].template get<bool>())
   {
      // ramp up time step for pseudo-transient continuation
      // TODO: the l2 norm of the weak residual is probably not ideal here
      // A better choice might be the l1 norm
      double res_norm = calcResidualNorm(state);
      if (std::abs(res_norm) <= 1e-14) return 1e14;
      double exponent = options["time-dis"]["res-exp"];
      double dt = options["time-dis"]["dt"].template get<double>() *
                  pow(res_norm0 / res_norm, exponent);
      return max(dt, dt_old);
   }
   else
      return AbstractSolver::calcStepSize(iter, t, t_final, dt_old, state);
}

void ThermalSolver::terminalHook(int iter, double t_final,
                                 const ParGridFunction &state)
{
   auto *state_gf = const_cast<ParGridFunction*>(&state);

   //#GradientGridFunctionCoefficient flux_coeff(state_gf);
   //#ParFiniteElementSpace h1_fes(mesh.get(), fec.get(), mesh->Dimension());
   //#ParGridFunction heat_flux(&h1_fes);
   //#heat_flux.ProjectCoefficient(flux_coeff);


   printField("therm_state", *state_gf, "theta");

}

void ThermalSolver::solveUnsteady(ParGridFunction &state)
{
   {
      // auto stein_field = getNewField();
      // stein_field->ProjectCoefficient(*coreloss);
      // auto i2r_field = getNewField();
      // i2r_field->ProjectCoefficient(*i2sigmainv);
      // printFields("loses",
      //             {stein_field.get(), i2r_field.get()},
      //             {"steinmetz", "i2r"});
   }

   AbstractSolver::solveUnsteady(state);
   // double t = 0.0;
   // double agg;
   // double gerror = 0;
   // evolver->SetTime(t);
   // ode_solver->Init(*evolver);

   // int precision = 8;
   // {
   //    ofstream osol("motor_heat_init.gf");
   //    osol.precision(precision);
   //    u->Save(osol);
   // }
   // // {
   // //     ofstream sol_ofs("motor_heat_init.vtk");
   // //     sol_ofs.precision(14);
   // //     mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
   // //     u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
   // //     sol_ofs.close();
   // // }

   // bool done = false;
   // double t_final = options["time-dis"]["t-final"].get<double>();
   // double dt = options["time-dis"]["dt"].get<double>();

   // // compute functional for first step, testing purposes
   // // if (rhoa != 0)
   // // {
   // //    agg = funca->GetIEAggregate(u.get());

   // //    cout << "aggregated temp constraint = " << agg << endl;

   // // // 	compare to actual max, ASSUMING UNIFORM CONSTRAINT
   // // // 	gerror = (u->Max()/max(1) - agg)/(u->Max()/max(1));
      
   // // }
   // // else
   // // {
   // //    agg = funct->GetTemp(u.get());
   // // }

   // for (int ti = 0; !done;)
   // {
   //    // if (options["time-dis"]["const-cfl"].get<bool>())
   //    // {
   //    //     dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
   //    // }
   //    double dt_real = min(dt, t_final - t);
   //    dt_real_ = dt_real;
   //    //if (ti % 100 == 0)
   //    {
   //       cout << "iter " << ti << ": time = " << t << ": dt = " << dt_real
   //          << " (" << round(100 * t / t_final) << "% complete)" << endl;
   //    }
   //    HypreParVector *TV = u->GetTrueDofs();
   //    ode_solver->Step(*TV, t, dt_real);
   //    *u = *TV;

   //    // // compute functional
   //    // if (rhoa != 0)
   //    // {
   //    //    agg = funca->GetIEAggregate(u.get());
   //    //    cout << "aggregated temp constraint = " << agg << endl;
   //    // }
   //    // else
   //    // {
   //    //    agg = funct->GetTemp(u.get());
   //    // }

   //    // evolver->updateParameters();

   //    ti++;

   //    done = (t >= t_final - 1e-8 * dt);
   // }

   // // if (rhoa != 0)
   // // {
   // //    cout << "aggregated constraint error at initial state = " << gerror << endl;
   // // }

   // {
   //    ofstream osol("motor_heat.gf");
   //    osol.precision(precision);
   //    u->Save(osol);
   // }
   
      
   // // sol_ofs.precision(14);
   // // mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
   // // u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
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

void ThermalSolver::constructConvection()
{
   auto &bcs = options["bcs"];
   if (bcs.contains("convection"))
   {
      if (options["problem-opts"].contains("convection-coeff"))
      {
         auto h = options["problem-opts"]["convection-coeff"].get<double>();
         convection.reset(new ConstantCoefficient(h));
      }
      else
      {
         throw MachException("Using convection boundary condition without"
                             "specifying heat transfer coefficient!\n");
      }
   }
}

void ThermalSolver::constructJoule()
{
   i2sigmainv.reset(new MeshDependentCoefficient());

   if (options["problem-opts"].contains("current"))
   {
      for (auto& component : options["components"])
      {
         int attr = component.value("attr", -1);

         std::string material = component["material"].get<std::string>();

         auto current = options["problem-opts"]["current-density"].get<double>();
         current *= options["problem-opts"].value("fill-factor", 1.0);

         double sigma = materials[material].value("sigma", 0.0);

         if (-1 != attr)
         {
            if (sigma > 1e-12)
            {
               std::unique_ptr<mfem::Coefficient> temp_coeff;
               temp_coeff.reset(new ConstantCoefficient(-current*current/sigma));
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
                  temp_coeff.reset(new ConstantCoefficient(-current*current/sigma));
                  i2sigmainv->addCoefficient(attribute, move(temp_coeff));
               }
            }
         }
      }
   }
}

void ThermalSolver::constructCore()
{
   /// only construct the coreloss coefficient if the magnetic field is known
   /// to the thermal solver
   if (res_fields.count("mvp") == 0)
   {
      return;
   }
   coreloss.reset(new MeshDependentCoefficient());

   for (auto& component : options["components"])
   {
      std::string material = component["material"].get<std::string>();

      /// check for each of these values --- if they do not exist they take
      /// the value of zero
      double rho_val = materials[material].value("rho", 0.0);
      double alpha = materials[material].value("alpha", 0.0);
      double freq = options["problem-opts"].value("frequency", 0.0);
      // double kh = materials[material].value("kh", 0.0);
      // double ke = materials[material].value("ke", 0.0);
      double ks = materials[material].value("ks", 0.0);
      double beta = materials[material].value("beta", 0.0);

      /// make sure that there is a coefficient
      double params =  alpha + ks + beta;

      int attr = component.value("attr", -1);
      if (-1 != attr)
      {
         if (params > 1e-12)
         {
            std::unique_ptr<mfem::Coefficient> temp_coeff;
            // temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
            //                                           kh, ke,
            //                                           res_fields.at("mvp")));
            temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
                                                      ks, beta,
                                                      res_fields.at("mvp")));
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
               //                                           kh, ke,
               //                                           res_fields.at("mvp")));
               temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
                                                         ks, beta,
                                                         res_fields.at("mvp")));

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
   constructConvection();
   constructJoule();
   constructCore();
}

void ThermalSolver::constructForms()
{
   mass.reset(new BilinearFormType(fes.get()));
   res.reset(new ParNonlinearForm(fes.get()));
   therm_load.reset(new MachLinearForm(fes.get()));
   load.reset(new MachLoad(*therm_load));
}

void ThermalSolver::addMassIntegrators(double alpha)
{
   mass->AddDomainIntegrator(new MassIntegrator(*rho_cv));
}

void ThermalSolver::addResVolumeIntegrators(double alpha)
{
   res->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
}

void ThermalSolver::addResBoundaryIntegrators(double alpha)
{
   if (convection)
   {
      auto &bcs = options["bcs"];
      auto conv_bdr = bcs["convection"].get<std::vector<int>>();

      int source = conv_bdr[0];
      std::unordered_set<int> conv_faces(conv_bdr.begin()+1, conv_bdr.end());

      double ambient_temp = options["problem-opts"]["init-temp"].get<double>();
      // res->AddBdrFaceIntegrator(new InteriorBoundaryOutFluxInteg(*kappa,
      //                                                            *convection,
      //                                                            source,
      //                                                            ambient_temp),
      //                                                            conv_faces);
      res->AddInteriorFaceIntegrator(
         new InteriorBoundaryOutFluxInteg(*kappa,
                                          *convection,
                                          source,
                                          ambient_temp,
                                          conv_faces,
                                          pumi_mesh.get()));
      
   }
}

void ThermalSolver::addLoadVolumeIntegrators(double alpha)
{
   // auto load_lf = dynamic_cast<ParLinearForm*>(load.get());
   /// add joule heating term
   therm_load->addDomainIntegrator(new DomainLFIntegrator(*i2sigmainv));
   /// add iron loss heating terms only if the EM field exists
   if (res_fields.find("mvp") != res_fields.end())
      therm_load->addDomainIntegrator(new DomainLFIntegrator(*coreloss));
}

void ThermalSolver::addLoadBoundaryIntegrators(double alpha)
{
   // auto *load_lf = dynamic_cast<ParLinearForm*>(load.get());
   // if (!load_lf)
   //    throw MachException("Couldn't cast load to LinearFormType!\n");

   /// determine type of flux function
   auto &bcs = options["bcs"];
   if (bcs.contains("outflux"))
   {
      if (options["problem-opts"].contains("outflux-type"))
      {
         if (options["problem-opts"]["outflux-type"].get<string>() == "test")
         {
            int dim = mesh->Dimension();
            flux_coeff.reset(new VectorFunctionCoefficient(dim, test_flux_func));
         }
         else
            throw MachException("Specified flux function not supported!\n");
      }
      else
      {
         throw MachException("Must specify outflux type if using outflux "
                             "boundary conditions!");
      }
      

      constexpr int idx = 0;
      
      // outward flux bc
      vector<int> tmp = bcs["outflux"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      therm_load->addBoundaryIntegrator(new BoundaryNormalLFIntegrator(
                                             *flux_coeff), bndry_marker[idx]);
   }
}

void ThermalSolver::constructEvolver()
{
   evolver.reset(new ThermalEvolver(ess_bdr, mass.get(), res.get(), load.get(),
                                    *out, 0.0, flux_coeff.get()));
   evolver->SetNewtonSolver(newton_solver.get());
}

ThermalEvolver::ThermalEvolver(Array<int> ess_bdr, BilinearFormType *mass,
                               ParNonlinearForm *res,
                               MachLoad *load,
                               std::ostream &outstream,
                               double start_time,
                               VectorCoefficient *_flux_coeff)
   : MachEvolver(ess_bdr, nullptr, mass, res, nullptr, load, nullptr,
                 outstream, start_time),
   flux_coeff(_flux_coeff)
{ };

ThermalEvolver::~ThermalEvolver() = default;

void ThermalEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   // if (flux_coeff)
   // {
   //    flux_coeff->SetTime(t);
   //    auto *load_lf = dynamic_cast<ParLinearForm*>(load);
   //    if (load_lf)
   //       load_lf->Assemble();
   //    else
   //       throw MachException("Couldn't cast load to LinearFormType!\n");
   // }

   MachEvolver::Mult(x, y);
}

void ThermalEvolver::ImplicitSolve(const double dt, const Vector &x,
                                   Vector &k)
{
   // /// re-assemble time dependent load vector
   // if (flux_coeff)
   // {
   //    flux_coeff->SetTime(t);
   //    auto *load_lf = dynamic_cast<ParLinearForm*>(load);
   //    if (load_lf)
   //       load_lf->Assemble();
   //    else
   //       throw MachException("Couldn't cast load to LinearFormType!\n");
   // }

   MachEvolver::ImplicitSolve(dt, x, k);
}

} // namespace mach
