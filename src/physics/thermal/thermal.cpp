#include <fstream>

#include "thermal.hpp"
#include "evolver.hpp"

using namespace std;
using namespace mfem;

// namespace
// {

// void fluxFunc(const Vector &x, double time, Vector &y)
// {
// 	y.SetSize(3);
// 	//use constant in time for now

// 	//assuming centered coordinate system, will offset
// 	// double th;// = atan(x(1)/x(0));

// 	if (x(0) > .5)
// 	{
// 		y(0) = 1;
// 	}
// 	else
// 	{
// 		y(0) = -(M_PI/2)*exp(-M_PI*M_PI*time/4);
// 		//cout << "outflux val = " << y(0) << std::endl;
// 	}

// 	y(1) = 0;
// 	y(2) = 0;
	
// }

// } // anonymous namespace

namespace mach
{

ThermalSolver::ThermalSolver(const nlohmann::json &options,
                             std::unique_ptr<mfem::Mesh> smesh,
                             MPI_Comm comm)
	: AbstractSolver(options, move(smesh), comm)
{
	mesh->EnsureNodes();
}

void ThermalSolver::initDerived()
{
	mesh->RemoveInternalBoundaries();
	AbstractSolver::initDerived();
   // AbstractSolver::initDerived();
	setInit = false;

	mesh->ReorientTetMesh();
	/// Override, only use 1st order H elements
	int fe_order = 1;// options["space-dis"]["degree"].get<int>();

	/// Create temperature grid function
	// u.reset(new GridFunType(fes.get()));
	th_exact.reset(new GridFunType(fes.get()));
	u_old.reset(new GridFunType(fes.get()));


	// /// Set static variables
	// setStaticMembers();

   *out << "Number of finite element unknowns: "
       << fes->GlobalTrueVSize() << endl;

   //  ifstream material_file(options["material-lib-path"].get<string>());
	// /// TODO: replace with mach exception
	// if (!material_file)
	// 	std::cerr << "Could not open materials library file!" << std::endl;
	// material_file >> materials;

	*out << "Constructing Material Coefficients..." << std::endl;

   // int dim = getMesh()->Dimension();
   // int order = options["space-dis"]["degree"].get<int>();
	
	// constructDensityCoeff();

	// constructHeatCoeff();

	// constructMassCoeff();

   // constructConductivity();
     
   // constructJoule();

	// constructCore();

	// std::cout << "Defining Finite Element Spaces..." << std::endl;
	// /// set essential BCs (none)
	// Array<int> ess_tdof_list;
	// mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
	// ess_bdr = 0;
   // fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

	/// set up the bilinear forms
	// m.reset(new BilinearFormType(fes.get()));
	// k.reset(new BilinearFormType(fes.get()));

	std::cout << "Creating Mass Matrix..." << std::endl;
	/// add mass integrator to m bilinear form
	// m->AddDomainIntegrator(new MassIntegrator(*rho_cv));
	/// assemble mass matrix
	// m->Assemble(0);

	// m->FormSystemMatrix(ess_tdof_list, M);

	/// add diffusion integrator to k bilinear form
	// k->AddDomainIntegrator(new DiffusionIntegrator(*kappa));


	/// set up the linear form (volumetric fluxes)
	// bs.reset(new LinearForm(fes.get()));

	/// add joule heating term
	// bs->AddDomainIntegrator(new DomainLFIntegrator(*i2sigmainv));
	// std::cout << "Constructing Boundary Conditions..." << std::endl;
	/// add iron loss heating terms
	// bs->AddDomainIntegrator(new DomainLFIntegrator(*coreloss));


	// std::cout << "Assembling Stiffness Matrix..." << std::endl;
	/// assemble stiffness matrix and linear form
	// k->Assemble(0);

	// k->FormSystemMatrix(ess_tdof_list, K);
	// std::cout << "Assembling Forcing Term..." << std::endl;
	// bs->Assemble();

	// std::cout << "Setting Up ODE Solver..." << std::endl;
	// /// define ode solver
	// ode_solver = NULL;
	// ode_solver.reset(new ImplicitMidpointSolver);
	// std::string ode_opt = 
	// 	options["time-dis"]["ode-solver"].get<std::string>();
	// if (ode_opt == "MIDPOINT")
	// {
	// 	ode_solver = NULL;
	// 	ode_solver.reset(new ImplicitMidpointSolver);
	// }
	// if (ode_opt == "RK4")
	// {
	// 	ode_solver = NULL;
	// 	ode_solver.reset(new RK4Solver);
	// }
 	
	// evolver.reset(new ThermalEvolver(opt_file_name, M, 
										// K, move(bs), *out));

	/// TODO: REPLACE WITH DOMAIN BASED TEMPERATURE MAXIMA ARRAY
	rhoa = options["problem-opts"]["rho-agg"].get<double>();
	//double max = options["max-temp"].get<double>();

	/// assemble max temp array
	max.SetSize(fes->GetMesh()->attributes.Size()+1);
	for (auto& component : options["components"])
	{
		auto  material = component["material"].get<std::string>();
		double mat_max = materials[material]["max-temp"].get<double>();
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
   return {u.get()};
}


void ThermalSolver::addOutputs()
{
	auto &fun = options["outputs"];
    int idx = 0;
	output.clear();
    if (fun.find("temp-agg") != fun.end())
    {
		rhoa = options["problem-opts"]["rho-agg"].template get<double>();
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
			// auto &bcs = options["bcs"];
			// int idx = 0;
			// bndry_marker.resize(bcs.size());
			// if (bcs.find("outflux") != bcs.end())
			// { // outward flux bc
        	// 	vector<int> tmp = bcs["outflux"].get<vector<int>>();
        	// 	bndry_marker[idx].SetSize(tmp.size(), 0);
        	// 	bndry_marker[idx].Assign(tmp.data());
        	// 	output.at("temp-agg").AddBdrFaceIntegrator(
			// 		new TempIntegrator(fes.get(), u.get()), bndry_marker[idx]);
        	// 	idx++;
    		// }
			output.at("temp-agg").AddDomainIntegrator(
			new TempIntegrator(fes.get(), u.get()));
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
	   
	// hold on to the initial state
	*u_old = *u;
	u_init.reset(new GridFunType(*u));
	// if (!setInit)
	// {
	// 	setInitialCondition(initialTemperature);
	// }

	int precision = 8;
	{
		ofstream osol("motor_heat_init.gf");
		osol.precision(precision);
		u->Save(osol);
	}
	{
	    ofstream sol_ofs_init("motor_heat_init.vtk");
	    sol_ofs_init.precision(14);
	    mesh->PrintVTK(sol_ofs_init, options["space-dis"]["degree"].get<int>() + 1);
	    u->SaveVTK(sol_ofs_init, "Solution", options["space-dis"]["degree"].get<int>() + 1);
	    sol_ofs_init.close();
   }
	ParaViewDataCollection *pd = NULL;
   {
      pd = new ParaViewDataCollection("joule_box_time_hist", mesh.get());
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("solution", u.get());
      pd->SetLevelsOfDetail(options["space-dis"]["degree"].get<int>()+1);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

	bool done = false;
	double t_final = options["time-dis"]["t-final"].get<double>();
	double dt = options["time-dis"]["dt"].get<double>();
   int ti_final;

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
		//save the state, if computing mesh sensitivities
		if (options.value("compute-sens", false))
    	{
			stringstream solname;
			solname << "state"<<ti<<".gf";
    	    ofstream ssol(solname.str()); ssol.precision(30);
			u->Save(ssol);
    	}

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
		pd->SetCycle(ti);
		pd->SetTime(t);
		pd->Save();

		// evolver->updateParameters();

		ti++;

		ti_final = ti;
		done = (t >= t_final - 1e-8 * dt);
	}

	if (rhoa != 0)
	{
		cout << "aggregated constraint error at initial state = " << gerror << endl;
	}

	// Save the final solution
	stringstream solname;
	solname << "state"<<ti_final<<".gf";
   ofstream ssol(solname.str()); ssol.precision(30);
	u->Save(ssol);

	
   ofstream sol_ofs("motor_heat.vtk");
	sol_ofs.precision(14);
	mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
	u->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
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
		auto current_density = options["problem-opts"]["current-density"].get<double>();

		double sigma = materials[material].value("sigma", 0.0);

		if (-1 != attr)
		{
			if (sigma > 1e-12)
			{
				std::unique_ptr<mfem::Coefficient> temp_coeff;
				temp_coeff.reset(new ConstantCoefficient(current_density*current_density/sigma));
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
					temp_coeff.reset(new ConstantCoefficient(current_density*current_density/sigma));
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
		double freq = options["problem-opts"].value("frequency", 0.0);
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
				temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
																		kh, ke, a_field));
				// temp_coeff.reset(new ConstantCoefficient(0.0));
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
					temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
																			kh, ke, a_field));
					// temp_coeff.reset(new ConstantCoefficient(0.0));
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
//     double time_beg, time_end;
//     if (0 == rank)
//     {
//        time_beg = MPI_Wtime();
//     }

// 	// we only need dJdu at the last state
// 	if (tr == ti_final)
// 	{
// 		// add the dJdu output, do this now to precompute max temperature and 
// 		// certain values for the functional so that we don't need it at every call
// 		addOutputs();
// 	}

//     // Step 0: allocate the adjoint variable
//     adj.reset(new GridFunType(fes.get()));

// 	// Step 1: get the right-hand side vector, dJdu, and make an appropriate
//    // alias to it, the state, and the adjoint
//    std::unique_ptr<GridFunType> dJdu(new GridFunType(fes.get()));
//    HypreParVector *state = u->GetTrueDofs();
//    HypreParVector *dJ = dJdu->GetTrueDofs();
//    HypreParVector *adjoint = adj->GetTrueDofs();
//    double energy = output.at(fun).GetEnergy(*u);
// 	output.at(fun).Mult(*state, *dJ);
// 	cout << "Last Functional Output: " << energy << endl;

// 	// // Step 2: get the last time step's Jacobian
// 	// HypreParMatrix *jac = evolver->GetOperator();
// 	// //TransposeOperator jac_trans = TransposeOperator(jac);
// 	// HypreParMatrix *jac_trans = jac->Transpose();
// 	HypreParVector *adjoint_old = adj_old->GetTrueDofs();

// 	if (tr == ti_final)
// 	{
// 		output.at(fun).Mult(*state, *dJ);
// 		dJ->Set(dt_real_, *dJ);
// 	}
// 	else //subsequent solves have K^T psi_old as rhs
// 	{
// 		stiff->MultTranspose(*adjoint_old, *dJ);
// 		dJ->Set(-dt_real_, *dJ);
// 		//technically should also Add(dt_real_, djdu), but it's 0 for this problem
// 	}

// 	// Step 2: get the time step's Jacobian and transpose it
//    Operator *jac = &evolver->GetGradient(*state);
//    // cast the ElementTransformation (for the domain element)
// 	 HypreParMatrix &jac_h =
// 	dynamic_cast<HypreParMatrix&>(*jac);
//     HypreParMatrix *jac_trans = jac_h.Transpose();

// 	// Step 3: Solve the adjoint problem
//    *out << "Solving adjoint problem:\n"
//          << "\tsolver: HypreGMRES\n"
//          << "\tprec. : Euclid ILU" << endl;
//    // constructLinearSolver(options["adj-solver"]);
//    //prec.reset(new HypreEuclid(fes->GetComm()));
//    double tol = options["adj-solver"]["rel-tol"].get<double>();
//    int maxiter = options["adj-solver"]["max-iter"].get<int>();
//    int ptl = options["adj-solver"]["print-lvl"].get<int>();
//    //solver.reset(new HypreGMRES(fes->GetComm()));
//    solver->SetOperator(*jac_trans);
// //    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetTol(tol);
// //    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetMaxIter(maxiter);
// //    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPrintLevel(ptl);
// //    dynamic_cast<mfem::HypreGMRES *>(solver.get())->SetPreconditioner(*dynamic_cast<HypreSolver *>(prec.get()));
//    solver->Mult(*dJ, *adjoint);
//    //adjoint->Set(dt_real_, *adjoint);

// 	if (tr != ti_final)
// 	{
// 		adjoint->Add(1, *adjoint_old);
// 	}

//    adj->SetFromTrueDofs(*adjoint);
//    if (0==rank)
//    {
//       time_end = MPI_Wtime();
//       cout << "Time for solving adjoint is " << (time_end - time_beg) << endl;
//    }

// 	// store previous adjoint (forward in time)
	
// 	adj_old.reset(new GridFunType(*adj));
// 	// {
//     //     ofstream sol_ofs_adj("motor_heat_adj.vtk");
//     //     sol_ofs_adj.precision(14);
//     //     mesh->PrintVTK(sol_ofs_adj, options["space-dis"]["degree"].get<int>());
//     //     adj->SaveVTK(sol_ofs_adj, "Adjoint", options["space-dis"]["degree"].get<int>());
//     //     sol_ofs_adj.close();
//     // }
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
	stiff.reset(new BilinearFormType(fes.get()));
	load.reset(new LinearFormType(fes.get()));
}

void ThermalSolver::addMassIntegrators(double alpha)
{
	mass->AddDomainIntegrator(new MassIntegrator(*rho_cv));
}

void ThermalSolver::addStiffVolumeIntegrators(double alpha)
{
	stiff->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
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

   evolver.reset(new ThermalEvolver(ess_bdr, mass.get(), stiff.get(), load.get(), *out,
												0.0, flux_coeff.get()));
	if (newton_solver == nullptr)
	{
    	// constructNewtonSolver();
	}
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
		y(0) = 1.0;
	}
	else
	{
		// y(0) = -(M_PI/2)*exp(-M_PI*M_PI*time/4);
		y(0) = -1.0;//(M_PI/2)*exp(-M_PI*M_PI*time/4);
		//cout << "outflux val = " << y(0) << std::endl;
	}

	//y(0) = 1;
	y(1) = 1.0;
	y(2) = 0.0000;
	
}

Vector* ThermalSolver::getMeshSensitivities()
{
// 	/// NOTE: Get the method's step from the ode solver
// 	if(options["time-dis"]["ode-solver"].template get<string>() != "MIDPOINT")
// 	{
// 		throw MachException("Only implemented for implicit midpoint!\n");
// 	}
	
// 	/// assign mesh node space to forms
// 	GridFunction *x_nodes_s = mesh->GetNodes();
//     FiniteElementSpace *mesh_fes_s = x_nodes_s->FESpace();
// 	SpaceType mesh_fes(*mesh_fes_s, *mesh);
// 	GridFunType x_nodes(&mesh_fes, x_nodes_s);
// 	j_mesh_sens.reset(new NonlinearFormType(&mesh_fes));

// 	// start by adding/computing the output partial
// 	/// NOTE: Should eventually support different outputs
// 	/// delJdelX
// 	if (rhoa != 0)
// 	{
// 		j_mesh_sens->AddDomainIntegrator(
// 			new AggregateResIntegrator(fes.get(), rhoa, max, u.get()));
// 	}
// 	else
// 	{
// 		j_mesh_sens->AddDomainIntegrator(
// 			new TempResIntegrator(fes.get(), u.get()));
// 	}
// 	std::unique_ptr<GridFunType> dLdX_w; //work vector
// 	dLdX.reset(new GridFunType(x_nodes));
// 	dLdX_w.reset(new GridFunType(x_nodes));
// 	j_mesh_sens->Mult(x_nodes, *dLdX_w);
// 	*dLdX = 0.0;
// 	dLdX->Add(1, *dLdX_w);
// 	adj_old.reset(new GridFunType(*u)); //placeholder

// 	// loop over time steps in reverse
// 	for (int tr = ti_final; tr > 0; tr--)
// 	{
// 		res_mesh_sens.reset(new NonlinearFormType(&mesh_fes));
//    		res_mesh_sens_l.reset(new LinearFormType(&mesh_fes));

// 		/// read the current and previous state from file
// 		//cout << "Reverse time step: " << tr<< endl;
// 		stringstream solname; stringstream solname_old;
// 		solname << "state"<<tr<<".gf"; solname_old << "state"<<tr-1<<".gf";
// 		std::ifstream sol(solname.str()); std::ifstream sol_old(solname_old.str()); 
// 		u.reset(new GridFunType(mesh.get(), sol));
// 		u_old.reset(new GridFunType(mesh.get(), sol_old));

// 		/// compute the adjoint at the current time step
// 		adj = NULL;
// 		solveForAdjoint(options["outputs"]["temp-agg"].get<std::string>());
	
// 		/// recompute dudt if needed
// 		Vector sub(u->Size()); 
// 		sub = 0.0;
// 		sub.Add(1, *u);
// 		sub.Add(-1, *u_old);
// 		dudt.reset(new GridFunType(fes.get()));
// 		dudt->Set(1.0/dt_real_, sub);
	
// 		/// add integrators R = [M + (dt/2)K]dudt + Ku + b = 0
// 		/// dJdX = delJdelX + adj^T dR/dX
// 		/// adj^T Ku
// 		res_mesh_sens->AddDomainIntegrator(
// 			new DiffusionResIntegrator(*kappa, u_old.get(), adj.get()));
// 		/// adj^T Mdudt
// 		res_mesh_sens->AddDomainIntegrator(
// 			new MassResIntegrator(*rho_cv, dudt.get(), adj.get()));
// 		/// adj^T (dt/2)Kdudt (for implicit midpoint)
// 		GridFunType dtdudt(fes.get());
// 		dtdudt.Set(1.0/2.0, sub);
// 		res_mesh_sens->AddDomainIntegrator(
// 			new DiffusionResIntegrator(*kappa, &dtdudt, adj.get()));
// 		/// adj^T load terms
// 		res_mesh_sens->AddDomainIntegrator(
// 			new DomainResIntegrator(*i2sigmainv, adj.get()));
// 		res_mesh_sens->AddDomainIntegrator(
// 			new DomainResIntegrator(*coreloss, adj.get()));

// 		// just to be sure, see if this residual goes to 0?
// 		// GridFunction R(fes.get());
// 		// R = 0.0;
// 		// mass->AddMult(*dudt, R);
// 		// stiff->AddMult(dtdudt, R);
// 		// stiff->AddMult(*u_old, R);
// 		// R.Add(1, *load);
// 		// cout << "Residual Norm (?): " << R.Norml2() << endl;

// 		// outward flux bc
// 		auto &bcs = options["bcs"];
// 		bndry_marker.resize(bcs.size());
// 		int idx = 0;
// 		if (bcs.find("outflux") != bcs.end())
// 		{ 
//     	    vector<int> tmp = bcs["outflux"].get<vector<int>>();
//     	    bndry_marker[idx].SetSize(tmp.size(), 0);
//     	    bndry_marker[idx].Assign(tmp.data());
//     	    res_mesh_sens_l->AddBdrFaceIntegrator(
// 				new BoundaryNormalResIntegrator(*flux_coeff, u_old.get(), 
// 												adj.get()), bndry_marker[idx]);
//     	    idx++;
// 		}

// 		/// Compute the derivatives and accumulate the result
// 		res_mesh_sens->Mult(x_nodes, *dLdX_w);
// 		//double t_final = options["time-dis"]["t-final"].get<double>();
// 		flux_coeff->SetTime(tr*dt_real_);
// 		res_mesh_sens_l->Assemble();
// 		dLdX->Add(-1, *dLdX_w);
// 		dLdX->Add(-1, *res_mesh_sens_l);
// 	}

// 	return dLdX.get();
}

Vector* ThermalSolver::getSurfaceMeshSensitivities()
{
	// Vector *dJdXvect = getMeshSensitivities();
	// GridFunction dJdX(mesh->GetNodes()->FESpace(), dJdXvect->GetData());

	// MSolver->setSens(&dJdX);
	// string dummy = "placeholder";
	// MSolver->solveForAdjoint(dummy);
	// return MSolver->getAdjoint();
}

double ThermalSolver::getOutput()
{
	// compute functional
	if (rhoa != 0)
	{
		return funca->GetIEAggregate(u.get());
	}
	else
	{
		return funct->GetTemp(u.get());
	}
}

// double ThermalSolver::initialTemperature(const Vector &x)
// {
//    return 100;
// }

// double ThermalSolver::temp_0 = 0.0;

ThermalEvolver::ThermalEvolver(Array<int> ess_bdr, BilinearFormType *mass,
										 BilinearFormType *stiff,
										 Vector *load,
										 std::ostream &outstream,
										 double start_time,
										 mfem::VectorCoefficient *_flux_coeff)
	 : MachEvolver(ess_bdr, nullptr, mass, nullptr, stiff, load, nullptr,
						outstream, start_time),
		flux_coeff(_flux_coeff), work(height)
{
   mass->ParFESpace()->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
	// mass->FormSystemMatrix(ess_tdof_list, mMat);
   // stiff->FormSystemMatrix(ess_tdof_list, kMat);
};

void ThermalEvolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	flux_coeff->SetTime(t);
	LinearFormType *load_lf = dynamic_cast<LinearFormType*>(load);
	if (load_lf)
		load_lf->Assemble();
	else
		throw MachException("Couldn't cast load to LinearFormType!\n");
	work.SetSize(x.Size());
	stiff->Mult(x, work);
	work += *load;
   mass_solver.Mult(work, y);
	y *= -1.0;
}

void ThermalEvolver::ImplicitSolve(const double dt, const Vector &x,
                                   Vector &k)
{
   // auto *T = Add(1.0, mMat, dt, kMat);

   // // t_solver->SetOperator(*T);
	// linsolver->SetOperator(*T);

   // kMat.Mult(x, work);
   // work.Neg();  
   // work.Add(-1, *load);
   // linsolver->Mult(work, k);



	// I thought setting this to false would help, it zeros out K each time
	// Still see the behavior where execution changes with each run
	newton->iterative_mode = false;
	flux_coeff->SetTime(t);
	// dynamic_cast<LinearFormType*>(load)->Assemble();
	LinearFormType *load_lf = dynamic_cast<LinearFormType*>(load);
	if (load_lf)
		load_lf->Assemble();
	else
		throw MachException("Couldn't cast load to LinearFormType!\n");
   setOperParameters(dt, &x);
   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   newton->Mult(zero, k);
   MFEM_VERIFY(newton->GetConverged(), "Newton solver did not converge!");
}
}	  
