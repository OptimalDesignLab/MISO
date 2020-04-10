#include <fstream>

#include "thermal.hpp"
#include "evolver.hpp"
#include "material_library.hpp"

using namespace std;
using namespace mfem;

namespace mach
{

ThermalSolver::ThermalSolver(
	 const std::string &opt_file_name,
    std::unique_ptr<mfem::Mesh> smesh,
	 int dim,
	 GridFunType *B)
	: AbstractSolver(opt_file_name, move(smesh)), mag_field(B)
{
	setInit = false;

	mesh->ReorientTetMesh();

	/// Create temperature grid function
	// u.reset(new GridFunType(fes.get()));
	th_exact.reset(new GridFunType(fes.get()));

	/// Set static variables
	setStaticMembers();

#ifdef MFEM_USE_MPI
   *out << "Number of finite element unknowns: "
       << fes->GlobalTrueVSize() << endl;
#else
   *out << "Number of finite element unknowns: "
        << fes->GetNDofs() << endl;
#endif

   //  ifstream material_file(options["material-lib-path"].get<string>());
	// /// TODO: replace with mach exception
	// if (!material_file)
	// 	std::cerr << "Could not open materials library file!" << std::endl;
	// material_file >> materials;

	materials = material_library;

	*out << "Constructing Material Coefficients..." << std::endl;
	
	constructDensityCoeff();

	constructHeatCoeff();

	constructMassCoeff();

   constructConductivity();
     
   constructJoule();

	constructCore();

	std::cout << "Defining Finite Element Spaces..." << std::endl;
	/// set essential BCs (none)
	Array<int> ess_tdof_list;
	mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
	ess_bdr = 0;
   fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

	/// set up the bilinear forms
	// m.reset(new BilinearFormType(fes.get()));
	// k.reset(new BilinearFormType(fes.get()));

	std::cout << "Creating Mass Matrix..." << std::endl;
	/// add mass integrator to m bilinear form
	// m->AddDomainIntegrator(new MassIntegrator(*rho_cv));
	/// assemble mass matrix
	// m->Assemble(0);

	m->FormSystemMatrix(ess_tdof_list, M);

	/// add diffusion integrator to k bilinear form
	// k->AddDomainIntegrator(new DiffusionIntegrator(*kappa));


	/// set up the linear form (volumetric fluxes)
	// bs.reset(new LinearForm(fes.get()));

	/// add joule heating term
	// bs->AddDomainIntegrator(new DomainLFIntegrator(*i2sigmainv));
	// std::cout << "Constructing Boundary Conditions..." << std::endl;
	/// add iron loss heating terms
	// bs->AddDomainIntegrator(new DomainLFIntegrator(*coreloss));


	std::cout << "Assembling Stiffness Matrix..." << std::endl;
	/// assemble stiffness matrix and linear form
	// k->Assemble(0);

	k->FormSystemMatrix(ess_tdof_list, K);
	std::cout << "Assembling Forcing Term..." << std::endl;
	// bs->Assemble();

	std::cout << "Setting Up ODE Solver..." << std::endl;
	/// define ode solver
	ode_solver = NULL;
	ode_solver.reset(new ImplicitMidpointSolver);
	std::string ode_opt = 
		options["time-dis"]["ode-solver"].get<std::string>();
	if (ode_opt == "MIDPOINT")
	{
		ode_solver = NULL;
		ode_solver.reset(new ImplicitMidpointSolver);
	}
	if (ode_opt == "RK4")
	{
		ode_solver = NULL;
		ode_solver.reset(new RK4Solver);
	}
 	
	evolver.reset(new ConductionEvolver(opt_file_name, M, 
										K, move(bs), *out));

	/// TODO: REPLACE WITH DOMAIN BASED TEMPERATURE MAXIMA ARRAY
	rhoa = options["rho-agg"].get<double>();
	//double max = options["max-temp"].get<double>();

	/// assemble max temp array
	max.SetSize(fes->GetMesh()->attributes.Size()+1);
	for (auto& component : options["components"])
	{
		double mat_max = component["max-temp"].get<double>();
		int attrib = component["attr"].get<int>();
		max(attrib) = mat_max;
	}

	/// pass through aggregation parameters for functional
	func.reset(new AggregateIntegrator(fes.get(), rhoa, max));
}

void ThermalSolver::solveUnsteady()
{
	double t = 0.0;
	double agg;
	double gerror = 0;
	evolver->SetTime(t);
	ode_solver->Init(*evolver);

	if (!setInit)
	{
		setInitialCondition(initialTemperature);
	}

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
		agg = func->GetIEAggregate(u.get());
		cout << "aggregated temp constraint = " << agg << endl;

		//compare to actual max, ASSUMING UNIFORM CONSTRAINT
		gerror = (u->Max()/max(1) - agg)/(u->Max()/max(1));
		
	}

	for (int ti = 0; !done;)
	{
		// if (options["time-dis"]["const-cfl"].get<bool>())
    	// {
    	//     dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
    	// }
    	double dt_real = min(dt, t_final - t);
    	//if (ti % 100 == 0)
    	{
			cout << "iter " << ti << ": time = " << t << ": dt = " << dt_real
              << " (" << round(100 * t / t_final) << "% complete)" << endl;
      }
#ifdef MFEM_USE_MPI
		HypreParVector *TV = u->GetTrueDofs();
		ode_solver->Step(*TV, t, dt_real);
		*u = *TV;
#else
		ode_solver->Step(*u, t, dt_real);
#endif

		// compute functional
		if (rhoa != 0)
		{
			agg = func->GetIEAggregate(u.get());
			cout << "aggregated temp constraint = " << agg << endl;

		}

		evolver->updateParameters();

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

void ThermalSolver::setStaticMembers()
{
	temp_0 = options["init-temp"].get<double>();
}

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
				temp_coeff.reset(new SteinmetzCoefficient(rho_val, alpha, freq,
																		kh, ke, mag_field));
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
																			kh, ke, mag_field));
					coreloss->addCoefficient(attribute, move(temp_coeff));
				}
			}
		}
	}
}

double ThermalSolver::calcL2Error(
    double (*u_exact)(const Vector &), int entry)
{
	// TODO: need to generalize to parallel
	FunctionCoefficient exsol(u_exact);
	th_exact->ProjectCoefficient(exsol);

	
	// sol_ofs.precision(14);
	// th_exact->SaveVTK(sol_ofs, "Analytic", options["space-dis"]["degree"].get<int>() + 1);
	// sol_ofs.close();

	return u->ComputeL2Error(exsol);
}

void ThermalSolver::addMassVolumeIntegrators()
{
	mass->AddDomainIntegrator(new MassIntegrator(*rho_cv));
}

void ThermalSolver::addStiffVolumeIntegrators(double alpha)
{
	stiff->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
}

void ThermalSolver::addLoadVolumeIntegrators(double alpha)
{
	/// add joule heating term
	load->AddDomainIntegrator(new DomainLFIntegrator(*i2sigmainv));
	/// add iron loss heating terms
	load->AddDomainIntegrator(new DomainLFIntegrator(*coreloss));
}

double ThermalSolver::initialTemperature(const Vector &x)
{
   return 100;
}

double ThermalSolver::temp_0 = 0.0;

ConductionEvolver::ConductionEvolver(const std::string &opt_file_name, 
									MatrixType &m, 
									MatrixType &k, 
									std::unique_ptr<mfem::LinearForm> b, 
									std::ostream &outstream)
	: ImplicitLinearEvolver(opt_file_name, m, k, move(b), outstream), zero(m.Height())
{
	/// set static members
	setStaticMembers();

	/// set initial boundary state	
	updateParameters();
}

void ConductionEvolver::setStaticMembers()
{
	outflux = options["bcs"]["const-val"].get<double>();
}

/// TODO: move this to addLoadBoundaryIntegrator
/// Make fluxFunc a regular function in this file in anonymous namespace


void ConductionEvolver::updateParameters()
{
	bb.reset(new LinearForm(force->FESpace()));
	rhs.reset(new LinearForm(force->FESpace()));

	/// add boundary integrator to linear form for flux BC, elsewhere is natural 0
	fluxcoeff.reset(new VectorFunctionCoefficient(3, fluxFunc));
	fluxcoeff->SetTime(t);
	auto &bcs = options["bcs"];
    bndry_marker.resize(bcs.size());
	int idx = 0;
	if (bcs.find("outflux") != bcs.end())
	{ // outward flux bc
        vector<int> tmp = bcs["outflux"].get<vector<int>>();
        bndry_marker[idx].SetSize(tmp.size(), 0);
        bndry_marker[idx].Assign(tmp.data());
        bb->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*fluxcoeff), bndry_marker[idx]);
        idx++;
	}
	bb->Assemble();

	rhs->Set(1, *force);
	rhs->Add(1, *bb);
}

void ConductionEvolver::fluxFunc(const Vector &x, double time, Vector &y)
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
		//cout << "outflux val = " << y(0) << std::endl;
	}

	y(1) = 0;
	y(2) = 0;
	
}

double ConductionEvolver::outflux = 0.0;

} // namespace mach

