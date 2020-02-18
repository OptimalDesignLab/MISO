#include "thermal.hpp"
#include "evolver.hpp"

#include <fstream>

using namespace std;
using namespace mfem;
//just copying magnetostatic.cpp for now
namespace mach
{

ThermalSolver::ThermalSolver(
	 const std::string &opt_file_name,
    std::unique_ptr<mfem::Mesh> smesh,
	 int dim)
	: AbstractSolver(opt_file_name, move(smesh)), sol_ofs("motor_heat.vtk")
{
	/// check for B field solution?
	Bfield = nullptr;

	setInit = false;

	mesh->ReorientTetMesh();
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H(Grad) finite element collection
    h_grad_coll.reset(new H1_FECollection(fe_order, dim));

	/// Create the H(Grad) finite element space
	h_grad_space.reset(new SpaceType(mesh.get(), h_grad_coll.get()));

	/// Create temperature grid function
	theta.reset(new GridFunType(h_grad_space.get()));
	th_exact.reset(new GridFunType(h_grad_space.get()));

	/// Set static variables
	setStaticMembers();

#ifdef MFEM_USE_MPI
   cout << "Number of finite element unknowns: "
       << h_grad_space->GlobalTrueVSize() << endl;
#else
   cout << "Number of finite element unknowns: "
        << h_grad_space->GetNDofs() << endl;
#endif

    ifstream material_file(options["material-lib-path"].get<string>());
	/// TODO: replace with mach exception
	if (!material_file)
		std::cerr << "Could not open materials library file!" << std::endl;
	material_file >> materials;

	std::cout << "Constructing Material Coefficients..." << std::endl;
	
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
    h_grad_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

	/// set up the bilinear forms
	m.reset(new BilinearFormType(h_grad_space.get()));
    k.reset(new BilinearFormType(h_grad_space.get()));

	std::cout << "Creating Mass Matrix..." << std::endl;
	/// add mass integrator to m bilinear form
	m->AddDomainIntegrator(new MassIntegrator(*rho_cv));
	/// assemble mass matrix
	m->Assemble(0);
	m->FormSystemMatrix(ess_tdof_list, M);

	/// add diffusion integrator to k bilinear form
	k->AddDomainIntegrator(new DiffusionIntegrator(*kappa));


	/// set up the linear form (volumetric fluxes)
	bs.reset(new LinearForm(h_grad_space.get()));

	/// add joule heating term
	bs->AddDomainIntegrator(new DomainLFIntegrator(*i2sigmainv));
	std::cout << "Constructing Boundary Conditions..." << std::endl;
	/// add iron loss heating terms
	bs->AddDomainIntegrator(new DomainLFIntegrator(*coreloss));


	std::cout << "Assembling Stiffness Matrix..." << std::endl;
	/// assemble stiffness matrix and linear form
	k->Assemble(0);

	k->FormSystemMatrix(ess_tdof_list, K);
	std::cout << "Assembling Forcing Term..." << std::endl;
	bs->Assemble();

	std::cout << "Setting Up ODE Solver..." << std::endl;
	/// define ode solver
	ode_solver = NULL;
 	ode_solver.reset(new ImplicitMidpointSolver);
	evolver.reset(new ConductionEvolver(opt_file_name, M, 
										K, move(bs), *out));
}

void ThermalSolver::solveUnsteady()
{
	double t = 0.0;
    evolver->SetTime(t);
    ode_solver->Init(*evolver);

	if(!setInit)
	{
		FunctionCoefficient theta_0(InitialTemperature);
    	theta->ProjectCoefficient(theta_0);
	}

    Vector u;
    theta->GetTrueDofs(u);

	int precision = 8;
    {
        ofstream osol("motor_heat_init.gf");
        osol.precision(precision);
        theta->Save(osol);
    }
	// {
    //     ofstream sol_ofs("motor_heat_init.vtk");
    //     sol_ofs.precision(14);
    //     mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
    //     theta->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
    //     sol_ofs.close();
    // }

	bool done = false;
    double t_final = options["time-dis"]["t-final"].get<double>();
    double dt = options["time-dis"]["dt"].get<double>();

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
	    HypreParVector *TV = theta->GetTrueDofs();
	    ode_solver->Step(*TV, t, dt_real);
	    *theta = *TV;
#else
      	ode_solver->Step(*theta, t, dt_real);
#endif
		evolver->updateParameters();

      	ti++;

      	done = (t >= t_final - 1e-8 * dt);
    }

    {
        ofstream osol("motor_heat.gf");
        osol.precision(precision);
        theta->Save(osol);
    }
	
        
        sol_ofs.precision(14);
        mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
        theta->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
        
    

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
		std::string material = component["material"].template get<std::string>();
		std::cout << material << '\n';
		{
			auto rho_val = materials[material]["rho"].template get<double>();
			rho_coeff.reset(new ConstantCoefficient(rho_val));
		}
		int attrib = component["attr"].template get<int>();
		rho->addCoefficient(component["attr"].template get<int>(), move(rho_coeff));
	}
}

void ThermalSolver::constructHeatCoeff()
{
	cv.reset(new MeshDependentCoefficient());

	for (auto& component : options["components"])
	{
		std::unique_ptr<mfem::Coefficient> cv_coeff;
		std::string material = component["material"].template get<std::string>();
		std::cout << material << '\n';
		{
			auto cv_val = materials[material]["cv"].template get<double>();
			cv_coeff.reset(new ConstantCoefficient(cv_val));
		}
		cv->addCoefficient(component["attr"].template get<int>(), move(cv_coeff));
	}
}


void ThermalSolver::constructMassCoeff()
{
	rho_cv.reset(new MeshDependentCoefficient());

	for (auto& component : options["components"])
	{
		std::unique_ptr<mfem::Coefficient> rho_cv_coeff;
		std::string material = component["material"].template get<std::string>();
		std::cout << material << '\n';
		{
			//auto attr = material["attr"].get<int>();
			auto cv_val = materials[material]["cv"].template get<double>();
			auto rho_val = materials[material]["rho"].template get<double>();
			//rho_cv_coeff.reset(new ProductCoefficient(rho->getCoefficient(attr), cv->getCoefficient(attr)));
			rho_cv_coeff.reset(new ConstantCoefficient(cv_val*rho_val));
		}
		rho_cv->addCoefficient(component["attr"].template get<int>(), move(rho_cv_coeff));
	}
}

void ThermalSolver::constructConductivity()
{
	kappa.reset(new MeshDependentCoefficient());

	for (auto& component : options["components"])
	{
		std::unique_ptr<mfem::Coefficient> kappa_coeff;
		std::string material = component["material"].template get<std::string>();
		std::cout << material << '\n';
		{
			auto kappa_val = materials[material]["kappa"].template get<double>();
			kappa_coeff.reset(new ConstantCoefficient(kappa_val));
		}
		kappa->addCoefficient(component["attr"].template get<int>(), move(kappa_coeff));

		///TODO: generate anisotropic conductivity for the copper windings
	}
}

void ThermalSolver::constructJoule()
{
	i2sigmainv.reset(new MeshDependentCoefficient());

	for (auto& component : options["components"])
	{
		std::unique_ptr<mfem::Coefficient> i2sigmainv_coeff;
		std::string material = component["material"].template get<std::string>();
		std::cout << material << '\n';
		if(materials[material]["conductor"].template get<bool>())
		{
			auto sigma = materials[material]["sigma"].template get<double>();
			auto current = options["motor-opts"]["current"].template get<double>();
			i2sigmainv_coeff.reset(new ConstantCoefficient(current*current/sigma));
			i2sigmainv->addCoefficient(component["attr"].template get<int>(), move(i2sigmainv_coeff));
		}
		
	}
}

void ThermalSolver::constructCore()
{
	coreloss.reset(new MeshDependentCoefficient());

	for (auto& component : options["components"])
	{
		std::unique_ptr<mfem::Coefficient> coreloss_coeff;
		std::string material = component["material"].template get<std::string>();
		std::cout << material << '\n';
		if(materials[material]["core"].template get<bool>())
		{
			auto rho_val = materials[material]["rho"].template get<double>(); 
			auto alpha = materials[material]["alpha"].template get<double>(); 
			auto freq = options["motor-opts"]["frequency"].template get<double>();
			auto kh = materials[material]["kh"].template get<double>(); 
			auto ke = materials[material]["ke"].template get<double>(); 
			Bmax = 0;
			if(Bfield == nullptr)
			{
				Bmax = 2.5; ///TODO: OBTAIN FROM MAGNETOSTATIC SOLVER
			}
			double loss = rho_val*(kh*freq*pow(Bmax, alpha) + ke*freq*freq*Bmax*Bmax);
			coreloss_coeff.reset(new ConstantCoefficient(loss));
			i2sigmainv->addCoefficient(component["attr"].template get<int>(), move(coreloss_coeff));
		}
		
	}
}

void ThermalSolver::setInitialTemperature(double (*f)(const Vector &))
{
	FunctionCoefficient theta_0(f);
    theta->ProjectCoefficient(theta_0);

	setInit = true;
}

double ThermalSolver::calcL2Error(
    double (*u_exact)(const Vector &), int entry)
{
    // TODO: need to generalize to parallel
    FunctionCoefficient exsol(u_exact);
	th_exact->ProjectCoefficient(exsol);

	
        sol_ofs.precision(14);
        th_exact->SaveVTK(sol_ofs, "Analytic", options["space-dis"]["degree"].get<int>() + 1);
		sol_ofs.close();

    return theta->ComputeL2Error(exsol);
}

double ThermalSolver::InitialTemperature(const Vector &x)
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
	double th;// = atan(x(1)/x(0));

	if (x(0) > .5)
	{
		y(0) = -1;
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

