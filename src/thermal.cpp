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
	: AbstractSolver(opt_file_name, move(smesh))
{
	setInit = false;

	mesh->ReorientTetMesh();
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H(Grad) finite element collection
    h_grad_coll.reset(new H1_FECollection(fe_order, dim));

	/// Create the H(Grad) finite element space
	h_grad_space.reset(new SpaceType(mesh.get(), h_grad_coll.get()));

	/// Create MVP grid function
	theta.reset(new GridFunType(h_grad_space.get()));

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
	//bs->AddDomainIntegrator(new IronLossIntegrator(rho_cv.get()));

	/// add boundary integrator to bilinear form for flux BC, elsewhere is natural
	///TODO: MOVE TO OPERATOR
	fluxcoeff.reset(new VectorFunctionCoefficient(3, FluxFunc));
	auto &bcs = options["bcs"];
    bndry_marker.resize(bcs.size());
	int idx = 0;
	if (bcs.find("outflux") != bcs.end())
    { // outward flux bc
        vector<int> tmp = bcs["outflux"].get<vector<int>>();
        bndry_marker[idx].SetSize(tmp.size(), 0);
        bndry_marker[idx].Assign(tmp.data());
        bs->AddBoundaryIntegrator(new BoundaryNormalLFIntegrator(*fluxcoeff), bndry_marker[idx]);
        idx++;
    }

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
	evolver.reset(new ConductionEvolver(M, K, *bs, *out));
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
	{
        ofstream sol_ofs("motor_heat_init.vtk");
        sol_ofs.precision(14);
        mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
        theta->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
        sol_ofs.close();
    }

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
      	ti++;

      	done = (t >= t_final - 1e-8 * dt);
    }

    {
        ofstream osol("motor_heat.gf");
        osol.precision(precision);
        theta->Save(osol);
    }
	{
        ofstream sol_ofs("motor_heat.vtk");
        sol_ofs.precision(14);
        mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
        theta->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
        sol_ofs.close();
    }

}

void ThermalSolver::setStaticMembers()
{
	outflux = options["bcs"]["const-val"].get<double>();
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
			auto current = options["current"].template get<double>();
			i2sigmainv_coeff.reset(new ConstantCoefficient(current*current/sigma));
			i2sigmainv->addCoefficient(component["attr"].template get<int>(), move(i2sigmainv_coeff));
		}
		
	}
}

void ThermalSolver::FluxFunc(const Vector &x, Vector &y )
{
	y.SetSize(3);
	//use constant in time for now

	//assuming centered coordinate system, will offset
	double th;// = atan(x(1)/x(0));

	if (x(2) > .5)
	{
		th = 1;
	}
	else
	{
		th = -1;
	}

	y(0) = 0;
	y(1) = 0;
	y(2) = th*outflux;
	
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
   return theta->ComputeL2Error(exsol);

//    double loc_norm = 0.0;
//    const FiniteElement *fe;
//    ElementTransformation *T;
//    DenseMatrix vals, exact_vals;
//    Vector loc_errs;

//    if (entry < 0)
//    {
//       // sum up the L2 error over all states
//       for (int i = 0; i < fes->GetNE(); i++)
//       {
//          fe = fes->GetFE(i);
//          const IntegrationRule *ir = &(fe->GetNodes());
//          T = fes->GetElementTransformation(i);
//          theta->GetValues(*T, *ir, vals);
//          exsol.Eval(exact_vals, *T, *ir);
//          vals -= exact_vals;
//          loc_errs.SetSize(vals.Width());
//          vals.Norm2(loc_errs);
//          for (int j = 0; j < ir->GetNPoints(); j++)
//          {
//             const IntegrationPoint &ip = ir->IntPoint(j);
//             T->SetIntPoint(&ip);
//             loc_norm += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
//          }
//       }
//    }
//    else
//    {
//       // calculate the L2 error for component index `entry`
//       for (int i = 0; i < fes->GetNE(); i++)
//       {
//          fe = fes->GetFE(i);
//          const IntegrationRule *ir = &(fe->GetNodes());
//          T = fes->GetElementTransformation(i);
//          theta->GetValues(*T, *ir, vals);
//          exsol.Eval(exact_vals, *T, *ir);
//          vals -= exact_vals;
//          loc_errs.SetSize(vals.Width());
//          vals.GetRow(entry, loc_errs);
//          for (int j = 0; j < ir->GetNPoints(); j++)
//          {
//             const IntegrationPoint &ip = ir->IntPoint(j);
//             T->SetIntPoint(&ip);
//             loc_norm += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
//          }
//       }
//    }
//    double norm;
// #ifdef MFEM_USE_MPI
//    MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, comm);
// #else
//    norm = loc_norm;
// #endif
//    if (norm < 0.0) // This was copied from mfem...should not happen for us
//    {
//       return -sqrt(-norm);
//    }
//    return sqrt(norm);
}

double ThermalSolver::InitialTemperature(const Vector &x)
{
   return 100;
}

double ThermalSolver::temp_0 = 0.0;
double ThermalSolver::outflux = 0.0;

} // namespace mach

