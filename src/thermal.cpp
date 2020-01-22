#include "magnetostatic.hpp"

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
	mesh->ReorientTetMesh();
	int fe_order = options["space-dis"]["degree"].get<int>();

	/// Create the H(Grad) finite element collection
    h_grad_coll.reset(new H1_FECollection(fe_order, dim));

	/// Create the H(Grad) finite element space
	h_grad_space.reset(new SpaceType(mesh.get(), h_grad_coll.get()));

	/// Create MVP grid function
	T.reset(new GridFunType(h_grad_space.get()));
	dTdt.reset(new GridFunType(h_grad_space.get()));
	rhs.reset(new GridFunType(h_grad_space.get()));


	current_vec.reset(new GridFunType(h_curl_space.get()));

#ifdef MFEM_USE_MPI
   //cout << "Number of finite element unknowns: "
   //     << h_grad_space->GlobalTrueSize() << endl;
#else
   cout << "Number of finite element unknowns: "
        << h_grad_space->GetNDofs() << endl;
#endif

    ifstream material_file(options["material-lib-path"].get<string>());
	/// TODO: replace with mach exception
	if (!material_file)
		std::cerr << "Could not open materials library file!" << std::endl;
	material_file >> materials;

	///TODO: Define these based on materials library
	constructDensityCoeff();

	constructHeatCoeff();

	constructMassCoeff();

    constructConductivity();
     
    constructElecConductivity();

	/// set up the bilinear form
	m.reset(new BilinearFormType(h_curl_space.get()));
    k.reset(new BilinearFormType(h_curl_space.get()));

	/// add mass integrator to bilinear form
	m->AddDomainIntegrator(new MassIntegrator(rho_cv.get()));

	/// add diffusion integrator to bilinear form
	k->AddDomainIntegrator(new DiffusionIntegrator(kappa.get()));

	/// add boundary integrator to bilinear form for flux BC, elsewhere is natural
	///TODO: Define Boundary Conditions, Make More General, Selectively Apply To Certain Faces
	auto &bcs = options["bcs"];
    bndry_marker.resize(bcs.size());
	int idx = 0;
	if (bcs.find("outflux") != bcs.end())
    { // isentropic vortex BC
        vector<int> tmp = bcs["vortex"].get<vector<int>>();
        bndry_marker[idx].SetSize(tmp.size(), 0);
        bndry_marker[idx].Assign(tmp.data());
        k->AddBdrFaceIntegrator(new OutwardHeatFluxBC(), bndry_marker[idx]);
        idx++;
    }
	

	/// set up the linear form (volumetric fluxes)
	b.reset(new LinearFormType(h_curl_space.get()));

	/// add joule heating term
	b->AddDomainIntegrator(new JouleIntegrator(sigma.get()));

	/// add iron loss heating terms
	b->AddDomainIntegrator(new IronLossIntegrator(rho_cv.get()));


	/// assemble matrices
	m->Assemble();
	k->Assemble();
	b->Assemble();

	m->Finalize();
	k->Finalize();
	b->Finalize();

	/// initialize dTdt 0
	*dTdt = 0.0;

	Array<int> thermal_ess_tdof_list;
    h_grad_space.GetEssentialTrueDofs(ess_bdr, thermal_ess_tdof_list);

	/// define ode solver
	ode_solver = NULL;
 	ode_solver.reset(new ImplicitMidpointSolver);

	/// Construct linear system solver
	///TODO: Think about preconditioner
#ifdef MFEM_USE_MPI
   // prec.reset(new HypreBoomerAMG());
   prec.reset(new HypreAMS(h_curl_space.get()));
   prec->SetPrintLevel(0); // Don't want preconditioner to print anything
	prec->SetSingularProblem();

   // solver.reset(new HyprePCG(h_curl_space->GetComm()));
	solver.reset(new HypreGMRES(h_curl_space->GetComm()));
	std::cout << "set tol\n";
    solver->SetTol(options["lin-solver"]["tol"].get<double>());
	std::cout << "set tol\n";
	std::cout << "set iter\n";
    solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
	std::cout << "set iter\n";
	std::cout << "set print\n";
    solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
	std::cout << "set print\n";
    //solver->SetPreconditioner(*prec);
#else
	#ifdef MFEM_USE_SUITESPARSE
	prec = NULL;
	solver.reset(new UMFPackSolver);
	#else
	//prec.reset(new GSSmoother);

	solver.reset(new CGSolver());
    solver->SetPrintLevel(options["lin-solver"]["print-lvl"].get<int>());
    solver->SetMaxIter(options["lin-solver"]["max-iter"].get<int>());
    solver->SetRelTol(options["lin-solver"]["rel-tol"].get<double>());
    solver->SetAbsTol(options["lin-solver"]["abs-tol"].get<double>());
    //solver->SetPreconditioner(*prec);
	#endif
#endif
}

void ThermalSolver::solveUnsteady()
{
	double t = 0.0;
    evolver->SetTime(t);
    ode_solver->Init(*evolver);

	int precision = 8;
    {
		ofstream omesh("motor_heat_init.mesh");
    	omesh.precision(precision);
    	mesh->Print(omesh);
    	ofstream osol("motor_heat_init.gf");
    	osol.precision(precision);
      	u->Save(osol);
   	}

	bool done = false;
    double t_final = options["time-dis"]["t-final"].get<double>();
    double dt = options["time-dis"]["dt"].get<double>();

	for (int ti = 0; !done;)
    {
      	if (options["time-dis"]["const-cfl"].get<bool>())
    	{
    	    dt = calcStepSize(options["time-dis"]["cfl"].get<double>());
    	}
    	double dt_real = min(dt, t_final - t);
    	if (ti % 100 == 0)
    	{
        	 cout << "iter " << ti << ": time = " << t << ": dt = " << dt_real
              << " (" << round(100 * t / t_final) << "% complete)" << endl;
      	}
#ifdef MFEM_USE_MPI
	    HypreParVector *TV = T->GetTrueDofs();
	    ode_solver->Step(*TV, t, dt_real);
	    *T = *TV;
#else
      	ode_solver->Step(*T, t, dt_real);
#endif
      	ti++;

      	done = (t >= t_final - 1e-8 * dt);
    }

    {
        ofstream osol("motor_heat.gf");
        osol.precision(precision);
        T->Save(osol);
    }
	{
        ofstream sol_ofs("motor_heat.vtk");
        sol_ofs.precision(14);
        mesh->PrintVTK(sol_ofs, options["space-dis"]["degree"].get<int>() + 1);
        T->SaveVTK(sol_ofs, "Solution", options["space-dis"]["degree"].get<int>() + 1);
        sol_ofs.close();
    }

}

void ThermalSolver::Mult(const Vector &X, Vector &dXdt)
{
   	ImplicitSolve(0.0, X, dXdt);
}

void ThermalSolver::ImplicitSolve(const double dt, const Vector &X, Vector &dXdt)
{
	k->Mult(*X, *rhs);
	*rhs -= *b;


	m->FormLinearSystem(thermal_ess_tdof_list, *dTdt, *rhs, A, X, B);
	solver->Mult(*B, dTdt);
	m->RecoverFEMSolution(dTdt, *rhs, *dTdt);
}

void ThermalSolver::constructDensityCoeff()
{
	rho.reset(new MeshDependentCoefficient());

	for (auto& material : options["materials"])
	{
		std::unique_ptr<mfem::Coefficient> rho_coeff;
		std::cout << material << '\n';
		{
			auto rho = material["rho"].get<double>();
			rho_coeff.reset(new ConstantCoefficient(rho));
		}
		rho->addCoefficient(material["attr"].get<int>(), move(rho_coeff));
	}
}

void ThermalSolver::constructHeatCoeff()
{
	cv.reset(new MeshDependentCoefficient());

	for (auto& material : options["materials"])
	{
		std::unique_ptr<mfem::Coefficient> cv_coeff;
		std::cout << material << '\n';
		{
			auto cv = material["cv"].get<double>();
			cv_coeff.reset(new ConstantCoefficient(cv));
		}
		cv->addCoefficient(material["attr"].get<int>(), move(cv_coeff));
	}
}


void ThermalSolver::constructMassCoeff()
{
	rho_cv.reset(new ProductCoefficient(rho,cv));
}

void ThermalSolver::constructConductivity()
{
	kappa.reset(new MeshDependentCoefficient());

	for (auto& material : options["materials"])
	{
		std::unique_ptr<mfem::Coefficient> kappa_coeff;
		std::cout << material << '\n';
		{
			auto kappa = material["kappa"].get<double>();
			kappa_coeff.reset(new ConstantCoefficient(kappa));
		}
		kappa->addCoefficient(material["attr"].get<int>(), move(kappa_coeff));

		///TODO: generate anisotropic conductivity for the copper windings
	}
}

void ThermalSolver::constructElecConductivity()
{
	sigmainv.reset(new MeshDependentCoefficient());

	for (auto& material : options["materials"])
	{
		std::unique_ptr<mfem::Coefficient> sigmainv_coeff;
		std::cout << material << '\n';
		if(material["conductor"].template get<bool>())
		{
			auto sigma = material["sigma"].get<double>();
			sigmainv_coeff.reset(new ConstantCoefficient(1/sigma));
		}
		sigmainv->addCoefficient(material["attr"].get<int>(), move(sigmainv_coeff));
	}
}

void ThermalSolver::constructCurrent()
{
	// Can we get this from the em solver? To avoid recomputing divergence-free current field
	// all we need is I^2*(1/Sigma)

	// assume isotropic
	double isr = sigmainv.Eval()*current_vec*current_vec;
}

} // namespace mach