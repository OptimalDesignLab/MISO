// #include <fstream>

// #include "thermal.hpp"
// #include "evolver.hpp"

// using namespace std;
// using namespace mfem;

// namespace mach
// {

// static std::default_random_engine gen(std::random_device{}());
// static std::uniform_real_distribution<double> uniform_rand(0.0, 1.0);

// static double thing = 1.0;
// void ThermalSolver::randState(const mfem::Vector &x, mfem::Vector &u)
// {
// 	thing = thing * -1.0;
//     for (int i = 0; i < u.Size(); ++i)
//     {
//         //u(i) = 2.0 * uniform_rand(gen) - 1.0;
// 		// if (i+1 == u.Size())
// 		// {
// 		// 	u(i) = x(0)*0.1;
// 		// }
// 		// else
// 		// {
// 		// 	u(i) = x(i+1)*0.1;
// 		// }
// 		u(i) = thing;
// 	}
// }

// void ThermalSolver::verifyMeshSensitivities()
// {
// 	std::cout << "Verifying Mesh Sensitivities..." << std::endl;
// 	int dim = mesh->SpaceDimension();
// 	double delta = 1e-7;
// 	double delta_cd = 1e-5;
// 	double dJdX_fd_v = -getOutput()/delta;
// 	double dJdX_cd_v;
// 	Vector *dJdX_vect = getMeshSensitivities();
//     // extract mesh nodes and get their finite-element space

//     GridFunction *x_nodes = mesh->GetNodes();
//     FiniteElementSpace *mesh_fes = x_nodes->FESpace();
// 	GridFunction dJdX(mesh_fes, dJdX_vect->GetData());
// 	GridFunction dJdX_fd(mesh_fes); GridFunction dJdX_cd(mesh_fes);
// 	GridFunction dJdX_fd_err(mesh_fes); GridFunction dJdX_cd_err(mesh_fes);
//     // initialize the vector that we use to perturb the mesh nodes
//     GridFunction v(mesh_fes);
//     VectorFunctionCoefficient v_rand(dim, randState);
//     v.ProjectCoefficient(v_rand);
//     // contract dJ/dX with v
//     double dJdX_v = (dJdX) * v;

// 	if(options["verify-full"].get<bool>())
// 	{
// 		for(int k = 0; k < x_nodes->Size(); k++)
// 		{
// 			GridFunction x_pert(*x_nodes);
// 			x_pert(k) += delta; mesh->SetNodes(x_pert);
//     		std::cout << "Solving Forward Step..." << std::endl;
// 			initDerived();
// 			constructLinearSolver(options["lin-solver"]);
// 			constructNewtonSolver();
// 			evolver->SetLinearSolver(solver.get());
//     		evolver->SetNewtonSolver(newton_solver.get());
// 			ConstantCoefficient u0(options["init-temp"].get<double>());
// 			u->ProjectCoefficient(u0);
// 			if(rhoa != 0)
// 			{
// 				funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 			}
// 			else
// 			{
// 				funct.reset(new TempIntegrator(fes.get()));
// 			}
//     		solveForState();
//     		std::cout << "Solver Done" << std::endl;
// 			dJdX_fd(k) = getOutput()/delta + dJdX_fd_v;
// 			x_pert(k) -= delta; mesh->SetNodes(x_pert);

// 		}
// 		//central difference
// 		for(int k = 0; k < x_nodes->Size(); k++)
// 		{
// 			//forward
// 			GridFunction x_pert(*x_nodes);
// 			x_pert(k) += delta_cd; mesh->SetNodes(x_pert);
//     		std::cout << "Solving Forward Step..." << std::endl;
// 			initDerived();
// 			constructLinearSolver(options["lin-solver"]);
// 			constructNewtonSolver();
// 			evolver->SetLinearSolver(solver.get());
//     		evolver->SetNewtonSolver(newton_solver.get());
// 			ConstantCoefficient u0(options["init-temp"].get<double>());
// 			u->ProjectCoefficient(u0);
// 			if(rhoa != 0)
// 			{
// 				funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 			}
// 			else
// 			{
// 				funct.reset(new TempIntegrator(fes.get()));
// 			}
//     		solveForState();
//     		std::cout << "Solver Done" << std::endl;
// 			dJdX_cd(k) = getOutput()/(2*delta_cd);
// 			//backward
// 			x_pert(k) -= 2*delta_cd; mesh->SetNodes(x_pert);
//     		std::cout << "Solving Backward Step..." << std::endl;
// 			initDerived();
// 			constructLinearSolver(options["lin-solver"]);
// 			constructNewtonSolver();
// 			evolver->SetLinearSolver(solver.get());
//     		evolver->SetNewtonSolver(newton_solver.get());
// 			u->ProjectCoefficient(u0);
// 			if(rhoa != 0)
// 			{
// 				funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 			}
// 			else
// 			{
// 				funct.reset(new TempIntegrator(fes.get()));
// 			}
//     		solveForState();
//     		std::cout << "Solver Done" << std::endl;
// 			dJdX_cd(k) -= getOutput()/(2*delta_cd);
// 			x_pert(k) += delta_cd; mesh->SetNodes(x_pert);
// 		}
// 		dJdX_fd_v = dJdX_fd*v;
// 		dJdX_cd_v = dJdX_cd*v;
// 		dJdX_fd_err += dJdX_fd; dJdX_fd_err -= dJdX;
// 		dJdX_cd_err += dJdX_cd; dJdX_cd_err -= dJdX;
// 		std::cout << "FD L2:  " << dJdX_fd_err.Norml2() << std::endl;
// 		std::cout << "CD L2:  " << dJdX_cd_err.Norml2() << std::endl;
// 		for(int k = 0; k < x_nodes->Size(); k++)
// 		{
// 			dJdX_fd_err(k) = dJdX_fd_err(k)/dJdX(k);
// 			dJdX_cd_err(k) = dJdX_cd_err(k)/dJdX(k);
// 		}
// 		stringstream fderrname;
// 		fderrname << "dJdX_fd_err.gf";
//     	ofstream fd(fderrname.str()); fd.precision(15);
// 		dJdX_fd_err.Save(fd);

// 		stringstream cderrname;
// 		cderrname << "dJdX_cd_err.gf";
//     	ofstream cd(cderrname.str()); cd.precision(15);
// 		dJdX_cd_err.Save(cd);

// 		stringstream analytic;
// 		analytic << "dJdX.gf";
//     	ofstream an(analytic.str()); an.precision(15);
// 		dJdX.Save(an);
// 	}
// 	else
// 	{
//     	// compute finite difference approximation
//     	GridFunction x_pert(*x_nodes);
//     	x_pert.Add(delta, v);
//     	mesh->SetNodes(x_pert);
//     	std::cout << "Solving Forward Step..." << std::endl;
// 		initDerived();
// 		constructLinearSolver(options["lin-solver"]);
// 		constructNewtonSolver();
// 		evolver->SetLinearSolver(solver.get());
//     	evolver->SetNewtonSolver(newton_solver.get());
// 		ConstantCoefficient u0(options["init-temp"].get<double>());
// 		u->ProjectCoefficient(u0);
// 		if(rhoa != 0)
// 		{
// 			funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 		}
// 		else
// 		{
// 			funct.reset(new TempIntegrator(fes.get()));
// 		}
//     	solveForState();
//     	std::cout << "Solver Done" << std::endl;
//     	dJdX_fd_v += getOutput()/delta;

// 		// central difference approximation
// 		std::cout << "Solving CD Backward Step..." << std::endl;
// 		x_pert = *x_nodes; x_pert.Add(-delta_cd, v);
// 		mesh->SetNodes(x_pert);
// 		initDerived();
// 		constructLinearSolver(options["lin-solver"]);
// 		constructNewtonSolver();
// 		evolver->SetLinearSolver(solver.get());
//     	evolver->SetNewtonSolver(newton_solver.get());
// 		u->ProjectCoefficient(u0);
// 		if(rhoa != 0)
// 		{
// 			funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 		}
// 		else
// 		{
// 			funct.reset(new TempIntegrator(fes.get()));
// 		}
//     	solveForState();
//     	std::cout << "Solver Done" << std::endl;
//     	dJdX_cd_v = -getOutput()/(2*delta_cd);

// 		std::cout << "Solving CD Forward Step..." << std::endl;
// 		x_pert.Add(2*delta_cd, v);
// 		mesh->SetNodes(x_pert);
// 		initDerived();
// 		constructLinearSolver(options["lin-solver"]);
// 		constructNewtonSolver();
// 		evolver->SetLinearSolver(solver.get());
//     	evolver->SetNewtonSolver(newton_solver.get());
// 		u->ProjectCoefficient(u0);
// 		if(rhoa != 0)
// 		{
// 			funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 		}
// 		else
// 		{
// 			funct.reset(new TempIntegrator(fes.get()));
// 		}
//     	solveForState();
//     	std::cout << "Solver Done" << std::endl;
//     	dJdX_cd_v += getOutput()/(2*delta_cd);
// 	}

// 	std::cout << "Volume Mesh Sensititivies:  " << std::endl;
//     std::cout << "Finite Difference:  " << dJdX_fd_v << std::endl;
// 	std::cout << "Central Difference: " << dJdX_cd_v << std::endl;
//     std::cout << "Analytic: 		  " << dJdX_v << std::endl;
// 	std::cout << "FD Relative: 		  " << (dJdX_v-dJdX_fd_v)/dJdX_v <<
// std::endl;
//     std::cout << "FD Absolute: 		  " << dJdX_v - dJdX_fd_v << std::endl;
// 	std::cout << "CD Relative: 		  " << (dJdX_v-dJdX_cd_v)/dJdX_v <<
// std::endl;
//     std::cout << "CD Absolute: 		  " << dJdX_v - dJdX_cd_v << std::endl;
// }

// void ThermalSolver::verifySurfaceMeshSensitivities()
// {
// #ifdef MFEM_USE_EGADS
// 	std::cout << "Verifying Surface Mesh Sensitivities..." << std::endl;

// 	int dim = mesh->SpaceDimension();
// 	double delta = 1e-7;
// 	double delta_cd = 1e-5;

// 	// extract mesh nodes and get their finite-element space
//     GridFunction *x_nodes = mesh->GetNodes();
//     FiniteElementSpace *mesh_fes = x_nodes->FESpace();
//     // initialize the vector that we use to perturb the mesh nodes
//     GridFunction v(mesh_fes);
//     VectorFunctionCoefficient v_rand(dim, randState);
//     v.ProjectCoefficient(v_rand);
// 	GridFunction x_pert(*x_nodes);
// 	GridFunction x_pert_true(*x_nodes);

// 	//(v_bnd, might not need v_bnd though)
// 	Array<int> ess_bdr_test(mesh->bdr_attributes.Max());
// 	ess_bdr_test = 1;
// 	Array<int> ess_tdof_list_test;
// 	mesh_fes->GetEssentialTrueDofs(ess_bdr_test, ess_tdof_list_test);
// 	GridFunction v_bnd(mesh_fes); v_bnd = 0.0;
// 	for (int p = 0; p < ess_tdof_list_test.Size(); p++)
// 	{
// 		int in = ess_tdof_list_test[p];
// 		v_bnd(in) = v(in);
// 	}
// 	MeshType* moved_mesh;
//     x_pert.Set(delta, v_bnd);
// 	// 		stringstream pertname;
// 	// 		pertname << "x_pert.gf";
//     // 	    ofstream pert(pertname.str());
// 	// 		x_pert.Save(pert);

// 	// set up the mesh movement solver (should make this it's own function)
// 	MSolver.reset(new LEAnalogySolver(
// 						options["mesh-move-opts-path"].get<string>(),
// 						&x_pert));
// 	MSolver->setMesh(mesh.get());	// give it the mesh pointer
// 	MSolver->setPert(&x_pert);
// 	MSolver->initDerived();

// 	double dJdXs_fd_v = -getOutput()/delta;
// 	double dJdXs_cd_v;
// 	Vector *dJdXs_vect = getSurfaceMeshSensitivities();
// 	GridFunction dJdXs(mesh_fes, dJdXs_vect->GetData());
// 	GridFunction dJdXs_fd(mesh_fes); GridFunction dJdXs_cd(mesh_fes);
// 	GridFunction dJdXs_fd_err(mesh_fes); GridFunction dJdXs_cd_err(mesh_fes);
// 	stringstream solname;
// 	solname << "surfacesens.gf";
//     ofstream ssol(solname.str());
// 	dJdXs.Save(ssol);

//     // contract dJ/dXs with v (v_bnd?)
//     double dJdXs_v = (dJdXs) * v_bnd;

// 	if(options["verify-full"].get<bool>())
// 	{
// 		for (int p = 0; p < ess_tdof_list_test.Size(); p++)
// 		{
// 			mesh->SetNodes(x_pert_true);
// 			int in = ess_tdof_list_test[p]; x_pert = 0.0;
// 			x_pert(in) += delta;
// 			MSolver->setMesh(mesh.get());	// give it the mesh pointer
// 			MSolver->setPert(&x_pert);
// 			MSolver->initDerived();
// 			MSolver->solveForState();
// 			moved_mesh = MSolver->getMesh();
// 			mesh->SetNodes(*moved_mesh->GetNodes());
// 			std::cout << "Solving Forward Step..." << std::endl;
// 			initDerived();
// 			constructLinearSolver(options["lin-solver"]);
// 			constructNewtonSolver();
// 			evolver->SetLinearSolver(solver.get());
//     		evolver->SetNewtonSolver(newton_solver.get());
// 			ConstantCoefficient u0(options["init-temp"].get<double>());
// 			u->ProjectCoefficient(u0);
// 			if(rhoa != 0)
// 			{
// 				funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 			}
// 			else
// 			{
// 				funct.reset(new TempIntegrator(fes.get()));
// 			}
//     		solveForState();
//     		std::cout << "Solver Done" << std::endl;
// 			dJdXs_fd(in) = getOutput()/delta + dJdXs_fd_v;
// 		}
// 		//central difference
// 		if(options["verify-cd"].get<bool>())
// 		{
// 			for(int p = 0; p < ess_tdof_list_test.Size(); p++)
// 			{
// 			mesh->SetNodes(x_pert_true);
// 			int in = ess_tdof_list_test[p]; x_pert = 0.0;
// 			x_pert(in) += delta_cd;
// 			MSolver->setMesh(mesh.get());	// give it the mesh pointer
// 			MSolver->setPert(&x_pert);
// 			MSolver->initDerived();
// 			MSolver->solveForState();
// 			moved_mesh = MSolver->getMesh();
// 			mesh->SetNodes(*moved_mesh->GetNodes());
// 			std::cout << "Solving Forward CD Step..." << std::endl;
// 			initDerived();
// 			constructLinearSolver(options["lin-solver"]);
// 			constructNewtonSolver();
// 			evolver->SetLinearSolver(solver.get());
//     		evolver->SetNewtonSolver(newton_solver.get());
// 			ConstantCoefficient u0(options["init-temp"].get<double>());
// 			u->ProjectCoefficient(u0);
// 			if(rhoa != 0)
// 			{
// 				funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 			}
// 			else
// 			{
// 				funct.reset(new TempIntegrator(fes.get()));
// 			}
//     		solveForState();
//     		std::cout << "Solver Done" << std::endl;
// 			dJdXs_cd(in) = getOutput()/(2*delta_cd);

// 			mesh->SetNodes(x_pert_true);
// 			x_pert = 0.0;
// 			x_pert(in) -= delta_cd;
// 			MSolver->setMesh(mesh.get());	// give it the mesh pointer
// 			MSolver->setPert(&x_pert);
// 			MSolver->initDerived();
// 			MSolver->solveForState();
// 			moved_mesh = MSolver->getMesh();
// 			mesh->SetNodes(*moved_mesh->GetNodes());
// 			std::cout << "Solving Backward CD Step..." << std::endl;
// 			initDerived();
// 			constructLinearSolver(options["lin-solver"]);
// 			constructNewtonSolver();
// 			evolver->SetLinearSolver(solver.get());
//     		evolver->SetNewtonSolver(newton_solver.get());
// 			u->ProjectCoefficient(u0);
// 			if(rhoa != 0)
// 			{
// 				funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 			}
// 			else
// 			{
// 				funct.reset(new TempIntegrator(fes.get()));
// 			}
//     		solveForState();
//     		std::cout << "Solver Done" << std::endl;
// 			dJdXs_cd(in) -= getOutput()/(2*delta_cd);
// 			}
// 		}
// 		dJdXs_fd_v = dJdXs_fd*v_bnd;
// 		dJdXs_cd_v = dJdXs_cd*v_bnd;
// 		dJdXs_fd_err += dJdXs_fd; dJdXs_fd_err -= dJdXs;
// 		dJdXs_cd_err += dJdXs_fd; dJdXs_cd_err -= dJdXs;
// 		std::cout << "FD L2:  " << dJdXs_fd_err.Norml2() << std::endl;
// 		std::cout << "CD L2:  " << dJdXs_cd_err.Norml2() << std::endl;
// 		for(int k = 0; k < x_nodes->Size(); k++)
// 		{
// 			dJdXs_fd_err(k) = dJdXs_fd_err(k)/dJdXs(k);
// 			dJdXs_cd_err(k) = dJdXs_cd_err(k)/dJdXs(k);
// 		}
// 		stringstream fderrname;
// 		fderrname << "dJdXs_fd_err.gf";
//     	ofstream fd(fderrname.str()); fd.precision(15);
// 		dJdXs_fd_err.Save(fd);

// 		stringstream cderrname;
// 		cderrname << "dJdXs_cd_err.gf";
//    	    ofstream cd(cderrname.str()); cd.precision(15);
// 		dJdXs_cd_err.Save(cd);

// 		stringstream analytic;
// 		analytic << "dJdXs.gf";
//    	    ofstream an(analytic.str()); an.precision(15);
// 		dJdXs.Save(an);
// 	}
// 	else
// 	{
//     	// compute finite difference approximation
// 		mesh->SetNodes(*x_nodes);
//     	x_pert = 0.0;
// 		x_pert.Add(delta, v_bnd);
// 		MSolver->solveForState();
// 		moved_mesh = MSolver->getMesh();
// 		mesh->SetNodes(*moved_mesh->GetNodes()); //hope this works
//     	std::cout << "Solving Forward Step..." << std::endl;
// 		initDerived();
// 		constructLinearSolver(options["lin-solver"]);
// 		constructNewtonSolver();
// 		evolver->SetLinearSolver(solver.get());
//     	evolver->SetNewtonSolver(newton_solver.get());
// 		ConstantCoefficient u0(options["init-temp"].get<double>());
// 		u->ProjectCoefficient(u0);
// 		if(rhoa != 0)
// 		{
// 			funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 		}
// 		else
// 		{
// 			funct.reset(new TempIntegrator(fes.get()));
// 		}
//     	solveForState();
//     	std::cout << "Solver Done" << std::endl;
//     	dJdXs_fd_v += getOutput()/delta;

// 		// now try central difference
// 		mesh->SetNodes(*x_nodes);
//     	x_pert = 0.0;
// 		x_pert.Add(-delta_cd, v_bnd);
// 		MSolver->setPert(&x_pert);
// 		MSolver->setMesh(mesh.get());	// give it the mesh pointer
// 		MSolver->initDerived();
// 		MSolver->solveForState();
// 		moved_mesh = MSolver->getMesh();
// 		mesh->SetNodes(*moved_mesh->GetNodes());
// 		std::cout << "Solving CD Backward Step..." << std::endl;
// 		initDerived();
// 		constructLinearSolver(options["lin-solver"]);
// 		constructNewtonSolver();
// 		evolver->SetLinearSolver(solver.get());
//     	evolver->SetNewtonSolver(newton_solver.get());
// 		u->ProjectCoefficient(u0);
// 		if(rhoa != 0)
// 		{
// 			funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 		}
// 		else
// 		{
// 			funct.reset(new TempIntegrator(fes.get()));
// 		}
//     	solveForState();
// 		dJdXs_cd_v = -getOutput()/(delta_cd);

// 		x_pert.Add(2*delta_cd, v_bnd);
// 		MSolver->setPert(&x_pert);
// 		MSolver->setMesh(mesh.get());	// give it the mesh pointer
// 		MSolver->initDerived();
// 		MSolver->solveForState();
// 		moved_mesh = MSolver->getMesh();
// 		mesh->SetNodes(*moved_mesh->GetNodes());
// 		std::cout << "Solving CD Forward Step..." << std::endl;
// 		initDerived();
// 		constructLinearSolver(options["lin-solver"]);
// 		constructNewtonSolver();
// 		evolver->SetLinearSolver(solver.get());
//     	evolver->SetNewtonSolver(newton_solver.get());
// 		u->ProjectCoefficient(u0);
// 		if(rhoa != 0)
// 		{
// 			funca.reset(new AggregateIntegrator(fes.get(), rhoa, max));
// 		}
// 		else
// 		{
// 			funct.reset(new TempIntegrator(fes.get()));
// 		}
//     	solveForState();
// 		dJdXs_cd_v += getOutput()/(delta_cd);
// 	}

// 	std::cout << "Surface Mesh Sensititivies:  " << std::endl;
//     std::cout << "Finite Difference:  " << dJdXs_fd_v << std::endl;
// 	std::cout << "Central Difference: " << dJdXs_cd_v << std::endl;
//     std::cout << "Analytic: 		  " << dJdXs_v << std::endl;
// 	std::cout << "FD Relative: 		  " << (dJdXs_v-dJdXs_fd_v)/dJdXs_v <<
// std::endl;
//     std::cout << "FD Absolute: 		  " << dJdXs_v - dJdXs_fd_v << std::endl;
// 	std::cout << "CD Relative: 		  " << (dJdXs_v-dJdXs_cd_v)/dJdXs_v <<
// std::endl;
//     std::cout << "CD Absolute: 		  " << dJdXs_v - dJdXs_cd_v << std::endl;

// }
// }