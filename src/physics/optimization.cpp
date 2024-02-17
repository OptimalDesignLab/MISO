#include "optimization.hpp"
#include "default_options.hpp"
#include "sbp_fe.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"
#include <ctime>
#include <chrono>
using namespace std;
using namespace mfem;
using namespace mach;
using namespace std::chrono;

namespace mach
{
	adept::Stack EulerProblem::diff_stack;

	OptProblem::OptProblem(mfem::Vector init,
												 const std::string &opt_file_name,
												 std::unique_ptr<mfem::Mesh> smesh)
			: Operator(0), designVar(init)
	{
		// 1. get the option file
		options = default_options;
		nlohmann::json file_options;
		ifstream options_file(opt_file_name);
		options_file >> file_options;
		options.merge_patch(file_options);
		cout << setw(3) << options << endl;

		// 2. initialize mesh related
		mesh = std::move(smesh);
		dim = mesh->Dimension();
	}

	LinearProblem::LinearProblem(mfem::Vector init,
															 const std::string &opt_file_name,
															 std::unique_ptr<mfem::Mesh> smesh)
			: OptProblem(init, opt_file_name, std::move(smesh))
	{
		cout << "============ Initializing Linear optimization problem =============\n";
		// 1.  Basic data
		num_state = 1;
		cout << "Problem dimension: " << dim << '\n';
		cout << "Number of State: " << num_state << '\n';

		// construct fespaces
		int dgd_degree = options["space-dis"]["DGD-degree"].get<int>();
		int extra = options["space-dis"]["extra-basis"].get<int>();
		fec.reset(new DG_FECollection(options["space-dis"]["degree"].get<int>(), dim, BasisType::GaussLobatto));
		fes_dgd.reset(new DGDSpace(mesh.get(), fec.get(), designVar, dgd_degree, extra,
															 num_state, Ordering::byVDIM));
		fes_full.reset(new FiniteElementSpace(mesh.get(), fec.get(), num_state,
																					Ordering::byVDIM));

		// 2. grid function
		u_dgd.reset(new CentGridFunction(fes_dgd.get()));
		u_full.reset(new GridFunction(fes_full.get()));
		*u_full = 0.0;
		*u_dgd = 0.0;

		res_dgd.reset(new BilinearForm(fes_dgd.get()));
		res_full.reset(new BilinearForm(fes_full.get()));
		b_dgd.SetSize(fes_dgd->GetTrueVSize());
		b_full.reset(new LinearForm(fes_full.get()));

		// 3. variable size
		ROMSize = u_dgd->Size();
		FullSize = u_full->Size();
		numDesignVar = designVar.Size();
		numBasis = numDesignVar / dim;

		// 4. print basic data
		cout << "dgd_degree is: " << dgd_degree << '\n';
		cout << "u_dgd size is " << u_dgd->Size() << '\n';
		cout << "u_full size is " << u_full->Size() << '\n';
		cout << "Full size model is: " << fes_full->GetTrueVSize() << '\n';
		cout << "DGD model size is " << num_state * dynamic_cast<DGDSpace *>(fes_dgd.get())->GetNDofs() << '\n';
		cout << "res_full size is " << res_full->Height() << " x " << res_full->Width() << '\n';
		cout << "res_dgd size is " << res_dgd->Height() << " x " << res_dgd->Width() << '\n';

		// 5. initialize solver
		solver.reset(new UMFPackSolver());
	}

	void LinearProblem::InitializeSolver(mfem::VectorFunctionCoefficient &velocity,
																			 mfem::FunctionCoefficient &inflow)
	{
		// get options and boundary markers
		double alpha = options["space-dis"]["alpha"].get<double>();
		auto &bcs = options["bcs"];
		vector<int> tmp = bcs["influx"].get<vector<int>>();
		influx_bdr.SetSize(tmp.size(), 0);
		influx_bdr.Assign(tmp.data());
		tmp = bcs["outflux"].get<vector<int>>();
		outflux_bdr.SetSize(tmp.size(), 0);
		outflux_bdr.Assign(tmp.data());

		// set integrators for DGD Operators
		res_dgd->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, alpha));
		res_dgd->AddInteriorFaceIntegrator(new DGTraceIntegrator(velocity, alpha));
		res_dgd->AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, alpha), outflux_bdr);

		// set integrators for full problem operator
		res_full->AddDomainIntegrator(new ConservativeConvectionIntegrator(velocity, alpha));
		res_full->AddInteriorFaceIntegrator(new DGTraceIntegrator(velocity, alpha));
		res_full->AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, alpha), outflux_bdr);

		// add rhs integrator
		b_full->AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, alpha), influx_bdr);

		// assemble operators
		int skip_zero = 0;
		res_full->Assemble(skip_zero);
		res_full->Finalize(skip_zero);
		res_dgd->Assemble(skip_zero);
		res_dgd->Finalize(skip_zero);
		b_full->Assemble();
		SparseMatrix *p = fes_dgd->GetCP();
		p->MultTranspose(*b_full, b_dgd);

		//  Get operators in handy
		k_full = &res_full->SpMat();
		k_dgd = &res_dgd->SpMat();
	}

	void LinearProblem::Mult(const mfem::Vector &x, mfem::Vector &y) const
	{
		// dJ/dc = pJ/pc - pJ/puc * (pR_dgd/puc)^{-1} * pR_dgd/pc
		y.SetSize(numDesignVar); // set y as pJpc
		Vector pJpuc(ROMSize);

		/// first compute some variables that used multiple times
		// 1. get pRpu, pR_dgd/pu_dgd
		SparseMatrix *pRpu = k_full;
		SparseMatrix *pR_dgdpuc = k_dgd;

		// ofstream prpu_save("prpu.txt");
		// pRpu->PrintMatlab(prpu_save);
		// prpu_save.close();

		// ofstream prdgdpuc_save("prdgdpuc.txt");
		// pR_dgdpuc->PrintMatlab(prdgdpuc_save);
		// prdgdpuc_save.close();

		// 2. compute full residual
		cout << "Calcuting full residual...\n";
		Vector r(FullSize);
		k_full->Mult(*u_full, r);
		r -= *b_full;

		// ofstream r_save("r_full.txt");
		// r.Print(r_save, 1);
		// r_save.close();

		/// loop over all design variables
		Vector ppupc_col(FullSize);
		Vector dptpc_col(ROMSize);

		DenseMatrix pPupc(FullSize, numDesignVar);
		DenseMatrix pPtpcR(ROMSize, numDesignVar);
		cout << "Calculating dp/dc_i:\n";
		for (int i = 0; i < numDesignVar; i++)
		{
			cout << "design var " << i;
			SparseMatrix *dPdci = new SparseMatrix(FullSize, ROMSize);
			// get dpdc
			fes_dgd->GetdPdc(i, x, *dPdci);

			// colume of intermediate pPu/pc
			dPdci->Mult(*u_dgd, ppupc_col);
			pPupc.SetCol(i, ppupc_col);

			// colume of pPt / pc * R
			dPdci->MultTranspose(r, dptpc_col);
			pPtpcR.SetCol(i, dptpc_col);
			delete dPdci;
		}

		// ofstream ppupc_save("ppupc.txt");
		// pPupc.PrintMatlab(ppupc_save);
		// ppupc_save.close();

		// ofstream pptpcr_save("pptpcr.txt");
		// pPtpcR.PrintMatlab(pptpcr_save);
		// pptpcr_save.close();

		// compute pJ/pc
		cout << "calculating pj/pc\n";
		Vector temp_vec1(FullSize);
		pRpu->MultTranspose(r, temp_vec1);
		pPupc.MultTranspose(temp_vec1, y);
		y *= 2.0;

		// ofstream pjpc_save("pjpc.txt");
		// y.Print(pjpc_save, 1);
		// pjpc_save.close();

		// compute pJ/puc
		cout << "calculating pj/puc\n";
		SparseMatrix *p = fes_dgd->GetCP();
		p->MultTranspose(temp_vec1, pJpuc);
		pJpuc *= 2.0;

		// ofstream p_save("p.txt");
		// p->PrintMatlab(p_save);
		// p_save.close();

		// ofstream pjpuc_save("pjpuc.txt");
		// pJpuc.Print(pjpuc_save, 1);
		// pjpuc_save.close();

		// compute pR_dgd / pc
		DenseMatrix *temp_mat1 = ::Mult(*pRpu, pPupc);
		SparseMatrix *Pt = Transpose(*p);
		DenseMatrix *pR_dgdpc = ::Mult(*Pt, *temp_mat1);
		*pR_dgdpc += pPtpcR;
		delete temp_mat1;

		// ofstream pt_save("pt.txt");
		// Pt->PrintMatlab(pt_save);
		// pt_save.close();
		// delete Pt;

		// ofstream prdgdpc_save("prdgdpc.txt");
		// pR_dgdpc->PrintMatlab(prdgdpc_save);
		// prdgdpc_save.close();

		// solve for adjoint variable
		Vector adj(ROMSize);
		SparseMatrix *pRt_dgdpuc = Transpose(*pR_dgdpuc);

		solver->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
		solver->SetPrintLevel(1);
		solver->SetOperator(*pRt_dgdpuc);
		solver->Mult(pJpuc, adj);
		delete pRt_dgdpuc;

		// ofstream adj_save("adj.txt");
		// adj.Print(adj_save, 1);
		// adj_save.close();

		// compute the total derivative
		cout << "calculating total derivative...";
		Vector temp_vec2(numDesignVar);
		pR_dgdpc->Transpose();
		pR_dgdpc->Mult(adj, temp_vec2);
		y -= temp_vec2;

		// ofstream djdc_save("djdc.txt");
		// y.Print(djdc_save, 1);
		// djdc_save.close();

		delete pR_dgdpc;
	}

	double LinearProblem::GetEnergy(const mfem::Vector &x) const
	{
		// build new DGD operators
		fes_dgd->buildProlongationMatrix(x);

		solver->SetOperator(*k_dgd);
		solver->Mult(b_dgd, *u_dgd);

		SparseMatrix *p = fes_dgd->GetCP();
		p->Mult(*u_dgd, *u_full);
		Vector r(FullSize);
		k_full->Mult(*u_full, r);
		return r * r;
	}

	LinearProblem::~LinearProblem()
	{
		cout << "deleting linear advection problem optimizer...\n";
	}

	EulerProblem::EulerProblem(mfem::Vector init,
														 const string &opt_file_name,
														 unique_ptr<mfem::Mesh> smesh)
			: OptProblem(init, opt_file_name, std::move(smesh))
	{
		cout << "============ Initializing Euler optimization problem =============\n";
		num_state = dim + 2;
		cout << "Problem dimension: " << dim << '\n';
		cout << "Number of State: " << num_state << '\n';

		int dgd_degree = options["space-dis"]["DGD-degree"].get<int>();
		int extra = options["space-dis"]["extra-basis"].get<int>();
		fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(), dim));
		fes_dgd.reset(new DGDSpace(mesh.get(), fec.get(), designVar, dgd_degree, extra,
															 num_state, Ordering::byVDIM));
		fes_full.reset(new FiniteElementSpace(mesh.get(), fec.get(), num_state,
																					Ordering::byVDIM));

		// construct the gridfunction
		u_dgd.reset(new CentGridFunction(fes_dgd.get()));
		u_full.reset(new GridFunction(fes_full.get()));

		// variable size
		ROMSize = u_dgd->Size();
		FullSize = u_full->Size();
		numDesignVar = designVar.Size();
		numBasis = numDesignVar / dim;

		// construct the residual forms
		res_dgd.reset(new NonlinearForm(fes_dgd.get()));
		res_full.reset(new NonlinearForm(fes_full.get()));

		// check some intermediate info
		cout << "Num of state variables: " << num_state << '\n';
		cout << "dgd_degree is: " << dgd_degree << '\n';
		cout << "u_dgd size is " << u_dgd->Size() << '\n';
		cout << "u_full size is " << u_full->Size() << '\n';
		cout << "Full size model is: " << fes_full->GetTrueVSize() << '\n';
		cout << "DGD model size is (should be number of basis): " << num_state * dynamic_cast<DGDSpace *>(fes_dgd.get())->GetNDofs() << '\n';
		cout << "res_full size is " << res_full->Height() << " x " << res_full->Width() << '\n';
		cout << "res_dgd size is " << res_dgd->Height() << " x " << res_dgd->Width() << '\n';

		if (options["problem-type"].get<string>() == "airfoil")
		{
			mach_fs = options["flow-param"]["mach"].get<double>();
			aoa_fs = options["flow-param"]["aoa"].get<double>() * M_PI / 180;
			iroll = options["flow-param"]["roll-axis"].get<int>();
			ipitch = options["flow-param"]["pitch-axis"].get<int>();
		}
	}

	void EulerProblem::InitializeSolver()
	{
		addVolumeIntegrators(1.0);
		addBoundaryIntegrators(1.0);
		addInterfaceIntegrators(1.0);

		// linear solver
		solver.reset(new UMFPackSolver());
		solver.get()->Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
		solver.get()->SetPrintLevel(1);
		// newton solver
		double nabstol = options["newton"]["abstol"].get<double>();
		double nreltol = options["newton"]["reltol"].get<double>();
		int nmaxiter = options["newton"]["maxiter"].get<int>();
		int nptl = options["newton"]["printlevel"].get<int>();
		newton_solver.reset(new mfem::NewtonSolver());
		newton_solver->iterative_mode = true;
		newton_solver->SetSolver(*solver);
		newton_solver->SetOperator(*res_dgd);
		newton_solver->SetPrintLevel(nptl);
		newton_solver->SetRelTol(nreltol);
		newton_solver->SetAbsTol(nabstol);
		newton_solver->SetMaxIter(nmaxiter);
	}

	void EulerProblem::SetInitialCondition(void (*u_init)(const mfem::Vector &,
																												mfem::Vector &))
	{
		VectorFunctionCoefficient u0(num_state, u_init);
		u_dgd->ProjectCoefficient(u0);
		u_full->ProjectCoefficient(u0);

		GridFunction u_test(fes_full.get());
		dynamic_cast<DGDSpace *>(fes_dgd.get())->GetProlongationMatrix()->Mult(*u_dgd, u_test);

		u_test -= *u_full;
		cout << "After projection, the difference norm is " << u_test.Norml2() << '\n';
	}

	void EulerProblem::SetInitialCondition(const mfem::Vector uic)
	{
		VectorConstantCoefficient u0(uic);
		u_full->ProjectCoefficient(u0);
		u_dgd->ProjectConstVecCoefficient(u0);

		GridFunType u_test(fes_full.get());
		dynamic_cast<DGDSpace *>(fes_dgd.get())->GetProlongationMatrix()->Mult(*u_dgd, u_test);

		u_test -= *u_full;
		cout << "After projection, the difference norm is " << u_test.Norml2() << '\n';
	}

	double EulerProblem::GetEnergy(const Vector &x) const
	{
		// build new DGD operators
		fes_dgd->buildProlongationMatrix(x);
		// solve for DGD solution
		Vector b(numBasis);
		newton_solver->Mult(b, *u_dgd);
		SparseMatrix *prolong = fes_dgd->GetCP();
		// milliseconds ms = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
		// ofstream cp_save("prolong"+to_string(ms.count())+".txt");
		// prolong->PrintMatlab(cp_save);
		// cp_save.close();
		prolong->Mult(*u_dgd, *u_full);
		Vector r(FullSize);
		res_full->Mult(*u_full, r);
		return r * r;
	}

	void EulerProblem::Mult(const Vector &x, Vector &y) const
	{
		// dJ/dc = pJ/pc - pJ/puc * (pR_dgd/puc)^{-1} * pR_dgd/pc
		y.SetSize(numDesignVar); // set y as pJpc
		Vector pJpuc(ROMSize);

		/// first compute some variables that used multiple times
		// 1. get pRpu, pR_dgd/pu_dgd
		SparseMatrix *pRpu = dynamic_cast<SparseMatrix *>(&res_full->GetGradient(*u_full));
		SparseMatrix *pR_dgdpuc = dynamic_cast<SparseMatrix *>(&res_dgd->GetGradient(*u_dgd));

		// ofstream prpu_save("prpu.txt");
		// pRpu->PrintMatlab(prpu_save);
		// prpu_save.close();

		// ofstream prdgdpuc_save("prdgdpuc.txt");
		// pR_dgdpuc->PrintMatlab(prdgdpuc_save);
		// prdgdpuc_save.close();

		// 2. compute full residual
		Vector r(FullSize);
		res_full->Mult(*u_full, r);

		// ofstream r_save("r_full.txt");
		// r.Print(r_save,1);
		// r_save.close();

		/// loop over all design variables
		Vector ppupc_col(FullSize);
		Vector dptpc_col(ROMSize);

		DenseMatrix pPupc(FullSize, numDesignVar);
		DenseMatrix pPtpcR(ROMSize, numDesignVar);
		for (int i = 0; i < numDesignVar; i++)
		{
			SparseMatrix *dPdci = new SparseMatrix(FullSize, ROMSize);
			// get dpdc
			fes_dgd->GetdPdc(i, x, *dPdci);

			// colume of intermediate pPu/pc
			dPdci->Mult(*u_dgd, ppupc_col);
			pPupc.SetCol(i, ppupc_col);

			// colume of pPt / pc * R
			dPdci->MultTranspose(r, dptpc_col);
			pPtpcR.SetCol(i, dptpc_col);
			delete dPdci;
		}

		// ofstream ppupc_save("ppupc.txt");
		// pPupc.PrintMatlab(ppupc_save);
		// ppupc_save.close();

		// ofstream pptpcr_save("pptpcr.txt");
		// pPtpcR.PrintMatlab(pptpcr_save);
		// pptpcr_save.close();

		// compute pJ/pc
		Vector temp_vec1(FullSize);
		pRpu->MultTranspose(r, temp_vec1);
		pPupc.MultTranspose(temp_vec1, y);
		y *= 2.0;

		// ofstream pjpc_save("pjpc.txt");
		// y.Print(pjpc_save,1);
		// pjpc_save.close();

		// compute pJ/puc
		SparseMatrix *P = fes_dgd->GetCP();
		P->MultTranspose(temp_vec1, pJpuc);
		pJpuc *= 2.0;

		// ofstream p_save("p.txt");
		// P->PrintMatlab(p_save);
		// p_save.close();

		// ofstream pjpuc_save("pjpuc.txt");
		// pJpuc.Print(pjpuc_save,1);
		// pjpuc_save.close();

		// compute pR_dgd / pc
		DenseMatrix *temp_mat1 = ::Mult(*pRpu, pPupc);
		SparseMatrix *Pt = Transpose(*P);
		DenseMatrix *pR_dgdpc = ::Mult(*Pt, *temp_mat1);
		delete Pt;
		*pR_dgdpc += pPtpcR;
		delete temp_mat1;

		// ofstream pt_save("pt.txt");
		// Pt->PrintMatlab(pt_save);
		// pt_save.close();

		// ofstream prdgdpc_save("prdgdpc.txt");
		// pR_dgdpc->PrintMatlab(prdgdpc_save);
		// prdgdpc_save.close();

		// solve for adjoint variable
		Vector adj(ROMSize);
		SparseMatrix *pRt_dgdpuc = Transpose(*pR_dgdpuc);
		UMFPackSolver umfsolver;
		umfsolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
		umfsolver.SetPrintLevel(1);
		umfsolver.SetOperator(*pRt_dgdpuc);
		umfsolver.Mult(pJpuc, adj);
		delete pRt_dgdpuc;

		// ofstream adj_sasve("adj.txt");
		// adj.Print(adj_save,1);
		// adj_save.close();

		// compute the total derivative
		Vector temp_vec2(numDesignVar);
		pR_dgdpc->Transpose();
		pR_dgdpc->Mult(adj, temp_vec2);
		y -= temp_vec2;

		// ofstream djdc_save("djdc.txt");
		// y.Print(djdc_save,1);
		// djdc_save.close();

		delete pR_dgdpc;
	}

	void EulerProblem::addVolumeIntegrators(double alpha)
	{
		double lps_coeff = options["space-dis"]["lps-coeff"].get<double>();
		res_full->AddDomainIntegrator(new IsmailRoeIntegrator<2, false>(diff_stack, 1.0));
		res_full->AddDomainIntegrator(new EntStableLPSIntegrator<2, false>(diff_stack, 1.0, lps_coeff));
		res_dgd->AddDomainIntegrator(new IsmailRoeIntegrator<2, false>(diff_stack, 1.0));
		res_dgd->AddDomainIntegrator(new EntStableLPSIntegrator<2, false>(diff_stack, 1.0, lps_coeff));
		if (options["shock-capturing"]["use"].template get<bool>() == true)
		{
			double sensor = options["shock-capturing"]["sensor-param"].get<double>();
			double k = options["shock-capturing"]["k-param"].get<double>();
			double eps = options["shock-capturing"]["eps-param"].get<double>();
			res_dgd->AddDomainIntegrator(new EntStableLPSShockIntegrator<2, false>(diff_stack, alpha, lps_coeff, sensor, k, eps, fec.get()));
			res_full->AddDomainIntegrator(new EntStableLPSShockIntegrator<2, false>(diff_stack, alpha, lps_coeff, sensor, k, eps, fec.get()));
		}
	}

	void EulerProblem::addBoundaryIntegrators(double alpha)
	{
		auto &bcs = options["bcs"];
		bndry_marker.resize(bcs.size());
		int idx = 0;
		if (bcs.find("wedge-shock") != bcs.end())
		{ // shock capturing wedge problem BC
			vector<int> tmp = bcs["wedge-shock"].get<vector<int>>();
			bndry_marker[idx].SetSize(tmp.size(), 0);
			bndry_marker[idx].Assign(tmp.data());
			res_full->AddBdrFaceIntegrator(
					new WedgeShockBC<2, false>(diff_stack, fec.get(), alpha),
					bndry_marker[idx]);
			res_dgd->AddBdrFaceIntegrator(
					new WedgeShockBC<2, false>(diff_stack, fec.get(), alpha),
					bndry_marker[idx]);
			idx++;
		}
		if (bcs.find("vortex") != bcs.end())
		{ // isentropic vortex BC
			if (dim != 2)
			{
				throw MachException("EulerSolver::addBoundaryIntegrators(alpha)\n"
														"\tisentropic vortex BC must use 2D mesh!");
			}
			vector<int> tmp = bcs["vortex"].get<vector<int>>();
			bndry_marker[idx].SetSize(tmp.size(), 0);
			bndry_marker[idx].Assign(tmp.data());
			res_full->AddBdrFaceIntegrator(
					new IsentropicVortexBC<2, false>(diff_stack, fec.get(), alpha),
					bndry_marker[idx]);
			res_dgd->AddBdrFaceIntegrator(
					new IsentropicVortexBC<2, false>(diff_stack, fec.get(), alpha),
					bndry_marker[idx]);
			idx++;
		}
		if (bcs.find("slip-wall") != bcs.end())
		{ // slip-wall boundary condition
			vector<int> tmp = bcs["slip-wall"].get<vector<int>>();
			bndry_marker[idx].SetSize(tmp.size(), 0);
			bndry_marker[idx].Assign(tmp.data());
			res_full->AddBdrFaceIntegrator(
					new SlipWallBC<2, false>(diff_stack, fec.get(), alpha),
					bndry_marker[idx]);
			res_dgd->AddBdrFaceIntegrator(
					new SlipWallBC<2, false>(diff_stack, fec.get(), alpha),
					bndry_marker[idx]);
			idx++;
		}
		if (bcs.find("far-field") != bcs.end())
		{
			// far-field boundary conditions
			vector<int> tmp = bcs["far-field"].template get<vector<int>>();
			mfem::Vector qfar(dim + 2);
			getFreeStreamState(qfar);
			bndry_marker[idx].SetSize(tmp.size(), 0);
			bndry_marker[idx].Assign(tmp.data());
			res_full->AddBdrFaceIntegrator(
					new FarFieldBC<2, false>(diff_stack, fec.get(), qfar, alpha),
					bndry_marker[idx]);
			res_dgd->AddBdrFaceIntegrator(
					new FarFieldBC<2, false>(diff_stack, fec.get(), qfar, alpha),
					bndry_marker[idx]);
			idx++;
		}
		// need to add farfield bc conditions
		for (int k = 0; k < bndry_marker.size(); ++k)
		{
			cout << "boundary_marker[" << k << "]: ";
			for (int i = 0; i < bndry_marker[k].Size(); ++i)
			{
				cout << bndry_marker[k][i] << ' ';
			}
			cout << '\n';
		}
	}

	void EulerProblem::addInterfaceIntegrators(double alpha)
	{
		double diss_coeff = options["space-dis"]["iface-coeff"].get<double>();
		res_full->AddInteriorFaceIntegrator(new InterfaceIntegrator<2, false>(diff_stack, diss_coeff, fec.get(), alpha));
		res_dgd->AddInteriorFaceIntegrator(new InterfaceIntegrator<2, false>(diff_stack, diss_coeff, fec.get(), alpha));
	}

	void EulerProblem::getFreeStreamState(mfem::Vector &q_ref)
	{
		q_ref = 0.0;
		q_ref(0) = 1.0;
		if (dim == 1)
		{
			q_ref(1) = q_ref(0) * mach_fs; // ignore angle of attack
		}
		else
		{
			q_ref(iroll + 1) = q_ref(0) * mach_fs * cos(aoa_fs);
			q_ref(ipitch + 1) = q_ref(0) * mach_fs * sin(aoa_fs);
		}
		q_ref(dim + 1) = 1 / (euler::gamma * euler::gami) + 0.5 * mach_fs * mach_fs;
	}

	void EulerProblem::checkJacobian(Vector &x)
	{
		Vector b(numBasis);
		newton_solver->Mult(b, *u_dgd);
		// SparseMatrix *prolong = fes_dgd->GetCP();
		//  get analytic jacobian
		Vector dJdc_analytic;
		Mult(x, dJdc_analytic);

		// get jacobian from fd method
		double J = GetEnergy(x);
		double Jp;
		double pert = 1e-7;
		Vector centerp(x);
		Vector dJdc_fd(x.Size());
		for (int i = 0; i < numDesignVar; i++)
		{
			cout << i << '\n';
			centerp(i) += pert;
			Jp = GetEnergy(centerp);
			dJdc_fd(i) = (Jp - J) / pert;
			centerp(i) -= pert;
		}
		cout << "jacobian norm is " << dJdc_analytic.Norml2();
		dJdc_fd -= dJdc_analytic;
		cout << ", difference norm is " << dJdc_fd.Norml2() << '\n';
	}

	void EulerProblem::printSolution(const Vector &c, const std::string &file_name)
	{
		// TODO: These mfem functions do not appear to be parallelized
		fes_dgd->buildProlongationMatrix(c);
		GridFunType u_test(fes_full.get());
		dynamic_cast<DGDSpace *>(fes_dgd.get())->GetProlongationMatrix()->Mult(*u_dgd, u_test);

		ofstream initial(file_name + ".vtk");
		initial.precision(14);
		mesh->PrintVTK(initial, 0);
		u_test.SaveVTK(initial, "solution", 0);
		initial.close();
	}

	EulerProblem::~EulerProblem()
	{
		cout << "Deleting the Euler problem optimization..." << '\n';
	}

} // namespace mfem