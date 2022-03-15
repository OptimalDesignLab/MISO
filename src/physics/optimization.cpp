#include "optimization.hpp"
#include "default_options.hpp"
#include "sbp_fe.hpp"
#include "euler_fluxes.hpp"
#include "euler_integ.hpp"

using namespace std;
using namespace mfem;
using namespace mach;


namespace mach
{


adept::Stack DGDOptimizer::diff_stack;


DGDOptimizer::DGDOptimizer(Vector init,
									const string &opt_file_name,
									unique_ptr<mfem::Mesh> smesh)
	: Operator(0), designVar(init)
{
	// get the option fileT
	options = default_options;
	nlohmann::json file_options;
	ifstream options_file(opt_file_name);
	options_file >> file_options;
	options.merge_patch(file_options);
	cout << setw(3) << options << endl;

	// construct mesh
	if (smesh == nullptr)
	{
		smesh.reset(new Mesh(options["mesh"]["file"].get<string>().c_str(),1,1));
	}
	mesh.reset(new Mesh(*smesh));
	dim = mesh->Dimension();
	num_state = dim+2;
	cout << "Number of elements: " << mesh->GetNE() << '\n';

	// construct fespaces
	int dgd_degree = options["space-dis"]["DGD-degree"].get<int>();
	int extra = options["space-dis"]["extra-basis"].get<int>();
	fec.reset(new DSBPCollection(options["space-dis"]["degree"].get<int>(),dim));
	fes_dgd.reset(new DGDSpace(mesh.get(),fec.get(),designVar,dgd_degree,extra,
							num_state,Ordering::byVDIM));
	fes_full.reset(new FiniteElementSpace(mesh.get(),fec.get(),num_state,
							 Ordering::byVDIM));

	// construct the gridfunction
	u_dgd.reset(new CentGridFunction(fes_dgd.get()));
	u_full.reset(new GridFunction(fes_full.get()));

	// construct the residual forms
	res_dgd.reset(new NonlinearForm(fes_dgd.get()));
	res_full.reset(new NonlinearForm(fes_full.get()));

	// check some intermediate info
   cout << "Num of state variables: " << num_state << '\n';
   cout << "dgd_degree is: " << dgd_degree << '\n';
   cout << "u_dgd size is " << u_dgd->Size() << '\n';
   cout << "u_full size is " << u_full->Size() << '\n';
   cout << "Full size model is: "<< fes_full->GetTrueVSize() << '\n';
   cout << "DGD model size is (should be number of basis): " << dynamic_cast<DGDSpace *>(fes_dgd.get())->GetNDofs() << '\n';
   cout << "res_full size is " << res_full->Height() << " x " << res_full->Width() << '\n';
	cout << "res_dgd size is " << res_dgd->Height() << " x " << res_dgd->Width() << '\n';

	// add integrators
	addVolumeIntegrators(1.0);
	addBoundaryIntegrators(1.0);
	addInterfaceIntegrators(1.0);
	cout << "done with adding integrators.\n";
}

void DGDOptimizer::InitializeSolver()
{
	cout << "Initialize solvers in DGD optimization.\n";
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

void DGDOptimizer::SetInitialCondition(void (*u_init)(const mfem::Vector &,
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

double DGDOptimizer::GetEnergy(const Vector &x) const
{
	// build new DGD operators
	fes_dgd->buildProlongationMatrix(x);
	// solve for DGD solution
	Vector b(numBasis);
	newton_solver->Mult(b,*u_dgd);
	Vector r(FullSize);
	SparseMatrix *prolong = fes_dgd->GetCP();
	prolong->Mult(*u_dgd,*u_full); 
	res_full->Mult(*u_full,r);
	return r * r;
}

void DGDOptimizer::Mult(const Vector &x, Vector &y) const
{
	// dJ/dc = pJ/pc - pJ/puc * (pR_dgd/puc)^{-1} * pR_dgd/pc
	cout << "In dgd::mult.\n";
	fes_dgd->buildProlongationMatrix(x);
	fes_dgd->GetCP()->Mult(*u_dgd,*u_full);
	y.SetSize(numDesignVar); // set y as pJpc
	Vector pJpuc(numBasis);
	Vector r(FullSize);
	res_full->Mult(*u_full,r);

	/// first compute some variables that used multiple times
	// 1. get pRpu, pR_dgd/pu_dgd
	SparseMatrix *pRpu = dynamic_cast<SparseMatrix*>(&res_full->GetGradient(*u_full));
	SparseMatrix *pR_dgdpuc = dynamic_cast<SparseMatrix*>(&res_dgd->GetGradient(*u_dgd));
	// cout << "Get 2 jacobian.\n";
	// ofstream jac_save("pRpu.txt");
	// pRpu->PrintMatlab(jac_save);
	// jac_save.close();
	// 2. compute full residual

	cout << "Get residual.\n";
	/// loop over all design variables
	Vector ppupc_col(FullSize);
	Vector dptpc_col(numBasis);
	SparseMatrix dPdci(FullSize,numBasis);
	DenseMatrix pPupc(FullSize,numDesignVar);
	DenseMatrix pPtpcR(numBasis,numDesignVar);
	for (int i = 0; i < numDesignVar; i++)
	{
		cout << i << '\n';
		// get dpdc
		fes_dgd->GetdPdc(i,dPdci);

		// colume of intermediate pPu/pc
		dPdci.Mult(*u_dgd,ppupc_col);
		pPupc.SetCol(i,ppupc_col);

		// colume of pPt / pc * R
		dPdci.MultTranspose(r,dptpc_col);
		pPtpcR.SetCol(i,dptpc_col);
	}
	cout << "Get dpdc.\n";
	// compute pJ/pc
	Vector temp_vec1(FullSize);
	pRpu->MultTranspose(r,temp_vec1);
	pPupc.MultTranspose(temp_vec1,y);
	y *= 2.0;
	cout << "Get pJ/pc.\n";

	// compute pJ/puc
	SparseMatrix *P = fes_dgd->GetCP();
	P->MultTranspose(temp_vec1,pJpuc);
	cout << "Get pJ/puc.\n";

	// compute pR_dgd / pc
	DenseMatrix *temp_mat1 = ::Mult(*pRpu,pPupc);
	SparseMatrix *Pt = Transpose(*P);
	DenseMatrix *pR_dgdpc = ::Mult(*Pt,*temp_mat1);
	*pR_dgdpc += pPtpcR;
	cout << "Get pRdgd/pc.\n";

	// solve for adjoint variable
	Vector adj(numBasis);
	SparseMatrix *pRt_dgdpuc = Transpose(*pR_dgdpuc);
	UMFPackSolver umfsolver;
	umfsolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
	umfsolver.SetPrintLevel(1);
	umfsolver.SetOperator(*pRt_dgdpuc);
	umfsolver.Mult(pJpuc,adj);
	cout << "Get adjoint.\n";


	// compute the total derivative
	Vector temp_vec2(numDesignVar);
	pR_dgdpc->Transpose();
	pR_dgdpc->Mult(adj,temp_vec2);
	y -= temp_vec2;
	cout << "Get dJdc.\n";

	delete Pt;
	delete pR_dgdpuc;
	delete pRt_dgdpuc;
	delete temp_mat1;
	delete pR_dgdpc;
}	


void DGDOptimizer::addVolumeIntegrators(double alpha)
{
	double lps_coeff = options["space-dis"]["lps-coeff"].get<double>();
	res_full->AddDomainIntegrator(new IsmailRoeIntegrator<2,false>(diff_stack,1.0));
	res_full->AddDomainIntegrator(new EntStableLPSIntegrator<2,false>(diff_stack,1.0,lps_coeff));
	res_dgd->AddDomainIntegrator(new IsmailRoeIntegrator<2,false>(diff_stack,1.0));
	res_dgd->AddDomainIntegrator(new EntStableLPSIntegrator<2,false>(diff_stack,1.0));
}


void DGDOptimizer::addBoundaryIntegrators(double alpha)
{
	auto &bcs = options["bcs"];
	bndry_marker.resize(bcs.size());
	int idx = 0;
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
          new IsentropicVortexBC<2,false>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
		res_dgd->AddBdrFaceIntegrator(
          new IsentropicVortexBC<2,false>(diff_stack, fec.get(), alpha),
          bndry_marker[idx]);
      idx++;
   }
   if (bcs.find("slip-wall") != bcs.end())
   { // slip-wall boundary condition
      vector<int> tmp = bcs["slip-wall"].get<vector<int>>();
      bndry_marker[idx].SetSize(tmp.size(), 0);
      bndry_marker[idx].Assign(tmp.data());
      res_full->AddBdrFaceIntegrator(
             new SlipWallBC<2,false>(diff_stack, fec.get(), alpha),
             bndry_marker[idx]);
		res_dgd->AddBdrFaceIntegrator(
             new SlipWallBC<2,false>(diff_stack, fec.get(), alpha),
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

void DGDOptimizer::addInterfaceIntegrators(double alpha)
{
	double diss_coeff = options["space-dis"]["iface-coeff"].get<double>();
	res_full->AddInteriorFaceIntegrator(new 
			InterfaceIntegrator<2,false>(diff_stack,diss_coeff,fec.get(),alpha));
	res_dgd->AddInteriorFaceIntegrator(new 
			InterfaceIntegrator<2,false>(diff_stack,diss_coeff,fec.get(),alpha));
}

DGDOptimizer::~DGDOptimizer()
{
   cout << "Deleting the DGD optmization..." << '\n';
}

} // namespace mfem