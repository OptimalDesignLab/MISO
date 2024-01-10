#include "linear_optimization.hpp"
#include "default_options.hpp"

using namespace std;
using namespace mfem;
using namespace mach;

namespace mach
{

LinearOptimizer::LinearOptimizer(Vector init,
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
	mesh = std::move(smesh);
	dim = mesh->Dimension();
	num_state = 1;
	cout << "Number of elements: " << mesh->GetNE() << '\n';

	// construct fespaces
	int dgd_degree = options["space-dis"]["DGD-degree"].get<int>();
	int extra = options["space-dis"]["extra-basis"].get<int>();
	fec.reset(new DG_FECollection(options["space-dis"]["degree"].get<int>(),dim, BasisType::GaussLobatto));
	fes_dgd.reset(new DGDSpace(mesh.get(),fec.get(),designVar,dgd_degree,extra,
							num_state,Ordering::byVDIM));
	fes_full.reset(new FiniteElementSpace(mesh.get(),fec.get(),num_state,
							 Ordering::byVDIM));

	// construct the gridfunction
	u_dgd.reset(new CentGridFunction(fes_dgd.get()));
	u_full.reset(new GridFunction(fes_full.get()));

	// variable size
	ROMSize = u_dgd->Size();
	FullSize = u_full->Size();
	numDesignVar = designVar.Size();
	numBasis = numDesignVar/dim;

	// construct the residual forms
	res_dgd.reset(new BilinearForm(fes_dgd.get()));
	res_full.reset(new BilinearForm(fes_full.get()));
	b_dgd.SetSize(fes_dgd->GetTrueVSize());
	b_full.reset(new LinearForm(fes_full.get()));

	// check some intermediate info
  cout << "Num of state variables: " << num_state << '\n';
  cout << "dgd_degree is: " << dgd_degree << '\n';
  cout << "u_dgd size is " << u_dgd->Size() << '\n';
  cout << "u_full size is " << u_full->Size() << '\n';
  cout << "Full size model is: "<< fes_full->GetTrueVSize() << '\n';
  cout << "DGD model size is (should be number of basis): " << num_state * dynamic_cast<DGDSpace *>(fes_dgd.get())->GetNDofs() << '\n';
  cout << "res_full size is " << res_full->Height() << " x " << res_full->Width() << '\n';
	cout << "res_dgd size is " << res_dgd->Height() << " x " << res_dgd->Width() << '\n';
}

void LinearOptimizer::InitializeSolver(VectorFunctionCoefficient& velocity, FunctionCoefficient& inflow)
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
	SparseMatrix* p = fes_dgd->GetCP();
	p->MultTranspose(*b_full, b_dgd);
		
  //  Get operators in handy
	k_full = &res_full->SpMat();
	k_dgd = &res_dgd->SpMat();
}

void LinearOptimizer::Mult(const mfem::Vector &x, mfem::Vector& y) const
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
	Vector r(FullSize);
	k_full->Mult(*u_full,r);
	r -= *b_full;
	cout << "f1\n";

	// ofstream r_save("r_full.txt");
	// r.Print(r_save,1);
	// r_save.close();

	/// loop over all design variables
	Vector ppupc_col(FullSize);
	Vector dptpc_col(ROMSize);
	
	DenseMatrix pPupc(FullSize,numDesignVar);
	DenseMatrix pPtpcR(ROMSize,numDesignVar);
	for (int i = 0; i < numDesignVar; i++)
	{
		SparseMatrix *dPdci = new SparseMatrix(FullSize,ROMSize);
		// get dpdc
		fes_dgd->GetdPdc(i,x,*dPdci);

		// colume of intermediate pPu/pc
		dPdci->Mult(*u_dgd,ppupc_col);
		pPupc.SetCol(i,ppupc_col);

		// colume of pPt / pc * R
		dPdci->MultTranspose(r,dptpc_col);
		pPtpcR.SetCol(i,dptpc_col);
		delete dPdci;
	}
	cout << "f2\n";

	// ofstream ppupc_save("ppupc.txt");
	// pPupc.PrintMatlab(ppupc_save);
	// ppupc_save.close();

	// ofstream pptpcr_save("pptpcr.txt");
	// pPtpcR.PrintMatlab(pptpcr_save);
	// pptpcr_save.close();

	// compute pJ/pc
	Vector temp_vec1(FullSize);
	pRpu->MultTranspose(r,temp_vec1);
	pPupc.MultTranspose(temp_vec1,y);
	y *= 2.0;
	cout << "f3\n";

	// ofstream pjpc_save("pjpc.txt");
	// y.Print(pjpc_save,1);
	// pjpc_save.close();

	// compute pJ/puc
	SparseMatrix* p = fes_dgd->GetCP();
	p->MultTranspose(temp_vec1,pJpuc);
	pJpuc *= 2.0;
	cout << "f4\n";

	// ofstream p_save("p.txt");
	// P->PrintMatlab(p_save);
	// p_save.close();

	// ofstream pjpuc_save("pjpuc.txt");
	// pJpuc.Print(pjpuc_save,1);
	// pjpuc_save.close();

	// compute pR_dgd / pc
	DenseMatrix *temp_mat1 = ::Mult(*pRpu,pPupc);
	SparseMatrix *Pt = Transpose(*p);
	DenseMatrix *pR_dgdpc = ::Mult(*Pt,*temp_mat1);
	delete Pt;
	*pR_dgdpc += pPtpcR;
	delete temp_mat1;
	cout << "f5\n";

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
	umfsolver.Mult(pJpuc,adj);
	delete pRt_dgdpuc;
	cout << "f6\n";

	// ofstream adj_sasve("adj.txt");
	// adj.Print(adj_save,1);
	// adj_save.close();


	// compute the total derivative
	Vector temp_vec2(numDesignVar);
	pR_dgdpc->Transpose();
	pR_dgdpc->Mult(adj,temp_vec2);
	y -= temp_vec2;
	cout << "f7\n";

	// ofstream djdc_save("djdc.txt");
	// y.Print(djdc_save,1);
	// djdc_save.close();
	
	delete pR_dgdpc;	
}

double LinearOptimizer::GetEnergy(const mfem::Vector &x) const
{
		// build new DGD operators
	fes_dgd->buildProlongationMatrix(x);


	UMFPackSolver umfsolver;
	umfsolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
	umfsolver.SetPrintLevel(1);
	umfsolver.SetOperator(*k_dgd);
	umfsolver.Mult(b_dgd, *u_dgd);


	SparseMatrix* p = fes_dgd->GetCP();
	p->Mult(*u_dgd,*u_full); 
	Vector r(FullSize);
	k_full->Mult(*u_full,r);
	return r * r;
}

} // namespace mach