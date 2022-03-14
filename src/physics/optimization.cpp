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


DGDOptimizer::DGDOptimizer(Vector init, const string &opt_file_name,
									unique_ptr<Mesh> smesh)
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
}


DGDOptimizer::~DGDOptimizer()
{
    cout << "Deleting the DGD optmization..." << '\n';
}


double DGDOptimizer::GetEnergy(const Vector &x) const
{
	Vector r(FullSize);
	SparseMatrix *prolong = fes_dgd->GetCP();
	prolong->Mult(*u_dgd,*u_full); 
	res_full->Mult(*u_full,r);
	return r * r;
}

void DGDOptimizer::Mult(const Vector &x, Vector &y) const
{
	// dJ/dc = pJ/pc - pJ/puc * (pR_dgd/puc)^{-1} * pR_dgd/pc

	y.SetSize(numDesignVar); // set y as pJpc
	Vector pJpuc(numBasis);
	
	/// first compute some variables that used multiple times
	// 1. get pRpu, pR_dgd/pu_dgd
	SparseMatrix *pRpu = dynamic_cast<SparseMatrix*>(&res_full->GetGradient(*u_full));
	SparseMatrix *pR_dgdpuc = dynamic_cast<SparseMatrix*>(&res_dgd->GetGradient(*u_dgd));

	// 2. compute full residual
	Vector r(FullSize);
	res_full->Mult(*u_full,r);

	/// loop over all design variables
	Vector ppupc_col(FullSize);
	Vector dptpc_col(numBasis);
	SparseMatrix dPdci(FullSize,numBasis);
	DenseMatrix pPupc(FullSize,numDesignVar);
	DenseMatrix pPtpcR(numBasis,numDesignVar);
	for (int i = 0; i < numDesignVar; i++)
	{
		// get dpdc
		fes_dgd->GetdPdc(i,dPdci);

		// colume of intermediate pPu/pc
		dPdci.Mult(*u_dgd,ppupc_col);
		pPupc.SetCol(i,ppupc_col);

		// colume of pPt / pc * R
		dPdci.MultTranspose(r,dptpc_col);
		pPtpcR.SetCol(i,dptpc_col);

		// clean data in dPdc
		dPdci.Clear();
	}

	// compute pJ/pc
	Vector temp_vec1(FullSize);
	pRpu->MultTranspose(r,temp_vec1);
	pPupc.MultTranspose(temp_vec1,y);
	y *= 2.0;

	// compute pJ/puc
	SparseMatrix *P = fes_dgd->GetCP();
	P->MultTranspose(temp_vec1,pJpuc);


	// compute pR_dgd / pc
	DenseMatrix *temp_mat1 = ::Mult(*pRpu,pPupc);

	SparseMatrix *Pt = Transpose(*P);
	DenseMatrix *pR_dgdpc = ::Mult(*Pt,*temp_mat1);
	*pR_dgdpc += pPtpcR;

	// solve for adjoint variable
	Vector adj(numBasis);
	SparseMatrix *pRt_dgdpuc = Transpose(*pR_dgdpuc);
	UMFPackSolver umfsolver;
	umfsolver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
	umfsolver.SetPrintLevel(1);
	umfsolver.SetOperator(*pRt_dgdpuc);
	umfsolver.Mult(pJpuc,adj);
	// DenseMatrixInverse pR_dgdpuc_inv(pR_dgdpuc);
	// pR_dgdpuc_inv.Mult(pJpuc,adj);


	// compute the total derivative
	Vector temp_vec2(numDesignVar);
	pR_dgdpc->Transpose();
	pR_dgdpc->Mult(adj,temp_vec2);
	y -= temp_vec2;

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
	int idx = 0;
   if (bcs.find("vortex") != bcs.end())
   { // isentropic vortex BC
      if (dim != 2)
      {
         throw MachException("EulerSolver::addBoundaryIntegrators(alpha)\n"
                             "\tisentropic vortex BC must use 2D mesh!");
      }
      vector<int> tmp = bcs["vortex"].template get<vector<int>>();
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
      vector<int> tmp = bcs["slip-wall"].template get<vector<int>>();
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
}

// void DGDOptimizer::addOutputs()
// {
// 	auto &fun = options["outputs"];
// 	using json_iter = nlohmann::json::iterator;
// 	int num_bndry_ouputs = 0;
// 	for (json_iter it = fun.begin(); it != fun.end(); ++it)
// 	{
// 		if (it->is_array()) ++num_bndry_outputs;
// 	}
// 	output_bndry_marker.resize(num_bndry_outputs);
// 	int idx = 0;
//    if (fun.find("drag") != fun.end())
//    { 
//       // drag on the specified boundaries
//       vector<int> tmp = fun["drag"].get<vector<int>>();
//       output_bndry_marker[idx].SetSize(tmp.size(), 0);
//       output_bndry_marker[idx].Assign(tmp.data());
//       output.emplace("drag", fes.get());
//       mfem::Vector drag_dir(dim);
//       drag_dir = 0.0;
//       if (dim == 1)
//       {
//          drag_dir(0) = 1.0;
//       }
//       else 
//       {
//          drag_dir(iroll) = cos(aoa_fs);
//          drag_dir(ipitch) = sin(aoa_fs);
//       }
//       output.at("drag").AddBdrFaceIntegrator(
//           new PressureForce<dim, entvar>(diff_stack, fec.get(), drag_dir),
//           output_bndry_marker[idx]);
//       idx++;
//    }
//    if (fun.find("lift") != fun.end())
//    { 
//       // lift on the specified boundaries
//       vector<int> tmp = fun["lift"].template get<vector<int>>();
//       output_bndry_marker[idx].SetSize(tmp.size(), 0);
//       output_bndry_marker[idx].Assign(tmp.data());
//       output.emplace("lift", fes.get());
//       mfem::Vector lift_dir(dim);
//       lift_dir = 0.0;
//       if (dim == 1)
//       {
//          lift_dir(0) = 0.0;
//       }
//       else
//       {
//          lift_dir(iroll) = -sin(aoa_fs);
//          lift_dir(ipitch) = cos(aoa_fs);
//       }
//       output.at("lift").AddBdrFaceIntegrator(
//           new PressureForce<dim, entvar>(diff_stack, fec.get(), lift_dir),
//           output_bndry_marker[idx]);
//       idx++;
//    }
//    if (fun.find("entropy") != fun.end())
//    {
//       // integral of entropy over the entire volume domain
//       output.emplace("entropy", fes.get());
//       output.at("entropy").AddDomainIntegrator(
//          new EntropyIntegrator<dim, entvar>(diff_stack));
//    }
// }

} // namespace mfem