/// Solve the steady isentropic vortex problem on a quarter annulus
#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "parcentgridfunc.hpp"

#include <fstream>
#include <iostream>
#include <random>

const bool entvar = false;

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief multiply exact solutions for testing
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void u_const(const mfem::Vector &x, mfem::Vector &u);
void u_poly(const mfem::Vector &x, mfem::Vector &u);
void u_exact(const mfem::Vector &x, mfem::Vector &u);
void u_const_single(const mfem::Vector &x, mfem::Vector &u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
Mesh buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang);


// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
	//  1. initialize MPI
	int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   OptionsParser args(argc, argv);
   int p = 1;
	int o = 1;
	int pr = 0;
	int num_state = 1;
	int nx = 4;
	int ny = 4;
	args.AddOption(&o, "-o", "--order", "order of prolongation matrix");
   args.AddOption(&p,  "-p", "--problem", "which problem to test");
	args.AddOption(&pr, "-c", "--processor", "processor to check info");
	args.AddOption(&nx, "-nr", "--nrad", "processor to check info");
	args.AddOption(&ny, "-nt", "--ntang", "processor to check info");
	args.AddOption(&num_state, "-v", "--vdim", "vdim");
   args.Parse();

   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }

   try
   {
		// construct the mesh
		int degree = o+1;
      unique_ptr<Mesh> mesh(new Mesh(buildQuarterAnnulusMesh(degree, nx, ny)));
		int dim = mesh->Dimension();
		int nel = mesh->GetNE();

		// generate user-defined partitioning
		int partitioning[nel];
		int nel_par = (int)floor(nel/num_procs);
		int nel_left = nel - nel_par * num_procs;
		int nel_par_a = nel_par;

		for (int i = 0; i < num_procs; i++)
		{
			if (i == num_procs - 1) nel_par_a += nel_left;
			for (int j = 0; j < nel_par_a; j++)
			{
				partitioning[i*nel_par+j] = i;
			}
		}


		// print the mesh in serial		
		ofstream sol_ofs("par_galerkin_mesh.vtk");
		sol_ofs.precision(14);
		mesh->PrintVTK(sol_ofs,0);

		unique_ptr<ParMesh> pmesh(new ParMesh(MPI_COMM_WORLD, *mesh, partitioning));

		// save the mesh in parallel
		string path("/Users/geyan/workspace/mach_dev/build/sandbox");
      pmesh->PrintVTU(path);

		long glb_id = pmesh->GetGlobalElementNum(0);
		if (pr == myid)
		{
			cout << "\n-------"
				  <<" Mesh Info "
				  << "-------\n";
			cout << setw(15) << "total # elmts:" << setw(8) << mesh->GetNE() << '\n';
			cout << setw(15) << "Processor id:" << setw(8) << myid << '\n';
			cout << setw(15) << "mesh dimension:" << setw(8) << dim << '\n';
			cout << setw(15) << "# of elements:" << setw(8) <<pmesh->GetNE() << '\n';
			cout << setw(15) << "Check ordering: " << endl;
			Vector cent1(dim), cent2(dim);
			int geom;
			ElementTransformation *eltransf;
			for (int i = 0; i < pmesh->GetNE(); i++)
			{
				glb_id = pmesh->GetGlobalElementNum(i);
				cout << i << " --> " << glb_id;
				geom = pmesh->GetElement(i)->GetGeometryType();
				eltransf = pmesh->GetElementTransformation(i);
				eltransf->Transform(Geometries.GetCenter(geom),cent2);
				eltransf = mesh->GetElementTransformation((int)glb_id);
				eltransf->Transform(Geometries.GetCenter(geom),cent1);
				cout << ",  ("<< cent1(0) - cent2(0) << ", " << cent1(1) - cent2(1) << ")\n";
			}
			cout << "-------------------------\n";

		}


		DSBPCollection fec(o,dim);
		FiniteElementSpace serial_fes(mesh.get(),&fec, num_state, mfem::Ordering::byVDIM);
		ParGDSpace pgd(mesh.get(), pmesh.get(), &serial_fes, partitioning, &fec,
							num_state,mfem::Ordering::byVDIM, o, pr);
		int dof_offset = pgd.GetMyTDofOffset();
		if (pr == myid)
		{
			Array<int> el_dofs1, el_dofs2;
			long glb_dof;
			cout << "\n-----------------------------------"
			<<" FiniteElementSpace Info "
			<< "-----------------------------------\n";
			cout << setw(3) << "id" << setw(30) << "dofs from serial mesh "
				  << setw(20) <<"local dofs order" << setw(30) << "global dof order\n";
			for (int i = 0; i < pmesh->GetNE(); i++)
			{
				glb_id = pmesh->GetGlobalElementNum(i);
				serial_fes.GetElementVDofs(glb_id, el_dofs1);
				pgd.GetElementVDofs(i,el_dofs2);
				cout << setw(3) << glb_id << ": (";
				for (int k=0; k<el_dofs1.Size(); k++)
				{
					cout << el_dofs1[k] << ' ';
				}
				cout << ") --> (";
				for (int k=0; k<el_dofs2.Size(); k++)
				{
					cout << dof_offset + el_dofs2[k] << ' ';
				}
				cout << ") --> (";
				for (int k=0; k<el_dofs2.Size(); k++)
				{
					cout << pgd.GetGlobalTDofNumber(el_dofs2[k]) << ' ';
				}
				cout << ")\n";
			}
			cout << "----------------------------------------------------"
				  << "---------------------------------------------\n";
		}
		
	

		// // 4. create parallel gd space and regular fespace
		// DSBPCollection fec(o, dim);
		// ParGDSpace gd(pumi_mesh, pmesh, &fec, vdim, mfem::Ordering::byVDIM, o, pr);
		// ParFiniteElementSpace pfes(pmesh, &fec, vdim, mfem::Ordering::byVDIM);

		// // 5. create the gridfucntions
		// mfem::ParCentGridFunction x_cent(&gd);
		// mfem::ParGridFunction x(&pfes);
		// mfem::ParGridFunction x_exact(&pfes);
		// if (1 == p)
		// {
		// 	if( 1 == vdim)
		// 	{
		// 		mfem::VectorFunctionCoefficient u0_fun(1, u_const_single);
		// 		x_exact.ProjectCoefficient(u0_fun);
		// 		x_cent.ProjectCoefficient(u0_fun);
		// 		x = 0.0;
		// 	}
		// 	else
		// 	{
		// 		mfem::VectorFunctionCoefficient u0_fun(dim+2, u_const);
		// 		x_exact.ProjectCoefficient(u0_fun);
		// 		x_cent.ProjectCoefficient(u0_fun);
		// 		x = 0.0;
		// 	}

		// }
		// else if (2 == p)
		// {
		// 	mfem::VectorFunctionCoefficient u0_fun(dim+2, u_poly);
		// 	x_exact.ProjectCoefficient(u0_fun);
		// 	x_cent.ProjectCoefficient(u0_fun);
		// 	x = 0.0;
		// }
		// else if(3 == p)
		// {
		// 	mfem::VectorFunctionCoefficient u0_fun(dim+2, u_exact);
		// 	x_exact.ProjectCoefficient(u0_fun);
		// 	x_cent.ProjectCoefficient(u0_fun);
		// 	x = 0.0;
		// }

		// // 6. Prolong the solution to real quadrature points
		// HypreParMatrix *prolong = gd.Dof_TrueDof_Matrix();
		// cout << "Get Prolongation matrix, the size is "
		// 	  << prolong->Height() << " x " << prolong->Width() << "\n\n";
		
		// cout << "x_cent size is: " << x_cent.Size()<<'\n';
		// HypreParVector *x_cent_true = x_cent.GetTrueDofs();
		// cout << "x_cent_true size is "<<  x_cent_true->Size() << '\n' << '\n';
		// const char *f1 = "x_cent_true";
		// x_cent_true->Print(f1);

		// HypreParVector *x_true = x.GetTrueDofs();
		// cout << "x_true size is " << x_true->Size() << '\n' << '\n';


		// HypreParVector *x_exact_true = x_exact.GetTrueDofs();
		// cout << "x_exact_true size is " << x_exact_true->Size() << '\n'<<'\n';
		// const char *f2 = "x_exact_true";
		// x_exact_true->Print(f2);

		// prolong->Mult(*x_cent_true, *x_true);
		// cout << "prolonged.\n";
		// const char *f3 = "x_true";
		// x_true->Print(f3);

		// // 7. compute the difference
		// x.SetFromTrueDofs(*x_true);
		// x_exact.Add(-1.0, x);
		// double loc_norm = x_exact.Norml2();
		// double norm;
		// MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		// if ( 0 == myid)
		// {
		// 	cout << "The projection norm is " << norm << '\n';
		// }

   }

   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
	MPI_Finalize();
   return 0;
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void u_exact(const mfem::Vector &x, mfem::Vector &q)
{
	int dim = x.Size();
	mfem::Vector u(dim+2);
	q.SetSize(dim+2);

   double ri = 1.0;
   double Mai = 0.5; //0.95 
   double rhoi = 2.0;
   double prsi = 1.0/euler::gamma;
   double rinv = ri/sqrt(x(0)*x(0) + x(1)*x(1));
   double rho = rhoi*pow(1.0 + 0.5*euler::gami*Mai*Mai*(1.0 - rinv*rinv),
                         1.0/euler::gami);
   double Ma = sqrt((2.0/euler::gami)*( ( pow(rhoi/rho, euler::gami) ) * 
                    (1.0 + 0.5*euler::gami*Mai*Mai) - 1.0 ) );
   double theta;
   if (x(0) > 1e-15)
   {
      theta = atan(x(1)/x(0));
   }
   else
   {
      theta = M_PI/2.0;
   }
   double press = prsi* pow( (1.0 + 0.5*euler::gami*Mai*Mai) / 
                 (1.0 + 0.5*euler::gami*Ma*Ma), euler::gamma/euler::gami);
   double a = sqrt(euler::gamma*press/rho);

   u(0) = rho;
   u(1) = -rho*a*Ma*sin(theta);
   u(2) = rho*a*Ma*cos(theta);
   u(3) = press/euler::gami + 0.5*rho*a*a*Ma*Ma;

   if (entvar == false)
   {
      q = u;
   }
   else
   {
      calcEntropyVars<double, 2>(u.GetData(), q.GetData());
   }
}

void u_const(const mfem::Vector &x, mfem::Vector &u)
{
	u.SetSize(x.Size()+2);
	u(0) = 1.0;
	u(1) = 2.0;
	u(2) = 3.0;
	u(3) = 4.0;
}

void u_poly(const mfem::Vector &x, mfem::Vector &u)
{
	u.SetSize(x.Size()+2);
   u = 0.0;
   for (int i = 2; i >= 0; i--)
   {
      u(0) += pow(x(0), i);
		u(1) += pow(x(1), i);
		u(2) += pow(x(0) + x(1), i);
		u(3) += pow(x(0) - x(1), i);
   }
}

void u_const_single(const mfem::Vector &x, mfem::Vector &u)
{
	u.SetSize(1);
	u(0) = 5.0;
}

Mesh buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   Mesh mesh = Mesh::MakeCartesian2D(num_rad, num_ang, Element::TRIANGLE,
                                     true /* gen. edges */, 2.0, M_PI*0.5,
                                     true);
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 2,
                                                    Ordering::byVDIM);

   // This lambda function transforms from (r,\theta) space to (x,y) space
   auto xy_fun = [](const Vector& rt, Vector &xy)
   {
      xy(0) = (rt(0) + 1.0)*cos(rt(1)); // need + 1.0 to shift r away from origin
      xy(1) = (rt(0) + 1.0)*sin(rt(1));
   };
   VectorFunctionCoefficient xy_coeff(2, xy_fun);
   GridFunction *xy = new GridFunction(fes);
   xy->MakeOwner(fec);
   xy->ProjectCoefficient(xy_coeff);

   mesh.NewNodes(*xy, true);
   return mesh;
}