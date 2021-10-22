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
void u_linear(const mfem::Vector  &x, mfem::Vector &u);
void u_poly(const mfem::Vector &x, mfem::Vector &u);
void u_exact(const mfem::Vector &x, mfem::Vector &u);
void u_exact_unsteady(const mfem::Vector &x, mfem::Vector &u);
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
      //unique_ptr<Mesh> mesh(new Mesh(buildQuarterAnnulusMesh(degree, nx, ny)));
		unique_ptr<Mesh> mesh(new Mesh("periodic_triangle1.mesh",1,1));
		for (int l = 0; l < 6; l++)
      {
         mesh->UniformRefinement();
      }
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

		unique_ptr<ParMesh> pmesh(new ParMesh(MPI_COMM_WORLD, *mesh, partitioning));

		long glb_id = pmesh->GetGlobalElementNum(0);
		if (pr == myid)
		{
			cout << "\n-------"
				  <<" Mesh Info "
				  << "-------\n";
			cout << setw(15) << "Processor id:" << setw(8) << myid << '\n';
			cout << setw(15) << "mesh dimension:" << setw(8) << dim << '\n';
			cout << setw(15) << "total # elmts:" << setw(8) << mesh->GetNE() << '\n';
			cout << setw(15) << "local # elmts:" << setw(8) << pmesh->GetNE() << '\n';
			cout << "-------------------------\n";

		}


		DSBPCollection fec(o,dim);
		FiniteElementSpace serial_fes(mesh.get(),&fec, num_state, mfem::Ordering::byVDIM);
		ParFiniteElementSpace pfes(pmesh.get(),&fec, num_state, mfem::Ordering::byVDIM);
		ParGDSpace pgd(mesh.get(), pmesh.get(), &fec,
							num_state,mfem::Ordering::byVDIM, o, pr);
		int tdof_offset = pgd.GetMyTDofOffset();
		int dof_offset = pgd.GetMyDofOffset();
		if (pr == myid)
		{

			Array<int> el_dofs1, el_dofs2;
			long glb_dof;
			cout << "\n-----------------------------------"
			<<" FiniteElementSpace Info "
			<< "-----------------------------------\n";
			cout << "pfes.GetVDim()  = " << pfes.GetVDim() << endl;
			cout << "pfes.GetVSize() = " << pfes.GetVSize() << endl;
			cout << "pfes.GetNDofs() = " << pfes.GetNDofs() << endl;
			cout << "pfes.GetTrueVSize() = " << pfes.GetTrueVSize() << endl;
			HYPRE_BigInt *offsets0 = pfes.GetDofOffsets();
			cout << "pfes offsets are " << offsets0[0] << ", "<< offsets0[1] << endl;
			HYPRE_BigInt *offsets1 = pfes.GetTrueDofOffsets();
			cout << "pfes true offsets are " << offsets1[0] << ", "<< offsets1[1] << endl;
			cout << "pgd.GetVDim()   = " << pgd.GetVDim() << endl;
			cout << "pgd.GetVSize()  = " << pgd.GetVSize() << endl;
			cout << "pgd.GetNDofs()  = " << pgd.GetNDofs() << endl;
			cout << "pgd.GetTrueVSize() = " << pgd.GetTrueVSize() << endl;
			HYPRE_BigInt *offsets2 = pgd.GetDofOffsets();
			cout << "pgd offsets are " << offsets2[0] << ", "<< offsets2[1] << endl;
			HYPRE_BigInt *offsets3 = pgd.GetTrueDofOffsets();
			cout << "pgd true offsets are " << offsets3[0] << ", "<< offsets3[1] << endl;
			cout << "pgd tdof_offset is " << tdof_offset << endl;
			cout << "pgd dof_offset is " << dof_offset << endl; 
			cout << "----------------------------------------------------"
				  << "---------------------------------------------\n";
		}
		

		// 5. create the gridfucntions
		mfem::ParCentGridFunction x_cent(&pgd, pr);
		mfem::ParGridFunction x(&pfes);
		mfem::ParGridFunction x_exact(&pfes);

		if (pr == myid)
		{
			cout << "ParCentGridFunction x_cent size is " << x_cent.Size() << endl;
			cout << "ParGridFunction x size is " << x.Size() << endl;
			cout << "ParGridFunction x_exact size is " << x_exact.Size() << endl;
		}
		if (1 == p)
		{
			if( 1 == num_state)
			{
				mfem::VectorFunctionCoefficient u0_fun(1, u_const_single);
				x_exact.ProjectCoefficient(u0_fun);
				x_cent.ProjectCoefficient(u0_fun);
				x = 0.0;
			}
			else
			{
				mfem::VectorFunctionCoefficient u0_fun(dim+2, u_const);
				x_exact.ProjectCoefficient(u0_fun);
				x_cent.ProjectCoefficient(u0_fun);
				x = 0.0;
			}

		}
		else if (2 == p)
		{
			mfem::VectorFunctionCoefficient u0_fun(dim+2, u_linear);
			x_exact.ProjectCoefficient(u0_fun);
			x_cent.ProjectCoefficient(u0_fun);
			x = 0.0;
		}
		else if(3 == p)
		{
			mfem::VectorFunctionCoefficient u0_fun(dim+2, u_exact);
			x_exact.ProjectCoefficient(u0_fun);
			x_cent.ProjectCoefficient(u0_fun);
			x = 0.0;
		}
		else if(4 == p)
		{
			mfem::VectorFunctionCoefficient u0_fun(dim+2, u_exact_unsteady);
			x_exact.ProjectCoefficient(u0_fun);
			x_cent.ProjectCoefficient(u0_fun);
			x = 0.0;
		}

		if (pr == myid)
		{
			cout << "---------------Check projection---------------\n";
			cout << "x_exact size is : " << x_exact.Size() << endl;
			cout << "x_center is: " << x_cent.Size() << endl;
			cout << "x size is " << x.Size() << endl;
			cout << "----------------------------------------------\n";
		}
		MPI_Barrier(MPI_COMM_WORLD);
		// 6. Prolong the solution to real quadrature points
		HypreParMatrix *prolong = pgd.Dof_TrueDof_Matrix();
		HypreParVector *x_cent_true = x_cent.GetTrueDofs();
		HypreParVector *x_exact_true = x_exact.GetTrueDofs();
		HypreParVector *x_true = x.GetTrueDofs();
		
		if (myid == pr)
		{
			cout << "Get Prolongation matrix, the size is "
				<< prolong->Height() << " x " << prolong->Width() << "\n";
			cout << "Global size is " << prolong->GetGlobalNumRows() << " x "
				  << prolong->GetGlobalNumCols()  << endl;
			cout << "x_exact_true size is " << x_exact_true->Size() << endl;
			cout << "x_cent_true size is "<<  x_cent_true->Size() << '\n';
			cout << "x_true size is " << x_true->Size() << '\n';
			
		}

		x_exact_true->Print("x_exact_true");
		x_cent_true->Print("x_cent_true");

		prolong->Mult(*x_cent_true, *x_true);
		x_true->Print("x_true");
		x.SetFromTrueDofs(*x_true);


		// 7. compute the differences
		x.Add(-1.0, x_exact);
		double loc_norm = x.Norml2();
		double loc_norm_inf = x.Normlinf();
		double norm;
		double norminf;
		MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&loc_norm_inf, &norminf, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if ( 0 == myid)
		{
			cout << "The projection norm is " << norm << endl;
			cout << "The inf norm is " << norminf << endl;
		}

		// 8. save the result
		{
			std::vector<ParGridFunction*> fields{&x_exact,&x};
			std::vector<std::string> names{"u_exact", "error"};
			ParaViewDataCollection paraview_dc("par_gd_check", pmesh.get());
   		paraview_dc.SetPrefixPath("par_gd_test");
			paraview_dc.SetLevelsOfDetail(1);
			paraview_dc.SetCycle(0.0);
			paraview_dc.SetDataFormat(VTKFormat::BINARY);
			paraview_dc.SetHighOrderOutput(true);
			paraview_dc.SetTime(0.0);  // set the time
			for (unsigned i = 0; i < fields.size(); ++i)
			{
				paraview_dc.RegisterField(names[i], fields[i]);
			}
			paraview_dc.Save();
		}

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

void u_exact_unsteady(const Vector &x, Vector& q)
{
   q.SetSize(4);
   Vector u0(4);
   double t = 0.0; // this could be an input...
   double x0 = 0.5;
   double y0 = 0.5;
   // scale is used to reduce the size of the vortex; need to scale the
   // velocity and pressure carefully to account for this
   double scale = 15.0;
   double xi = (x(0) - x0)*scale - t;
   double eta = (x(1) - y0)*scale;
   double M = 0.5;
   double epsilon = 1.0;
   double f = 1.0 - (xi*xi + eta*eta);
   // density
   u0(0) = pow(1.0 - epsilon*epsilon*euler::gami*M*M*exp(f)/(8*M_PI*M_PI),
               1.0/euler::gami);
   // x velocity
   u0(1) = 1.0 - epsilon*eta*exp(f*0.5)/(2*M_PI);
   u0(1) *= scale*u0(0);
   // y velocity
   u0(2) = epsilon*xi*exp(f*0.5)/(2*M_PI);
   u0(2) *= scale*u0(0);
   // pressure, used to get the energy
   double press = pow(u0(0), euler::gamma)/(euler::gamma*M*M);
   press *= scale*scale;
   u0(3) = press/euler::gami + 0.5*(u0(1)*u0(1) + u0(2)*u0(2))/u0(0);
   if (entvar == false)
   {
      q = u0;
   }
   else
   {
      calcEntropyVars<double, 2>(u0.GetData(), q.GetData());
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

void u_linear(const mfem::Vector &x, mfem::Vector &u)
{
	u.SetSize(x.Size()+2);
	u(0) = 1.0;
	u(1) = 2.0;
	u(2) = x(0);
	u(3) = x(1)*x(1);
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