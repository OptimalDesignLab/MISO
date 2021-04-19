/// Solve the steady isentropic vortex problem on a quarter annulus
#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "parcentgridfunc.hpp"

#include <fstream>
#include <iostream>
#include <random>

#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>

const bool entvar = true;

using namespace std;
using namespace mfem;
using namespace mach;

/// \brief multiply exact solutions for testing
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - conservative variables stored as a 4-vector
void u_const(const mfem::Vector &x, mfem::Vector &u);
void u_poly(const mfem::Vector &x, mfem::Vector &u);
void u_exact(const mfem::Vector &x, mfem::Vector &u);


// This function will be used to check the local R and the assembled prolongation matrix
int main(int argc, char *argv[])
{
	//  1. initialize MPI
	int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options
	const char *mesh_file = "./annulus_1/annulus.smb";
	const char *model_file ="./annulus_1/annulus.dmg";

   OptionsParser args(argc, argv);
   int p = 1;
	int o = 1;
   int rf = 0;
	int pr = 0;
	args.AddOption(&o, "-o", "--order", "order of prolongation matrix");
   args.AddOption(&p,  "-p", "--problem", "which problem to test");
	args.AddOption(&mesh_file, "-m", "--mesh", "mesh file to use");
	args.AddOption(&model_file, "-mo", "--model", "model file to use");
   args.AddOption(&rf, "-r", "--refine", "level of refinement");
	args.AddOption(&pr, "-pr", "--processor", "processor to check info");
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
		PCU_Comm_Init();
		gmi_register_mesh();
		// 3. load apf mesh and ParMesh
		apf::Mesh2* pumi_mesh;
   	pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);
      int dim = pumi_mesh->getDimension();
		pumi_mesh->verify();
		mfem::ParMesh *pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);

		// save the mesh
		string path("/Users/geyan/workspace/mach_dev/build/sandbox");
      pmesh->PrintVTU(path);

		if (pr == myid)
		{
			cout << "\n-------"
				  <<" Mesh Info "
				  << "-------\n";
			cout << setw(15) << "Processor id:" << setw(8) << myid << '\n';
			cout << setw(15) << "mesh dimension:" << setw(8) << dim << '\n';
			cout << setw(15) << "# of elements:" << setw(8) <<pmesh->GetNE() << '\n';
			cout << "-------------------------\n";
		}
	

		// 4. create parallel gd space and regular fespace
		DSBPCollection fec(o, dim);
		ParGDSpace gd(pumi_mesh, pmesh, &fec, dim+2, mfem::Ordering::byVDIM, o, pr);
		ParFiniteElementSpace pfes(pmesh, &fec, dim+2, mfem::Ordering::byVDIM);

		// 5. create the gridfucntions
		mfem::ParCentGridFunction x_cent(&gd);
		mfem::ParGridFunction x(&pfes);
		//mfem::ParGridFunction x_exact(&pfes);
		HypreParMatrix *prolong = gd.Dof_TrueDof_Matrix();
		if (1 == p)
		{
			mfem::VectorFunctionCoefficient u0_fun(dim+2, u_const);
			//x_exact.ProjectCoefficient(u0_fun);
			x_cent.ProjectCoefficient(u0_fun);
			x = 0.0;
		}
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

		// 6. Prolong the solution to real quadrature points
		HypreParMatrix *prolong1 = gd.Dof_TrueDof_Matrix();
		cout << "Get Prolongation matrix, the size is "
			  << prolong->Height() << " x " << prolong->Width() << "\n\n";

		// HypreParVector *x_true = x.GetTrueDofs();
		// cout << "x_true size is " << x_true->Size() << '\n' << '\n';


		// cout << "x_cent ParCentGridFunction size is: " << x_cent.Size()<<'\n';
		// x_cent.Print(cout, 4);
		// HypreParVector *x_cent_true = x_cent.GetTrueDofs();
		// cout << "x_cent_true size is "<<  x_cent_true->Size() << '\n' << '\n';
		// const char *f1 = "x_cent_true";
		// x_cent_true->Print(f1);

		//HypreParVector *x_exact_true = x_exact.GetTrueDofs();
		//cout << "x_exact_true size is " << x_exact_true->Size() << '\n'<<'\n';
		//const char *f2 = "x_exact_true";
		//x_exact_true->Print(f2);

		// prolong->Mult(*x_cent_true, *x_true);
		// cout << "prolonged.\n";
		// 7. compute the difference
		// double norm;
		// double loc_norm = (*diff) * (*diff);
		// MPI_Allreduce(&loc_norm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		// norm = sqrt(norm);

		// if ( 0 == myid)
		// {
		// 	cout << "The projection norm is " << norm << '\n';
		// }
       delete pmesh;
	   pumi_mesh->destroyNative();
		apf::destroyMesh(pumi_mesh);
		PCU_Comm_Free();
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
	u(0) = 1.0;
	u(1) = 2.0;
	u(2) = 3.0;
	u(3) = 4.0;
}

void u_poly(const mfem::Vector &x, mfem::Vector &u)
{
   u = 0.0;
   for (int i = 2; i >= 0; i--)
   {
      u(0) += pow(x(0), i);
		u(1) += pow(x(1), i);
		u(2) += pow(x(0) + x(1), i);
		u(3) += pow(x(0) - x(1), i);
   }
}