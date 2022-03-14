#include "mfem.hpp"
#include "euler.hpp"
#include "galer_diff.hpp"
#include "rbfgridfunc.hpp"
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace mfem;
using namespace mach;


std::default_random_engine gen(std::random_device{}());
std::uniform_real_distribution<double> normal_rand(0.0,1.0);

/// \brief Defines the exact solution for the steady isentropic vortex
/// \param[in] x - coordinate of the point at which the state is needed
/// \param[out] u - state variables stored as a 4-vector
void uexact(const Vector &x, Vector& u);
void upoly(const Vector &x, Vector& u);
void utest(const Vector &x, Vector &u);
void usingle(const mfem::Vector &x, mfem::Vector &u);

/// Generate quarter annulus mesh 
/// \param[in] degree - polynomial degree of the mapping
/// \param[in] num_rad - number of nodes in the radial direction
/// \param[in] num_ang - number of nodes in the angular direction
std::unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad,
                                              int num_ang);
mfem::Vector buildBasisCenters(int, int);

template<typename T>
void writeBasisCentervtp(const mfem::Vector &q, T& stream);

int main(int argc, char *argv[])
{
   const char *options_file = "galerkin_difference.json";
   int myid = 0;
   // Parse command-line options
   OptionsParser args(argc, argv);
   int degree = 1;
   int nx = 1;
   int ny = 1;
   int numRad = 10;
   int numTheta = 10;
   int extra = 1;
   args.AddOption(&options_file, "-o", "--options",
                  "Options file to use.");
   args.AddOption(&degree, "-d", "--degree", "poly. degree of mesh mapping");
   args.AddOption(&nx, "-nr", "--num-rad", "number of radial segments");
   args.AddOption(&ny, "-nt", "--num-theta", "number of angular segments");
   args.AddOption(&numRad,"-br","--numrad","number of radius points");
   args.AddOption(&numTheta,"-bt","--numtheta","number of anglular points");
   args.AddOption(&extra,"-e","--extra","number of anglular points");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   try
   {
      // generate the mesh
      unique_ptr<Mesh> smesh = buildQuarterAnnulusMesh(degree+1, nx, ny);
      ofstream savevtk("dgd_test.vtk");
      smesh->PrintVTK(savevtk,0);
      savevtk.close();
      std::cout << "Number of elements " << smesh->GetNE() <<'\n';
      int dim = smesh->Dimension();
      int num_state = dim+2;

      // initialize the basis centers
      int numBasis = smesh->GetNE();
      Vector center(2*numBasis);
      Vector loc(dim);
      for (int k = 0; k < numBasis; k++)
      {  
         smesh->GetElementCenter(k,loc);
         center(k*2) = loc(0);
         center(k*2+1) = loc(1);         
      }

      // Array<Vector *> center = buildBasisCenters(numRad,numTheta);
      // int numBasis = numRad * numTheta;
      ofstream centerwrite("center.vtp");
      writeBasisCentervtp(center, centerwrite);
      centerwrite.close();
      
      // initialize the fe collection and rbf space
      DSBPCollection fec(degree,smesh->Dimension());
      DGDSpace dgdSpace(smesh.get(),&fec,center,degree,extra,num_state,Ordering::byVDIM);
      FiniteElementSpace fes(smesh.get(),&fec,num_state,Ordering::byVDIM);

      //================== Construct the gridfunction and apply the exact solution =======
      mfem::VectorFunctionCoefficient u0_fun(num_state, upoly);
      mfem::CentGridFunction x_cent(&dgdSpace);
      x_cent.ProjectCoefficient(u0_fun);
      ofstream x_centprint("x_cent.txt");
      x_cent.Print(x_centprint,4);
      x_centprint.close();


      mfem::GridFunction x_exact(&fes);
      x_exact.ProjectCoefficient(u0_fun);
      //============== Prolong the solution to SBP nodes =================================
      //============== and check the simple l2 norm error ================================
      ofstream sol_ofs("dgd_test.vtk");
      sol_ofs.precision(14);
      smesh->PrintVTK(sol_ofs,0);
      x_exact.SaveVTK(sol_ofs,"exact",0);

      mfem::GridFunction x(&fes);
      x = 0.0;
      dgdSpace.GetProlongationMatrix()->Mult(x_cent,x);
      ofstream x_prolongprint("x_prolong.txt");
      x.Print(x_prolongprint,4);
      x_prolongprint.close();

      //============== Prolong the solution to SBP nodes =================================
      //============== and check the simple l2 norm error ======
      x.SaveVTK(sol_ofs,"prolong",0);
      x -= x_exact;
      x.SaveVTK(sol_ofs,"error",0);
      sol_ofs.close();
      cout << "Check the projection l2 error: " << x.Norml2() << '\n';

      SparseMatrix *prolong = dgdSpace.GetCP();
      DenseMatrix *p = prolong->ToDenseMatrix();
      ofstream p_save("p.txt");
      p_save << std::fixed << setprecision(16);
      for (int i = 0; i < p->Height(); i++)
      {
         for (int j = 0; j < p->Width(); j++)
         {
            p_save << (*p)(i,j)  << ' ';
         }
         p_save << '\n';
      }
      p_save.close();

      //============== check dpdc =================================
      // fd method
      int pert_idx = 1;
      double pert = 1e-7;
      Vector center_p(center);
      Vector center_m(center);
      center_p(pert_idx) = center_p(pert_idx) + pert;
      center_m(pert_idx) = center_p(pert_idx) - pert;

      dgdSpace.buildProlongationMatrix(center_p);
      SparseMatrix *p_plus = dgdSpace.GetCP();
      DenseMatrix *pd_plus = p_plus->ToDenseMatrix();
      ofstream pp_save("p_plus.txt");
      pp_save << std::fixed << setprecision(16);
      for (int i = 0; i < pd_plus->Height(); i++)
      {
         for (int j = 0; j < pd_plus->Width(); j++)
         {
            pp_save << (*pd_plus)(i,j)  << ' ';
         }
         pp_save << '\n';
      }
      pp_save.close();

      dgdSpace.buildProlongationMatrix(center_m);
      SparseMatrix *p_minus = dgdSpace.GetCP();
      DenseMatrix *pd_minus = p_minus->ToDenseMatrix();
      ofstream pm_save("p_minus.txt");
      pm_save << std::fixed << setprecision(16);
      for (int i = 0; i < pd_minus->Height(); i++)
      {
         for (int j = 0; j < pd_minus->Width(); j++)
         {
            pm_save << (*pd_minus)(i,j)  << ' ';
         }
         pm_save << '\n';
      }
      pm_save.close();

      *pd_plus -= *pd_minus;

      *pd_plus *= (1./pert);
      ofstream fd_save("dpdc_fd.txt");
      fd_save << std::fixed << setprecision(16);
      for (int i = 0; i < pd_plus->Height(); i++)
      {
         for (int j = 0; j < pd_plus->Width(); j++)
         {
            fd_save << (*pd_plus)(i,j)  << ' ';
         }
         fd_save << '\n';
      }
      fd_save.close();

      SparseMatrix dpdc(pd_plus->Height(),pd_plus->Width());
      dgdSpace.GetdPdc(pert_idx,dpdc);

      // test
      //DenseMatrix *testmat = Mult(*p_minus,*pd_minus);

      DenseMatrix *dpdc_dense = dpdc.ToDenseMatrix();
      ofstream dpdc_save("dpdc.txt");
      for (int i = 0; i < dpdc_dense->Height(); i++)
      {
         for (int j = 0; j < dpdc_dense->Width(); j++)
         {
            dpdc_save << (*dpdc_dense)(i,j)  << ' ';
         }
         dpdc_save << '\n';
      }
      dpdc_save.close();

      *dpdc_dense -= *pd_plus;

      cout << "Check dpdc error norm: " << dpdc_dense->FNorm2() << '\n';
      delete dpdc_dense;
      delete p;
      delete pd_plus;
      delete pd_minus;

   }   
   catch (MachException &exception)
   {
      exception.print_message();
   }
   catch (std::exception &exception)
   {
      cerr << exception.what() << endl;
   }
}

// Exact solution; note that I reversed the flow direction to be clockwise, so
// the problem and mesh are consistent with the LPS paper (that is, because the
// triangles are subdivided from the quads using the opposite diagonal)
void uexact(const Vector &x, Vector& q)
{
   q.SetSize(4);
   Vector u(4);
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
}

void upoly(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(4);
   u = 0.0;
   for (int p = 0; p < 4; p++)
   {
      u(p) = pow(x(0), p);
   }
}

void utest(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(4);
   u(0) = 1.0;
   u(1) = 2.0;
   u(2) = 3.0;
   u(3) = 4.0;
}

void usingle(const mfem::Vector &x, mfem::Vector &u)
{
   u.SetSize(1);
   u(0) = 2.0;
}

unique_ptr<Mesh> buildQuarterAnnulusMesh(int degree, int num_rad, int num_ang)
{
   auto mesh_ptr = unique_ptr<Mesh>(new Mesh(num_rad, num_ang,
                                             Element::TRIANGLE, true /* gen. edges */,
                                             2.0, M_PI*0.5, true));
   // strategy:
   // 1) generate a fes for Lagrange elements of desired degree
   // 2) create a Grid Function using a VectorFunctionCoefficient
   // 4) use mesh_ptr->NewNodes(nodes, true) to set the mesh nodes
   
   // Problem: fes does not own fec, which is generated in this function's scope
   // Solution: the grid function can own both the fec and fes
   H1_FECollection *fec = new H1_FECollection(degree, 2 /* = dim */);
   FiniteElementSpace *fes = new FiniteElementSpace(mesh_ptr.get(), fec, 2,
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

   mesh_ptr->NewNodes(*xy, true);
   return mesh_ptr;
}

mfem::Vector buildBasisCenters(int numRad, int numTheta)
{
   double r,theta;
   int numBasis = numRad * numTheta;
   Vector basisCenter(2*numBasis);
   for (int i = 0; i < numBasis; i++)
   {
      r = 1.0 + 2.0 * normal_rand(gen);
      theta = M_PI/2.0 * normal_rand(gen);
      basisCenter(i*2) = r *cos(theta);
      basisCenter(i*2+1) = r * sin(theta);
   }
   return basisCenter;
}

 // Output a QuadratureScheme as an XML .vtp file for visualisation in ParaView or anything else that
    // supports XML VTK files
template <typename T>
void writeBasisCentervtp(const mfem::Vector &center, T &stream)
{
   stream << "<?xml version=\"1.0\"?>\n";
   stream << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
   stream << "<PolyData>\n";
   stream << "<Piece NumberOfPoints=\"" << center.Size() << "\" NumberOfVerts=\"" << center.Size() << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
   stream << "<Points>\n";
   stream << "  <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";
   int numBasis = center.Size()/2;
   for (int i = 0; i < numBasis; i++)
   {
      stream << center(2*i) << ' ' << center(2*i+1) << ' ' << 0.0 << ' ';
   }
   stream << "</DataArray>\n";
   stream << "</Points>\n";
   stream << "<Verts>\n";
   stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
   for (size_t i = 0; i < center.Size(); ++i)
      stream << i << ' ';
   stream << "</DataArray>\n";
   stream << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
   for (size_t i = 1; i <= center.Size(); ++i)
      stream << i << ' ';
   stream << "</DataArray>\n";
   stream << "</Verts>\n";
   stream << "<PointData Scalars=\"w\">\n";
   stream << "  <DataArray type=\"Float32\" Name=\"w\" NumberOfComponents=\"1\" format=\"ascii\">";
   for (int i = 0; i < center.Size(); i++)
      stream << 1.0 << ' ';
   stream << "</DataArray>\n";
   stream << "</PointData>\n";
   stream << "</Piece>\n";
   stream << "</PolyData>\n";
   stream << "</VTKFile>\n";
}