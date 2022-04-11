#ifndef MACH_CUT_QUAD
#define MACH_CUT_QUAD
#include "mfem.hpp"
#include <fstream>
#include <iostream>
// #include "algoim_quad.hpp"
#include "algoim_levelset.hpp"
#include <list>
using namespace mfem;
using namespace std;
using namespace Algoim;
namespace mach
{
template <int N>
struct circle
{
   double xscale;
   double yscale;
   double min_x;
   double min_y;
   double radius;
   double xc = 0.0;
   double yc = 0.0;
   double lsign;
   template <typename T>
   T operator()(const blitz::TinyVector<T, N> &x) const
   {
      // level-set function to work in physical space
      // return -1 * (((x[0] - 5) * (x[0] - 5)) +
      //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
      // level-set function for reference elements
      return lsign * ((((x[0] * xscale) + min_x - xc) *
                       ((x[0] * xscale) + min_x - xc)) +
                      (((x[1] * yscale) + min_y - yc) *
                       ((x[1] * yscale) + min_y - yc)) -
                      (radius * radius));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 *
      // (x(1) - 5)));
      return blitz::TinyVector<T, N>(
          lsign * (2.0 * xscale * ((x(0) * xscale) + min_x - xc)),
          lsign * (2.0 * yscale * ((x(1) * yscale) + min_y - yc)));
   }
};
template <int N, int ls>
class CutCell
{
public:
   CutCell(mfem::Mesh *_mesh) : mesh(_mesh) { phi = constructLevelSet(); }

   std::vector<TinyVector<double, N>> constructNormal(
       std::vector<TinyVector<double, N>> Xc) const
   {
      std::vector<TinyVector<double, N>> nor;
      int nbnd = Xc.size();
      TinyVector<double, N> nsurf;
      for (int i = 0; i < nbnd; i++)
      {
         std::vector<double> dX;
         for (int j = 0; j < N; ++j)
         {
            if (i == 0)
            {
               dX.push_back(Xc.at(i + 1)(j) - Xc.at(i)(j));
            }
            else if (i == nbnd - 1)
            {
               dX.push_back(Xc.at(i)(j) - Xc.at(i - 1)(j));
            }
            else
            {
               dX.push_back(0.5 * (Xc.at(i + 1)(j) - Xc.at(i - 1)(j)));
            }
         }
         double nx = dX.at(1);
         double ny = -dX.at(0);
         double ds = 1.0 / sqrt((nx * nx) + (ny * ny));
         nsurf(0) = nx * ds;
         nsurf(1) = ny * ds;
         nor.push_back(nsurf);
      }
      return nor;
   }
   std::vector<TinyVector<double, N - 1>> getCurvature(
       std::vector<TinyVector<double, N>> Xc) const
   {
      std::vector<TinyVector<double, N - 1>> kappa;
      int nbnd = Xc.size();
      double a0 = 0.2969;
      double a1 = -0.126;
      double a2 = -0.3516;
      double a3 = 0.2843;
      double a4 = -0.1015;  // -0.1036 for closed te
      double tc = 0.12;
      double theta = 0.0;
      // thickness
      for (int i = 0; i < nbnd; i++)
      {
         double xc = Xc.at(i)(0);
         double yt = tc * ((a0 * sqrt(xc)) + (a1 * xc) + (a2 * (pow(xc, 2))) +
                      (a3 * (pow(xc, 3))) + (a4 * (pow(xc, 4)))) / (0.2);
         double dytdx = tc * ((0.5 * a0 / sqrt(xc)) + a1 + (2.0 * a2 * xc) +
                         (3.0 * a3 * pow(xc, 2)) + (4.0 * a4 * pow(xc, 3))) /(0.2);
         double d2ytdx = tc * ((-0.25 * a0 / pow(xc, 1.5)) + (2.0 * a2) +
                          (6.0 * a3 * xc) + (12.0 * a4 * pow(xc, 2))) /0.2;
         double ysu = yt * cos(theta);
         double dydx = dytdx * cos(theta);
         double d2ydx = d2ytdx * cos(theta);
         double roc = (pow((1 + (dydx * dydx)), 1.5)) / abs(d2ydx);
         if (xc == 0.0)
         {
             double rle = 0.5 * pow(a0 * tc / 0.20, 2);
             kappa.push_back(1.0/rle);
         }
         else if(i ==0 || i==nbnd-1)
         {
            kappa.push_back(0.0);
         }
         else
         {
             kappa.push_back(1.0/roc);
         }
        // kappa.push_back(0.0);
      }
      return kappa;
   }
   /// construct levelset using given geometry points
   Algoim::LevelSet<2> constructLevelSet() const
   {
      std::vector<TinyVector<double, N>> Xc;
      std::vector<TinyVector<double, N>> nor;
      std::vector<TinyVector<double, N - 1>> kappa;
      int nel = mesh->GetNE();
      // int nbnd = 8*sqrt(nel);
      // cout << "nbnd " << nbnd << endl;
      /// parameters
      // double rho = 10 * nbnd;
      double delta = 1e-10;
      double xc = 0.0;
      double yc = 0.0;
      phi_c.xscale = 1.0;
      phi_c.yscale = 1.0;
      phi_c.min_x = 0.0;
      phi_c.min_y = 0.0;
      /// radius
      double a, b;
      if (ls == 1)
      {
         a = 1.0;
         b = 1.0;
         phi_c.lsign = -1.0;
         phi_c.radius = 1.0;
      }
      else
      {
         a = 3.0;
         phi_c.lsign = 1.0;
         phi_c.radius = 3.0;
      }
      const char *geometry_file = "naca_0012_64.dat";
      ifstream file;
      file.open(geometry_file);
      /// read the boundary coordinates from user-provided file
      while (1)
      {
         if (file.eof()) break;
         TinyVector<double, N> x;
         for (int j = 0; j < N; ++j)
         {
            file >> x(j);
         }
         Xc.push_back(x);
      }
      file.close();
      /// get the number of boundary points
      int nbnd = Xc.size();
      cout << "nbnd " << nbnd << endl;
      double rho = 10 * nbnd;
      /// construct the normal vector for all boundary points
      nor = constructNormal(Xc);
      /// get the curvature vector for all boundary points
      kappa = getCurvature(Xc);
/// use this if not reading from file
#if 0
      for (int k = 0; k < nbnd; ++k)
      {
         double theta = k * 2.0 * M_PI / nbnd;
         TinyVector<double, N> x, nrm;
         x(0) = a * cos(theta) + xc;
         x(1) = a * sin(theta) + yc;
         nrm(0) = 2.0 * (x(0) - xc);
         nrm(1) = 2.0 * (x(1) - yc);
         double ds = mag(nrm);
         TinyVector<double, N> ni;
         ni = nrm / ds;
         Xc.push_back(x);
         nor.push_back(ni);
         /// curvature correction
         TinyVector<double, N> dx, d2x;
         dx = {-a * sin(theta), a * cos(theta)};
         d2x = {-a * cos(theta), -a * sin(theta)};
         double num = (dx(0) * d2x(1) - dx(1) * d2x(0));
         double mag_dx = mag(dx);
         double den = mag_dx * mag_dx * mag_dx;
         TinyVector<double, N - 1> curv;
         curv(0) = num / den;
         kappa.push_back(curv);
         //kappa.push_back(0.0);
      }
#endif
      double lsign;
      /// initialize levelset
      if (ls == 1)
      {
         lsign = -1.0;
      }
      else
      {
         lsign = 1.0;
      }
      /// translate airfoil
      TinyVector<double, N> xcent;
      xcent(0) = 19.5;
      xcent(1) = 20.0;
      std::vector<TinyVector<double, N>> Xcoord;
      for (int k = 0; k < nbnd; ++k)
      {
         TinyVector<double, N> xs;
         for (int d = 0; d < N; ++d)
         {
            xs(d) = Xc.at(k)(d) + xcent(d);
         }
         Xcoord.push_back(xs);
      }
      Algoim::LevelSet<2> phi;
      phi.initializeLevelSet(Xcoord, nor, kappa, rho, lsign, delta);
      phi.xscale = 1.0;
      phi.yscale = 1.0;
      phi.min_x = 0.0;
      phi.min_y = 0.0;
      TinyVector<double, 2> xle, xte;
      xle(0) = 19.5;
      xle(1) = 20.0;
      xte(0) = 19.997592;
      xte(1) = 20.0;
      std::cout << std::setprecision(10) << std::endl;
      cout << "phi , gradphi at leading edge: " << endl;
      cout << phi(xle) <<  " , " << phi.grad(xle) << endl;
      cout << "phi , gradphi at trailing edge: " << endl;
      cout << phi(xte) <<  " , " << phi.grad(xte) << endl;
      cout << "============================== " << endl;
/// just checking normal vectors
#if 0 
      cout << "norm vectors " << endl;
      blitz::TinyVector<double, 2> beta, beta_e;
      beta = phi.grad(x);
      beta_e = phi_e.grad(x);
      double xc = 0.5;
      double yc = 0.5;
      double nx = beta(0);
      double ny = beta(1);
      double nx_e = beta_e(0);
      double ny_e = beta_e(1);
      double ds = sqrt((nx * nx) + (ny * ny));
      double ds_e = sqrt((nx_e * nx_e) + (ny_e * ny_e));
      Vector nrm, nrm_e;
      nrm.SetSize(2);
      nrm_e.SetSize(2);
      nrm(0) = nx / ds;
      nrm(1) = ny / ds;
      nrm_e(0) = nx_e / ds_e;
      nrm_e(1) = ny_e / ds_e;;
      cout << "exact norm vector: " << endl;
      nrm_e.Print();
      cout << "norm vector using ls: " << endl;
      nrm.Print();
#endif
      return phi;
   }

   /// function that checks if an element is `cut` by `embedded geometry` or not
   bool cutByGeom(int &elemid) const
   {
      Element *el = mesh->GetElement(elemid);
      mfem::Array<int> v;
      el->GetVertices(v);
      int k, l, n;
      k = 0;
      l = 0;
      n = 0;
      for (int i = 0; i < v.Size(); ++i)
      {
         double *coord = mesh->GetVertex(v[i]);
         TinyVector<double, N> x;
         x(0) = coord[0];
         x(1) = coord[1];
         Vector lvsval(v.Size());
         lvsval(i) = -phi(x);
         if (elemid == 19 || elemid == 20)
         {
            if (i==0)
            {
               cout << "element id: " << elemid << endl;
            }

            cout << "lvsval cut " << lvsval(i) << endl;
         }
         if ((lvsval(i) < 0) && (abs(lvsval(i)) > 1e-16))
         {
            k = k + 1;
         }
         if ((lvsval(i) > 0))
         {
            l = l + 1;
         }
         if ((lvsval(i) == 0) || (abs(lvsval(i)) < 1e-16))
         {
            n = n + 1;
         }
      }
      if (elemid == 19 || elemid == 20)
      {
         cout << "k " << k << " , "
              << "l " << l << endl;
      }
      if ((k == v.Size()) || (l == v.Size()))
      {
         return false;
      }
      
      if (((k == 3) || (l == 3)) && (n == 1))
      {
         return false;
      }
      else
      {
         return true;
      }
   }
   /// function that checks if an element is inside the `embedded geometry` or
   /// not
   bool insideBoundary(int &elemid) const
   {
      Element *el = mesh->GetElement(elemid);
      mfem::Array<int> v;
      el->GetVertices(v);
      int k;
      k = 0;
      // cout << "elemid " << elemid << endl;
      for (int i = 0; i < v.Size(); ++i)
      {
         double *coord = mesh->GetVertex(v[i]);
         Vector lvsval(v.Size());
         TinyVector<double, N> x;
         x(0) = coord[0];
         x(1) = coord[1];
         if (ls == 1)
         {
            lvsval(i) = -phi(x);
         }
         else
         {
            lvsval(i) = -phi(x);
         }
         if ((lvsval(i) < 0) || (lvsval(i) == 0) || (abs(lvsval(i)) <= 1e-16))
         {
            k = k + 1;
         }
      }
      if (k == v.Size())
      {
         return true;
      }
      else
      {
         return false;
      }
   }

   /// function to get element center
   void GetElementCenter(int id, mfem::Vector &cent) const
   {
      cent.SetSize(mesh->Dimension());
      int geom = mesh->GetElement(id)->GetGeometryType();
      ElementTransformation *eltransf = mesh->GetElementTransformation(id);
      eltransf->Transform(Geometries.GetCenter(geom), cent);
   }
   /// function to get element center
   void GetElementCenter(Mesh *mesh, int id, mfem::Vector &cent)
   {
      cent.SetSize(mesh->Dimension());
      int geom = mesh->GetElement(id)->GetGeometryType();
      ElementTransformation *eltransf = mesh->GetElementTransformation(id);
      eltransf->Transform(Geometries.GetCenter(geom), cent);
   }
   /// find bounding box for a given cut element
   void findBoundingBox(int id,
                        blitz::TinyVector<double, N> &xmin,
                        blitz::TinyVector<double, N> &xmax) const
   {
      Element *el = mesh->GetElement(id);
      mfem::Array<int> v;
      Vector min, max;
      min.SetSize(N);
      max.SetSize(N);
      for (int d = 0; d < N; d++)
      {
         min(d) = infinity();
         max(d) = -infinity();
      }
      el->GetVertices(v);
      for (int iv = 0; iv < v.Size(); ++iv)
      {
         double *coord = mesh->GetVertex(v[iv]);
         for (int d = 0; d < N; d++)
         {
            if (coord[d] < min(d))
            {
               min(d) = coord[d];
            }
            if (coord[d] > max(d))
            {
               max(d) = coord[d];
            }
         }
      }
      xmin = {min[0], min[1]};
      xmax = {max[0], max[1]};
   }

   /// find worst cut element size
   void GetCutsize(vector<int> cutelems,
                   std::map<int, IntegrationRule *> &cutSquareIntRules,
                   double &cutsize) const
   {
      cutsize = 0.0;
   }
   /// get integration rule for cut elements
   void GetCutElementIntRule(
       vector<int> cutelems,
       int order,
       double radius,
       std::map<int, IntegrationRule *> &cutSquareIntRules) const
   {
      cout << "#cut elements " << cutelems.size() << endl;
      double tol = 1e-16;
      QuadratureRule<N> qp;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir;
         blitz::TinyVector<double, N> xmin;
         blitz::TinyVector<double, N> xmax;
         blitz::TinyVector<double, N> xupper;
         blitz::TinyVector<double, N> xlower;
         // standard reference element
         xlower = {0, 0};
         xupper = {1, 1};
         int dir = -1;
         int side = -1;
         int elemid = cutelems.at(k);
         ElementTransformation *trans = mesh->GetElementTransformation(elemid);
         findBoundingBox(elemid, xmin, xmax);
         // phi = constructLevelSet();
         // phi.xscale = xmax[0] - xmin[0];
         // phi.yscale = xmax[1] - xmin[1];
         // phi.min_x = xmin[0];
         // phi.min_y = xmin[1];
         double xscale = xmax[0] - xmin[0];
         double yscale = xmax[1] - xmin[1];
         xlower = {xmin[0], xmin[1]};
         xupper = {xmax[0], xmax[1]};
         // cout << "x/ymin inside cut rule() " << phi.min_x << " , " <<
         // phi.min_y << endl;
         auto q =
             Algoim::quadGen<N>(phi,
                                Algoim::BoundingBox<double, N>(xlower, xupper),
                                dir,
                                side,
                                order);
         int i = 0;
         ir = new IntegrationRule(q.nodes.size());
         for (const auto &pt : q.nodes)
         {
            IntegrationPoint &ip = ir->IntPoint(i);
            ip.x = (pt.x[0] - xmin[0]) / xscale;
            ip.y = (pt.x[1] - xmin[1]) / yscale;
            ip.weight = pt.w / trans->Weight();
            // if (elemid == 1)
            // {
            TinyVector<double, N> xp;
            xp[0] = (pt.x[0]);  //* phi.xscale) + phi.min_x;
            xp[1] = (pt.x[1]);  // * phi.yscale) + phi.min_y;
            qp.evalIntegrand(xp, pt.w);
            // }
            i = i + 1;
            MFEM_ASSERT(ip.weight > 0,
                        "integration point weight is negative in domain "
                        "integration from Saye's method");
            MFEM_ASSERT(
                (phi(pt.x) < tol),
                " phi = "
                    << phi(pt.x) << " : "
                    << " levelset function positive at the quadrature point "
                       "domain integration (Saye's method)");
         }
         cutSquareIntRules[elemid] = ir;
      }
      std::ofstream f("element_quad_rule_ls_bnds_outer.vtp");
      Algoim::outputQuadratureRuleAsVtpXML(qp, f);
      std::cout << "  scheme.vtp file written, containing " << qp.nodes.size()
                << " quadrature points\n";
   }

   /// get integration rule for cut segments
   void GetCutSegmentIntRule(
       vector<int> cutelems,
       vector<int> cutinteriorFaces,
       int order,
       double radius,
       std::map<int, IntegrationRule *> &cutSegmentIntRules,
       std::map<int, IntegrationRule *> &cutInteriorFaceIntRules) const
   {
      QuadratureRule<N> qp, qface;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir;
         blitz::TinyVector<double, N> xmin;
         blitz::TinyVector<double, N> xmax;
         blitz::TinyVector<double, N> xupper;
         blitz::TinyVector<double, N> xlower;
         int side;
         int dir;
         double tol = 1e-16;
         // standard reference element
         // xlower = {0, 0};
         // xupper = {1, 1};
         int elemid = cutelems.at(k);
         ElementTransformation *trans = mesh->GetElementTransformation(elemid);
         findBoundingBox(elemid, xmin, xmax);
         // phi.xscale = xmax[0] - xmin[0];
         // phi.yscale = xmax[1] - xmin[1];
         // phi.min_x = xmin[0];
         // phi.min_y = xmin[1];
         // phi.radius = radius;
         double xscale = xmax[0] - xmin[0];
         double yscale = xmax[1] - xmin[1];
         xlower = {xmin[0], xmin[1]};
         xupper = {xmax[0], xmax[1]};
         // phi.radius = radius;
         dir = N;
         side = -1;
         auto q =
             Algoim::quadGen<N>(phi,
                                Algoim::BoundingBox<double, N>(xlower, xupper),
                                dir,
                                side,
                                order);
         int i = 0;
         ir = new IntegrationRule(q.nodes.size());
         for (const auto &pt : q.nodes)
         {
            IntegrationPoint &ip = ir->IntPoint(i);
            ip.x = (pt.x[0] - xmin[0]) / xscale;
            ip.y = (pt.x[1] - xmin[1]) / yscale;
            ip.weight = pt.w / sqrt(trans->Weight());
            TinyVector<double, N> xp;
            xp[0] = (pt.x[0]);  //* phi.xscale) + phi.min_x;
            xp[1] = (pt.x[1]);  //* phi.yscale) + phi.min_y;
            qp.evalIntegrand(xp, pt.w);
            i = i + 1;
            // cout << "elem " << elemid << " , " << ip.weight << endl;
            // double xqp = (pt.x[0]);// * phi.xscale) + phi.min_x;
            // double yqp = (pt.x[1]);// * phi.yscale) + phi.min_y;
            MFEM_ASSERT(
                ip.weight > 0,
                "integration point weight is negative in curved surface "
                "int rule from Saye's method");
         }
         cutSegmentIntRules[elemid] = ir;
         mfem::Array<int> orient;
         mfem::Array<int> fids;
         mesh->GetElementEdges(elemid, fids, orient);
         int fid;
         for (int c = 0; c < fids.Size(); ++c)
         {
            fid = fids[c];
            if (find(cutinteriorFaces.begin(), cutinteriorFaces.end(), fid) !=
                cutinteriorFaces.end())
            {
               if (cutInteriorFaceIntRules[fid] == NULL)
               {
                  // cout << "fid " << fid  << endl;
                  FaceElementTransformations *trans;
                  trans = mesh->GetInteriorFaceTransformations(fid);
                  // cout << "trans  " << trans->Face->ElementNo << endl;
                  // cout << "face elements " << trans->Elem1No <<  "  , " <<
                  // trans->Elem2No << endl;
                  mfem::Array<int> v;
                  mesh->GetEdgeVertices(fid, v);
                  double *v1coord, *v2coord;
                  v1coord = mesh->GetVertex(v[0]);
                  v2coord = mesh->GetVertex(v[1]);
                  if (v1coord[0] == v2coord[0])
                  {
                     dir = 0;
                     if (v1coord[0] < xmax[0])
                     {
                        side = 0;
                     }
                     else
                     {
                        side = 1;
                     }
                  }
                  else
                  {
                     dir = 1;
                     if (v1coord[1] < xmax[1])
                     {
                        side = 0;
                     }
                     else
                     {
                        side = 1;
                     }
                  }
                  auto q = Algoim::quadGen<N>(
                      phi,
                      Algoim::BoundingBox<double, N>(xlower, xupper),
                      dir,
                      side,
                      order);
                  int i = 0;
                  ir = new IntegrationRule(q.nodes.size());
                  for (const auto &pt : q.nodes)
                  {
                     IntegrationPoint &ip = ir->IntPoint(i);
                     ip.y = 0.0;
                     if (dir == 0)
                     {
                        if (-1 == orient[c])
                        {
                           ip.x = 1 - (pt.x[1] - xmin[1]) / yscale;
                           // cout << "pt.x[1] " << pt.x[1] << endl;
                           // cout << "ip.x " << ip.x << endl;
                        }
                        else
                        {
                           ip.x = (pt.x[1] - xmin[1]) / yscale;
                           // cout << "pt.x[1] " << pt.x[1] << endl;
                           // cout << "ip.x " << ip.x << endl;
                        }
                     }
                     else if (dir == 1)
                     {
                        if (-1 == orient[c])
                        {
                           ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
                           // cout << "pt.x[0] " << pt.x[0] << endl;
                           // cout << "ip.x " << ip.x << endl;
                        }
                        else
                        {
                           ip.x = (pt.x[0] - xmin[0]) / xscale;
                           // cout << "pt.x[0] " << pt.x[0] << endl;
                           // cout << "ip.x " << ip.x << endl;
                        }
                     }
                     TinyVector<double, N> xp;
                     xp[0] = pt.x[0];
                     xp[1] = pt.x[1];
                     qface.evalIntegrand(xp, pt.w);
                     trans->SetIntPoint(&ip);
                     ip.weight = pt.w / trans->Weight();
                     i = i + 1;
                     // scaled to original element space
                     double xq = (pt.x[0]);  //* phi.xscale) + phi.min_x;
                     double yq = (pt.x[1]);  //* phi.yscale) + phi.min_y;
                     MFEM_ASSERT(ip.weight > 0,
                                 "integration point weight is negative from "
                                 "Saye's method");
                     MFEM_ASSERT(
                         (phi(pt.x) < tol),
                         " phi = " << phi(pt.x) << " : "
                                   << "levelset function positive at the "
                                      "quadrature point (Saye's method)");
                     MFEM_ASSERT(
                         (xq <= (max(v1coord[0], v2coord[0]))) &&
                             (xq >= (min(v1coord[0], v2coord[0]))),
                         "integration point (xcoord) not on element face "
                         "(Saye's rule)");
                     MFEM_ASSERT(
                         (yq <= (max(v1coord[1], v2coord[1]))) &&
                             (yq >= (min(v1coord[1], v2coord[1]))),
                         "integration point (ycoord) not on element face "
                         "(Saye's rule)");
                  }
                  cutInteriorFaceIntRules[fid] = ir;
               }
            }
         }
      }  /// loop over cut elements
      std::ofstream f("cut_segment_quad_rule_ls_bnds_outer.vtp");
      Algoim::outputQuadratureRuleAsVtpXML(qp, f);
      std::cout << "  scheme.vtp file written, containing " << qp.nodes.size()
                << " quadrature points\n";
      /// quad rule for faces
      std::ofstream face("cut_face_quad_rule_ls_bnds_outer.vtp");
      Algoim::outputQuadratureRuleAsVtpXML(qface, face);
      std::cout << "  scheme.vtp file written, containing "
                << qface.nodes.size() << " quadrature points\n";
   }
   /// get integration rule for cut segments
   void GetCutBdrSegmentIntRule(
       vector<int> cutelems,
       vector<int> cutBdrFaces,
       int order,
       double radius,
       std::map<int, IntegrationRule *> &cutBdrFaceIntRules)
   {
      QuadratureRule<N> qbdrface;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir;
         blitz::TinyVector<double, N> xmin;
         blitz::TinyVector<double, N> xmax;
         blitz::TinyVector<double, N> xupper;
         blitz::TinyVector<double, N> xlower;
         int side;
         int dir;
         double tol = 1e-16;
         // standard reference element
         // xlower = {0, 0};
         // xupper = {1, 1};
         int elemid = cutelems.at(k);
         findBoundingBox(elemid, xmin, xmax);
         // phi.xscale = xmax[0] - xmin[0];
         // phi.yscale = xmax[1] - xmin[1];
         // phi.min_x = xmin[0];
         // phi.min_y = xmin[1];
         double xscale = xmax[0] - xmin[0];
         double yscale = xmax[1] - xmin[1];
         mfem::Array<int> orient;
         mfem::Array<int> fids;
         xlower = {xmin[0], xmin[1]};
         xupper = {xmax[0], xmax[1]};
         mesh->GetElementEdges(elemid, fids, orient);
         int fid;
         for (int c = 0; c < fids.Size(); ++c)
         {
            fid = fids[c];
            if (find(cutBdrFaces.begin(), cutBdrFaces.end(), fid) !=
                cutBdrFaces.end())
            {
               if (cutBdrFaceIntRules[elemid] == NULL)
               {
                  FaceElementTransformations *trans;
                  trans = mesh->GetFaceElementTransformations(fid);
                  // cout << "fid " << fid << endl;
                  // cout << "trans  " << trans->Face->ElementNo << endl;
                  // cout << "bdr face int rule for " << fid << endl;
                  mfem::Array<int> v;
                  mesh->GetEdgeVertices(fid, v);
                  double *v1coord, *v2coord;
                  v1coord = mesh->GetVertex(v[0]);
                  v2coord = mesh->GetVertex(v[1]);
                  // cout << " x vert " << v1coord[0] << " , " << v2coord[0] <<
                  // endl; cout << " y vert " << v1coord[1] << " , " <<
                  // v2coord[1] << endl; cout << abs(v1coord[0] - v2coord[0]) <<
                  // endl;
                  if (abs(v1coord[0] - v2coord[0]) < 1e-15)
                  {
                     dir = 0;

                     if (abs(v1coord[0] - xmax[0]) > 1e-15)
                     {
                        side = 0;
                     }
                     else
                     {
                        side = 1;
                     }
                  }
                  else
                  {
                     dir = 1;
                     if (abs(v1coord[1] - xmax[1]) > 1e-15)
                     {
                        side = 0;
                     }
                     else
                     {
                        side = 1;
                     }
                  }

                  // cout << "dir " << dir << endl;
                  // cout << "side " << side << endl;

                  auto q = Algoim::quadGen<N>(
                      phi,
                      Algoim::BoundingBox<double, N>(xlower, xupper),
                      dir,
                      side,
                      order);
                  int i = 0;
                  ir = new IntegrationRule(q.nodes.size());
                  for (const auto &pt : q.nodes)
                  {
                     IntegrationPoint &ip = ir->IntPoint(i);
                     ip.y = 0.0;
                     if (dir == 0)
                     {
                        if (v1coord[1] < v2coord[1])
                        {
                           if (-1 == orient[c])
                           {
                              ip.x = 1 - (pt.x[1] - xmin[1]) / yscale;
                           }
                           else
                           {
                              ip.x = (pt.x[1] - xmin[1]) / yscale;
                           }
                        }
                        else
                        {
                           if (1 == orient[c])
                           {
                              ip.x = 1 - (pt.x[1] - xmin[1]) / yscale;
                           }
                           else
                           {
                              ip.x = (pt.x[1] - xmin[1]) / yscale;
                           }
                        }
                     }
                     else if (dir == 1)
                     {
                        if (v1coord[0] < v2coord[0])
                        {
                           if (-1 == orient[c])
                           {
                              ip.x = 1.0 - (pt.x[0] - xmin[0]) / xscale;
                           }
                           else
                           {
                              ip.x = (pt.x[0] - xmin[0]) / xscale;
                           }
                        }
                        else
                        {
                           if (1 == orient[c])
                           {
                              ip.x = 1.0 - (pt.x[0] - xmin[0]) / xscale;
                           }
                           else
                           {
                              ip.x = (pt.x[0] - xmin[0]) / xscale;
                           }
                        }
                     }
                     trans->SetIntPoint(&ip);
                     ip.weight = pt.w / trans->Weight();
                     i = i + 1;
                     // scaled to original element space
                     TinyVector<double, N> xp;
                     xp[0] = pt.x[0];
                     xp[1] = pt.x[1];
                     qbdrface.evalIntegrand(xp, pt.w);
                     double xq = (pt.x[0] * phi.xscale) + phi.min_x;
                     double yq = (pt.x[1] * phi.yscale) + phi.min_y;
                     // cout << setprecision(
                     //             11)
                     //      << "int rule " << xq << " , " << yq << " : " <<
                     //      pt.w << endl;
                     // cout << "ymax " << v2coord[1] << " , ymin " <<
                     // v1coord[1] << endl;

                     // cout << "phi(pt.x) " << phi(pt.x) << endl;

                     MFEM_ASSERT(ip.weight > 0,
                                 "integration point weight is negative from "
                                 "Saye's method");
                     MFEM_ASSERT(
                         (phi(pt.x) < tol),
                         " phi = " << phi(pt.x) << " : "
                                   << "levelset function positive at the "
                                      "quadrature point (Saye's method)");
                     MFEM_ASSERT((xq <= (max(v1coord[0], v2coord[0]))) &&
                                     (xq >= (min(v1coord[0], v2coord[0]))),
                                 "integration point (xcoord) not on element "
                                 "face (Saye's rule)");
                     MFEM_ASSERT((yq <= (max(v1coord[1], v2coord[1]))) &&
                                     (yq >= (min(v1coord[1], v2coord[1]))),
                                 "integration point (ycoord) not on element "
                                 "face (Saye's rule)");
                  }
                  if (ir->Size() > 0)
                  {
                     cutBdrFaceIntRules[elemid] = ir;
                  }
               }
            }
         }
      }
      /// quad rule for faces
      std::ofstream face("cut_face_quad_rule_ls_bnds_fbdr_outer.vtp");
      Algoim::outputQuadratureRuleAsVtpXML(qbdrface, face);
      std::cout << "  scheme.vtp file written, containing "
                << qbdrface.nodes.size() << " quadrature points\n";
   }

protected:
   mfem::Mesh *mesh;
   mutable circle<N> phi_c;
   mutable Algoim::LevelSet<2> phi;
};
}  // namespace mach

#endif