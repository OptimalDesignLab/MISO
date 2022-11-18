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
   double a;  // semi-major
   double b;  // semi-minor
   double xc;
   double yc;
   double lsign;
   template <typename T>
   T operator()(const blitz::TinyVector<T, N> &x) const
   {
      // level-set function to work in physical space
      // return -1 * (((x[0] - 5) * (x[0] - 5)) +
      //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
      // level-set function for reference elements
      return lsign * (((((x[0] * xscale) + min_x - xc) *
                        ((x[0] * xscale) + min_x - xc)) /
                       (a * a)) +
                      ((((x[1] * yscale) + min_y - yc) *
                        ((x[1] * yscale) + min_y - yc)) /
                       (b * b)) -
                      (1.0));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 *
      // (x(1) - 5)));
      return blitz::TinyVector<T, N>(
          lsign * (2.0 * xscale * ((x(0) * xscale) + min_x - xc)) / (a * a),
          lsign * (2.0 * yscale * ((x(1) * yscale) + min_y - yc)) / (b * b));
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
         if (i == nbnd - 1)
         {
            nsurf(0) = 1.0;
            nsurf(1) = 0.0;
         }
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
      double a4 = -0.1036;  // -0.1036 for closed te
      double tc = 0.12;
      double theta = 0.0;
      // thickness
      for (int i = 0; i < nbnd; i++)
      {
         double xc = Xc.at(i)(0);
         double yt = tc *
                     ((a0 * sqrt(xc)) + (a1 * xc) + (a2 * (pow(xc, 2))) +
                      (a3 * (pow(xc, 3))) + (a4 * (pow(xc, 4)))) /
                     (0.2);
         double dytdx = tc *
                        ((0.5 * a0 / sqrt(xc)) + a1 + (2.0 * a2 * xc) +
                         (3.0 * a3 * pow(xc, 2)) + (4.0 * a4 * pow(xc, 3))) /
                        (0.2);
         double d2ytdx = tc *
                         ((-0.25 * a0 / pow(xc, 1.5)) + (2.0 * a2) +
                          (6.0 * a3 * xc) + (12.0 * a4 * pow(xc, 2))) /
                         0.2;
         double ysu = yt * cos(theta);
         double dydx = dytdx * cos(theta);
         double d2ydx = d2ytdx * cos(theta);
         double roc = (pow((1 + (dydx * dydx)), 1.5)) / abs(d2ydx);
         if (xc == 0.0)
         {
            double rle = 0.5 * pow(a0 * tc / 0.20, 2);
            kappa.push_back(1.0 / rle);
         }
         else if (i == 0 || i == nbnd - 1 || i == nbnd - 2)
         {
            kappa.push_back(0.0);
         }
         else
         {
            kappa.push_back(1.0 / roc);
         }
         // kappa.push_back(0.0);
      }
      return kappa;
   }
#if 0
   /// construct exact levelset
   circle<2> constructLevelSet() const
   {
      circle<2> phi_ls;
      phi_ls.xscale = 1.0;
      phi_ls.yscale = 1.0;
      phi_ls.min_x = 0.0;
      phi_ls.min_y = 0.0;
      phi_ls.xc = 5.0;
      phi_ls.yc = 5.0;
      if (ls == 1)
      {
         phi_ls.lsign = -1.0;
         phi_ls.a = 0.5;
         phi_ls.b = 0.5;
      }
      else
      {
         phi_ls.lsign = 1.0;
         phi_ls.a = 4.0;
         phi_ls.b = 4.0;
      }
      return phi_ls;
   }
#endif
#if 1
   /// construct levelset using given geometry points
   Algoim::LevelSet<2> constructLevelSet() const
   {
      std::vector<TinyVector<double, N>> Xc;
      std::vector<TinyVector<double, N>> nor;
      std::vector<TinyVector<double, N - 1>> kappa;
      int nel = mesh->GetNE();
      int nbnd;
      // nbnd =  sqrt(nel) ; //128;
      /// parameters
      double delta = 1e-10;
      double xc = 0.0;
      double yc = 0.0;
      /// radius
      double a, b;
      if (ls == 1)
      {
         a = 4.0;
         b = 1.0;
      }
      else
      {
         a = 3.0;
         b = 3.0;
      }
/// use this if reading from file
#if 1
      const char *geometry_file = "NACA_0012_200pts.dat";
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
      /// construct the normal vector for all boundary points
      nor = constructNormal(Xc);
      /// get the curvature vector for all boundary points
      kappa = getCurvature(Xc);
      /// get the number of boundary points
      nbnd = Xc.size();
      cout << "nbnd " << nbnd << endl;
#endif
      double rho = 10 * nbnd;
/// use this if not reading from file
#if 0
      for (int k = 0; k < nbnd; ++k)
      {
         double theta = k * 2.0 * M_PI / nbnd;
         TinyVector<double, N> x, nrm;
         x(0) = a * cos(theta);
         x(1) = b * sin(theta);
         nrm(0) = 2.0 * (x(0)) / (a * a);
         nrm(1) = 2.0 * (x(1)) / (b * b);
         double ds = mag(nrm);
         TinyVector<double, N> ni;
         ni = nrm / ds;
         Xc.push_back(x);
         nor.push_back(ni);
         /// curvature correction
         TinyVector<double, N> dx, d2x;
         dx = {-a * sin(theta), b * cos(theta)};
         d2x = {-a * cos(theta), -b * sin(theta)};
         double num = (dx(0) * d2x(1) - dx(1) * d2x(0));
         double mag_dx = mag(dx);
         double den = mag_dx * mag_dx * mag_dx;
         TinyVector<double, N - 1> curv;
         curv(0) = num / den;
         kappa.push_back(curv);
         // kappa.push_back(0.0);
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
      /// translate ellipse/airfoil
      TinyVector<double, N> xcent;
      xcent(0) = 10.0;
      xcent(1) = 10.0;
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
      Algoim::LevelSet<2> phi_ls;
      phi_ls.initializeLevelSet(Xcoord, nor, kappa, rho, lsign, delta);
      phi_ls.xscale = 1.0;
      phi_ls.yscale = 1.0;
      phi_ls.min_x = 0.0;
      phi_ls.min_y = 0.0;
      TinyVector<double, 2> xle, xte;
      xle(0) = 19.5;
      xle(1) = 20.0;
      xte(0) = 19.997592;
      xte(1) = 20.0;
      // xle(0) = 6.0;
      // xle(1) = 10.0;
      // xte(0) = 14.0;
      // xte(1) = 10.0;
      // xle(0) = 1.5;
      // xle(1) = 2.0;
      // xte(0) = 2.5;
      // xte(1) = 2.0;
      std::cout << std::setprecision(10) << std::endl;
      cout << "phi , gradphi at leading edge: " << endl;
      cout << phi_ls(xle) << " , " << phi_ls.grad(xle) << endl;
      cout << "phi , gradphi at trailing edge: " << endl;
      cout << phi_ls(xte) << " , " << phi_ls.grad(xte) << endl;
      cout << "============================== " << endl;
      return phi_ls;
   }
#endif
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

   /// function that checks if face is immersed or not
   bool findImmersedFace(int &fid) const
   {
      mfem::Array<int> v;
      mesh->GetEdgeVertices(fid, v);
      double *v1coord, *v2coord;
      v1coord = mesh->GetVertex(v[0]);
      v2coord = mesh->GetVertex(v[1]);
      TinyVector<double, N> xv1, xv2;
      xv1(0) = v1coord[0];
      xv1(1) = v1coord[1];
      xv2(0) = v2coord[0];
      xv2(1) = v2coord[1];
      if (phi(xv1) > 0.0 && phi(xv2) > 0.0)
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
      double tol = 1e-14;
      QuadratureRule<N> qp;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir = NULL;
         blitz::TinyVector<double, N> xmin;
         blitz::TinyVector<double, N> xmax;
         blitz::TinyVector<double, N> xupper;
         blitz::TinyVector<double, N> xlower;
         // standard reference element
         // xlower = {0, 0};
         // xupper = {1, 1};
         int dir = -1;
         int side = -1;
         int elemid = cutelems.at(k);
         ElementTransformation *trans = mesh->GetElementTransformation(elemid);
         findBoundingBox(elemid, xmin, xmax);
         // phi.xscale = 1.0;
         // phi.yscale = 1.0;
         // phi.min_x = 0.0;
         // phi.min_y = 0.0;
         double xscale = xmax[0] - xmin[0];
         double yscale = xmax[1] - xmin[1];
         xlower = {xmin[0], xmin[1]};
         xupper = {xmax[0], xmax[1]};
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
            trans->SetIntPoint(&ip);
            ip.weight = pt.w / trans->Weight();
            TinyVector<double, N> xp;
            xp[0] = (pt.x[0]);  //* phi.xscale) + phi.min_x;
            xp[1] = (pt.x[1]);  // * phi.yscale) + phi.min_y;
            qp.evalIntegrand(xp, pt.w);
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
      std::cout << "  scheme.vtp file written for cut cells, containing " << qp.nodes.size()
                << " quadrature points\n";
   }
   /// get integration rule for cut segments
   void GetCutSegmentIntRule(
       vector<int> cutelems,
       vector<int> cutinteriorFaces,
       int order,
       double radius,
       std::map<int, IntegrationRule *> &cutSegmentIntRules,
       std::map<int, IntegrationRule *> &cutInteriorFaceIntRules)
   {
      QuadratureRule<N> qp, qface;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir = NULL;
         blitz::TinyVector<double, N> xmin;
         blitz::TinyVector<double, N> xmax;
         blitz::TinyVector<double, N> xupper;
         blitz::TinyVector<double, N> xlower;
         int side;
         int dir;
         double tol = 1e-14;
         // standard reference element
         // xlower = {0, 0};
         // xupper = {1, 1};
         int elemid = cutelems.at(k);
         ElementTransformation *trans = mesh->GetElementTransformation(elemid);
         findBoundingBox(elemid, xmin, xmax);
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
            trans->SetIntPoint(&ip);
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
                  if (abs(v1coord[0] - v2coord[0]) < 1e-15)
                  {
                     dir = 0;
                     // if (v1coord[0] > v2coord[1])
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
                              ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
                              // cout << "pt.x[0] " << pt.x[0] << endl;
                              // cout << "ip.x " << ip.x << endl;
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
                              ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
                           }
                           else
                           {
                              ip.x = (pt.x[0] - xmin[0]) / xscale;
                           }

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
      std::cout << "  scheme.vtp file written for cut segments, containing " << qp.nodes.size()
                << " quadrature points\n";
      /// quad rule for faces
      std::ofstream face("cut_face_quad_rule_ls_bnds_outer.vtp");
      Algoim::outputQuadratureRuleAsVtpXML(qface, face);
      std::cout << "  scheme.vtp file written for interior faces, containing "
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
                  mfem::Array<int> v;
                  mesh->GetEdgeVertices(fid, v);
                  double *v1coord, *v2coord;
                  v1coord = mesh->GetVertex(v[0]);
                  v2coord = mesh->GetVertex(v[1]);
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
                              ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
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
                              ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
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
                     double xq = (pt.x[0]);
                     double yq = (pt.x[1]);
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
                  cutBdrFaceIntRules[elemid] = ir;
                  // delete ir; /// this is a problem
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
   // mutable circle<N> phi_c;
   Algoim::LevelSet<N> phi;
   //circle<N> phi;
};
}  // namespace mach

#endif