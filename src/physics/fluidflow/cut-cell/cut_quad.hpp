#ifndef MACH_CUT_QUAD
#define MACH_CUT_QUAD
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "algoim_quad.hpp"
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
   double xmin;
   double ymin;
   double radius;
   double xc = 0.5;
   double yc = 0.5;
   template <typename T>
   T operator()(const blitz::TinyVector<T, N> &x) const
   {
      // level-set function to work in physical space
      // return -1 * (((x[0] - 5) * (x[0] - 5)) +
      //               ((x[1]- 5) * (x[1] - 5)) - (0.5 * 0.5));
      // level-set function for reference elements
      return -1 *
             ((((x[0] * xscale) + xmin - xc) * ((x[0] * xscale) + xmin - xc)) +
              (((x[1] * yscale) + ymin - yc) * ((x[1] * yscale) + ymin - yc)) -
              (radius * radius));
   }
   template <typename T>
   blitz::TinyVector<T, N> grad(const blitz::TinyVector<T, N> &x) const
   {
      // return blitz::TinyVector<T, N>(-1 * (2.0 * (x(0) - 5)), -1 * (2.0 *
      // (x(1) - 5)));
      return blitz::TinyVector<T, N>(
          -1 * (2.0 * xscale * ((x(0) * xscale) + xmin - xc)),
          -1 * (2.0 * yscale * ((x(1) * yscale) + ymin - yc)));
   }
};
template <int N>
class CutCell
{
public:
   CutCell(mfem::Mesh *_mesh) : mesh(_mesh)
   {
      std::vector<TinyVector<double, N>> Xc;
      std::vector<TinyVector<double, N>> nor;
      int nbnd = 64;
      cout << "nbnd " << nbnd << endl;
      /// parameters
      double rho = 10.0 * nbnd;
      /// radius
      double a = 0.5;
      for (int k = 0; k < nbnd; ++k)
      {
         double theta = k * 2.0 * M_PI / nbnd;
         TinyVector<double, N> x, nrm;
         x(0) = a * cos(theta);
         x(1) = a * sin(theta);
         nrm(0) = 2.0 * x(0);
         nrm(1) = 2.0 * x(1);
         double ds = mag(nrm);
         TinyVector<double, N> ni;
         ni = nrm / ds;
         Xc.push_back(x);
         nor.push_back(ni);
      }
      /// evaluate levelset and it's gradient
      // Algoim::LevelSet<N> phi(Xc, nor, rho, delta);
      phi.xscale = 1.0;
      phi.yscale = 1.0;
      phi.xmin = 0.0;
      phi.ymin = 0.0;
      phi.radius = 0.5;
   }
   /// function that checks if an element is `cut` by `embedded geometry` or not
   bool cutByGeom(int &elemid) const
   {
      Element *el = mesh->GetElement(elemid);
      Array<int> v;
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
         // cout << "lsv is " << lsv << endl;
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
      Array<int> v;
      el->GetVertices(v);
      int k;
      k = 0;
      for (int i = 0; i < v.Size(); ++i)
      {
         double *coord = mesh->GetVertex(v[i]);
         Vector lvsval(v.Size());
         TinyVector<double, N> x;
         x(0) = coord[0];
         x(1) = coord[1];
         lvsval(i) = -phi(x);
         if ((lvsval(i) < 0) || (lvsval(i) == 0))
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

   /// find bounding box for a given cut element
   void findBoundingBox(int id,
                        blitz::TinyVector<double, N> &xmin,
                        blitz::TinyVector<double, N> &xmax) const
   {
      Element *el = mesh->GetElement(id);
      Array<int> v;
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
      double tol = 1e-16;
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
         findBoundingBox(elemid, xmin, xmax);
         phi.xscale = xmax[0] - xmin[0];
         phi.yscale = xmax[1] - xmin[1];
         phi.xmin = xmin[0];
         phi.ymin = xmin[1];
         // phi.radius = radius;
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
            ip.x = pt.x[0];
            ip.y = pt.x[1];
            ip.weight = pt.w;
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
         xlower = {0, 0};
         xupper = {1, 1};
         int elemid = cutelems.at(k);
         findBoundingBox(elemid, xmin, xmax);
         phi.xscale = xmax[0] - xmin[0];
         phi.yscale = xmax[1] - xmin[1];
         phi.xmin = xmin[0];
         phi.ymin = xmin[1];
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
            ip.x = pt.x[0];
            ip.y = pt.x[1];
            ip.weight = pt.w;
            i = i + 1;
            // cout << "elem " << elemid << " , " << ip.weight << endl;
            double xqp = (pt.x[0] * phi.xscale) + phi.xmin;
            double yqp = (pt.x[1] * phi.yscale) + phi.ymin;
            MFEM_ASSERT(
                ip.weight > 0,
                "integration point weight is negative in curved surface "
                "int rule from Saye's method");
         }
         cutSegmentIntRules[elemid] = ir;
         Array<int> orient;
         Array<int> fids;
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
                  Array<int> v;
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
                           ip.x = 1 - pt.x[1];
                        }
                        else
                        {
                           ip.x = pt.x[1];
                        }
                     }
                     else if (dir == 1)
                     {
                        if (-1 == orient[c])
                        {
                           ip.x = 1 - pt.x[0];
                        }
                        else
                        {
                           ip.x = pt.x[0];
                        }
                     }
                     ip.weight = pt.w;
                     i = i + 1;
                     // scaled to original element space
                     double xq = (pt.x[0] * phi.xscale) + phi.xmin;
                     double yq = (pt.x[1] * phi.yscale) + phi.ymin;
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
      }
   }

protected:
   mfem::Mesh *mesh;
   mutable circle<N> phi;
};
}  // namespace mach

#endif