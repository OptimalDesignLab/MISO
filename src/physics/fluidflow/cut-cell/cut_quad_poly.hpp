#ifndef MACH_CUT_QUAD_POLY
#define MACH_CUT_QUAD_POLY
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <list>
#include "quadrature_multipoly.hpp"
#include "quadrature_general.hpp"
using namespace mfem;
using namespace std;
using namespace algoim;
namespace mach
{
#if 0
template <int N>
struct LevelSetF
{
   double xscale;
   double yscale;
   double min_x;
   double min_y;
   double a;  // semi-major
   double b;  // semi-minor
   double xc;
   double yc;
   int lsign;

   double operator()(const uvector<double, 2> &x) const
   {
      // level-set function for reference elements
      return lsign * (((((x(0) * xscale) + min_x - xc) *
                        ((x(0) * xscale) + min_x - xc)) /
                       (a * a)) +
                      ((((x(1) * yscale) + min_y - yc) *
                        ((x(1) * yscale) + min_y - yc)) /
                       (b * b)) -
                      (1.0));
   }
   uvector<double, N> grad(const uvector<double, 2> &x) const
   {
      uvector<double, N> phi_x;
      phi_x(0) =
          lsign * (2.0 * xscale * ((x(0) * xscale) + min_x - xc)) / (a * a);
      phi_x(1) =
          lsign * (2.0 * yscale * ((x(1) * yscale) + min_y - yc)) / (b * b);
      return phi_x;
   }
};
#endif
template <int N>
struct LevelSetF
{
   double xscale;
   double yscale;
   double min_x;
   double min_y;
   /// Vector of boundary coordinates
   vector<uvector<double, N>> xbnd;
   /// Vector of boundary normal vectors
   vector<uvector<double, N>> norm;
   /// curvature at every point
   vector<uvector<double, N - 1>> kappa;
   /// penalty parameter
   double rho;
   /// sign of lsf
   double lsign;
   /// parameter that smooths distance near zero
   double delta;
   double operator()(const uvector<double, 2> &xs) const
   {
      using std::exp;
      using std::sqrt;
      uvector<double, N> x;
      x(0) = xs(0) * xscale + min_x;
      x(1) = xs(1) * yscale + min_y;
      int nbnd = xbnd.size();
      /// find minimum distance
      double min_dist = 1e+100;
      for (int i = 0; i < nbnd; ++i)
      {
         uvector<double, N> x_diff;
         x_diff = x - xbnd.at(i);
         double deli_x = sqrt(sqrnorm(x_diff) + delta);
         min_dist = min(min_dist, deli_x);
      }
      double denom = 0.0;
      double phi = 0.0;
      /// evaluate the level-set
      for (int i = 0; i < nbnd; ++i)
      {
         uvector<double, N> x_diff;
         uvector<double, N> norvx, norvy;
         uvector<double, N> proj;
         x_diff = x - xbnd.at(i);
         double dist1 = dot(x_diff, norm.at(i));
         double nx = norm.at(i)(0);
         double ny = norm.at(i)(1);
         norvx(0) = 1.0 - (nx * nx);
         norvx(1) = -nx * ny;
         norvy(0) = -nx * ny;
         norvy(1) = 1.0 - (ny * ny);
         proj(0) = dot(norvx, x_diff);
         proj(1) = dot(norvy, x_diff);
         double dist2 = 0.5 * kappa.at(i)(0) * dot(x_diff, proj);
         double dist = dist1 + dist2;
         double delx = sqrt(sqrnorm(x_diff) + delta);
         double expc = exp(-rho * (delx - min_dist));
         denom += expc;
         phi += dist * expc;
      }
      phi = phi / denom;
      return lsign * phi;
   }
   /// calculate the gradient of level-set function
   /// \param[in] x - uvector of point where gradphi(x) needs to be
   /// calculated \param[out] phi_bar - level-set function gradient value
   uvector<double, N> grad(const uvector<double, N> &xs) const
   {
      using std::exp;
      using std::sqrt;
      uvector<double, N> x;
      x(0) = xs(0) * xscale + min_x;
      x(1) = xs(1) * yscale + min_y;
      int nbnd = xbnd.size();
      /// find minimum distance
      double min_dist = 1e+100;
      for (int i = 0; i < nbnd; ++i)
      {
         uvector<double, N> x_diff;
         x_diff = x - xbnd.at(i);
         double deli_x = sqrt(sqrnorm(x_diff) + delta);
         min_dist = min(min_dist, deli_x);
      }
      double numer = 0.0;
      double denom = 0.0;
      for (int i = 0; i < nbnd; ++i)
      {
         uvector<double, N> x_diff;
         x_diff = x - xbnd.at(i);
         uvector<double, N> norvx, norvy;
         uvector<double, N> proj;
         double perp1 = dot(x_diff, norm.at(i));
         double nx = norm.at(i)(0);
         double ny = norm.at(i)(1);
         norvx(0) = 1.0 - (nx * nx);
         norvx(1) = -nx * ny;
         norvy(0) = -nx * ny;
         norvy(1) = 1.0 - (ny * ny);
         proj(0) = dot(norvx, x_diff);
         proj(1) = dot(norvy, x_diff);
         double perp2 = 0.5 * kappa.at(i)(0) * dot(x_diff, proj);
         double perp = perp1 + perp2;
         double delx = sqrt(sqrnorm(x_diff) + delta);
         double expc = exp(-rho * (delx - min_dist));
         denom += expc;
         numer += perp * expc;
      }
      double ls = numer / denom;
      // start reverse sweep
      // return ls
      double ls_bar = 1.0;
      // ls = numer / denom
      double numer_bar = ls_bar / denom;
      double denom_bar = -(ls_bar * ls) / denom;
      uvector<double, N> phi_bar;
      phi_bar = 0.0;
      for (int i = 0; i < nbnd; ++i)
      {
         uvector<double, N> x_diff;
         x_diff = x - xbnd.at(i);
         double dist = sqrt(sqrnorm(x_diff) + delta);
         uvector<double, N> norvx, norvy;
         uvector<double, N> proj;
         double perp1 = dot(x_diff, norm.at(i));
         double nx = norm.at(i)(0);
         double ny = norm.at(i)(1);
         norvx(0) = 1.0 - (nx * nx);
         norvx(1) = -nx * ny;
         norvy(0) = -nx * ny;
         norvy(1) = 1.0 - (ny * ny);
         proj(0) = dot(norvx, x_diff);
         proj(1) = dot(norvy, x_diff);
         double perp2 = 0.5 * kappa.at(i)(0) * dot(x_diff, proj);
         double perp = perp1 + perp2;
         double expfac = exp(-rho * (dist - min_dist));
         // denom += expfac
         double expfac_bar = denom_bar;
         expfac_bar += numer_bar * perp;
         // numer += perp*expfac
         double perp_bar = numer_bar * expfac;
         // expfac = exp(-rho*dist)
         double dist_bar = -expfac_bar * expfac * rho;
         // perp = dot(levset.normal[:,i], x - xc)
         phi_bar += perp_bar * norm.at(i);
         // add curvature correction
         phi_bar += perp_bar * kappa.at(i)(0) * proj;
         // dist = sqrt(dot(x - xc, x - xc) + levset.delta)
         phi_bar += (dist_bar / dist) * x_diff;
      }
      // cout << "phi_bar " << phi_bar << endl;
      return lsign * phi_bar;
   }
};

template <int N, int ls>
class CutCell
{
public:
   CutCell(mfem::Mesh *_mesh) : mesh(_mesh) { phi = constructLevelSet(); }

   std::vector<uvector<double, N>> constructNormal(
       std::vector<uvector<double, N>> Xc) const
   {
      std::vector<uvector<double, N>> nor;
      int nbnd = Xc.size();
      uvector<double, N> nsurf;
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

   /// this function constructs the curvature at given boundary points of
   /// level-set geometry
   template <bool cte = false>
   std::vector<uvector<double, N - 1>> getCurvature(
       std::vector<uvector<double, N>> Xc) const
   {
      std::vector<uvector<double, N - 1>> kappa;
      int nbnd = Xc.size();
      double a0 = 0.2969;
      double a1 = -0.126;
      double a2 = -0.3516;
      double a3 = 0.2843;
      double a4 = -0.1015;
      if (cte)
      {
         a4 = -0.1036;
      }
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
         else if (i == 0 || i == 1 || i == nbnd - 1 || i == nbnd - 2)
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
   LevelSetF<2> constructLevelSet() const
   {
      LevelSetF<2> phi_ls;
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
         phi_ls.a = 3.0;
         phi_ls.b = 3.0;
      }
      return phi_ls;
   }
#endif
#if 1
   /// construct approximate levelset for an airfoil
   LevelSetF<2> constructLevelSet() const
   {
      std::vector<uvector<double, N>> Xc;
      std::vector<uvector<double, N>> xbnd;
      std::vector<uvector<double, N>> nor;
      std::vector<uvector<double, N - 1>> kappa;
      const int npts = 51;
      const int nbnd = 2 * npts - 3;
      double tc = 0.12;
      uvector<double, npts - 1> beta;
      // double beta_max = M_PI / 1.02;
      double beta_max = 0.936*M_PI;
      double dbeta = beta_max / (npts - 2);
      beta(0) = beta_max;
      for (int i = 1; i < npts - 1; ++i)
      {
         beta(i) = beta(i - 1) - dbeta;
      }
      constexpr bool cte = true;
      uvector<double, nbnd> xb;
      uvector<double, nbnd> yb;
      double a0 = 0.2969;
      double a1 = -0.126;
      double a2 = -0.3516;
      double a3 = 0.2843;
      double a4 = -0.1015;
      if (cte)
      {
         a4 = -0.1036;
      }
      /// upper boundary
      /// upper boundary
      for (int i = 0; i < npts; ++i)
      {
         xb(i) = (1.0 - cos(beta(i))) / 2.0;
         double term1 =
             (a0 * pow(xb(i), 0.5)) + (a1 * xb(i)) + (a2 * (pow(xb(i), 2)));
         double term2 = (a3 * (pow(xb(i), 3))) + (a4 * (pow(xb(i), 4)));
         yb(i) = 5.0 * tc * (term1 + term2);
      }
      /// lower boundary
      for (int i = 0; i < npts - 2; ++i)
      {
         xb(i + npts - 1) = xb(npts - 3 - i);
         yb(i + npts - 1) = -yb(npts - 3 - i);
      }
      for (int i = 0; i < nbnd; ++i)
      {
         uvector<double, N> x;
         x(0) = xb(i);
         x(1) = yb(i);
         Xc.push_back(x);
      }
      /// use this if not reading from file
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
      /// construct the normal vector for all boundary points
      nor = constructNormal(Xc);
      kappa = getCurvature<cte>(Xc);
      uvector<double, N> xcent;
      xcent(0) = 19.5;
      xcent(1) = 20.0;
      for (int k = 0; k < nbnd; ++k)
      {
         uvector<double, N> xs;
         for (int d = 0; d < N; ++d)
         {
            xs(d) = Xc.at(k)(d) + xcent(d);
         }
         xbnd.push_back(xs);
      }
      /// parameters
      double delta = 1e-10;
      int ratio = 10;
      double rho = ratio * nbnd;
      /// construct levelset
      LevelSetF<2> phi_ls;
      phi_ls.xbnd = xbnd;
      phi_ls.norm = nor;
      phi_ls.kappa = kappa;
      phi_ls.lsign = lsign;
      phi_ls.rho = rho;
      phi_ls.xscale = 1.0;
      phi_ls.yscale = 1.0;
      phi_ls.min_x = 0.0;
      phi_ls.min_y = 0.0;
      phi_ls.delta = delta;
      uvector<double, 2> xle, xte;
      xle(0) = 19.5;
      xle(1) = 20.0;
      xte(0) = 20.5;
      xte(1) = 20.0;
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
         uvector<double, N> x;
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
         uvector<double, N> x;
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
      uvector<double, N> xv1, xv2;
      xv1(0) = v1coord[0];
      xv1(1) = v1coord[1];
      xv2(0) = v2coord[0];
      xv2(1) = v2coord[1];
      if (phi(xv1) >= 0.0 && phi(xv2) >= 0.0)
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
                        uvector<double, N> &xmin,
                        uvector<double, N> &xmax) const
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
      xmin(0) = min(0);
      xmin(1) = min(1);
      xmax(0) = max(0);
      xmax(1) = max(1);
   }

   /// find worst cut element size
   void GetCutsize(vector<int> cutelems,
                   std::map<int, IntegrationRule *> &cutSquareIntRules,
                   double &cutsize) const
   {
      cutsize = 0.0;
   }
   // Given a set of quadrature points and weights, output them to an VTP XML
   // file for visualisation
   // purposes, e.g., using ParaView
   void outputQuadratureRuleAsVtp(const std::vector<uvector<double, N + 1>> &q,
                                  std::string fn) const
   {
      static_assert(N == 2 || N == 3,
                    "outputQuadratureRuleAsVtpXML only supports 2D and 3D "
                    "quadrature schemes");
      std::ofstream stream(fn);
      stream << "<?xml version=\"1.0\"?>\n";
      stream << "<VTKFile type=\"PolyData\" version=\"0.1\" "
                "byte_order=\"LittleEndian\">\n";
      stream << "<PolyData>\n";
      stream << "<Piece NumberOfPoints=\"" << q.size() << "\" NumberOfVerts=\""
             << q.size()
             << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" "
                "NumberOfPolys=\"0\">\n";
      stream << "<Points>\n";
      stream << "  <DataArray type=\"Float32\" Name=\"Points\" "
                "NumberOfComponents=\"3\" format=\"ascii\">";
      for (const auto &pt : q)
         stream << pt(0) << ' ' << pt(1) << ' ' << (N == 3 ? pt(2) : 0.0)
                << ' ';
      stream << "</DataArray>\n";
      stream << "</Points>\n";
      stream << "<Verts>\n";
      stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" "
                "format=\"ascii\">";
      for (size_t i = 0; i < q.size(); ++i)
         stream << i << ' ';
      stream << "</DataArray>\n";
      stream
          << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
      for (size_t i = 1; i <= q.size(); ++i)
         stream << i << ' ';
      stream << "</DataArray>\n";
      stream << "</Verts>\n";
      stream << "<PointData Scalars=\"w\">\n";
      stream << "  <DataArray type=\"Float32\" Name=\"w\" "
                "NumberOfComponents=\"1\" format=\"ascii\">";
      for (const auto &pt : q)
         stream << pt(N) << ' ';
      stream << "</DataArray>\n";
      stream << "</PointData>\n";
      stream << "</Piece>\n";
      stream << "</PolyData>\n";
      stream << "</VTKFile>\n";
   };
   /// get volume and surface quadrature rule on cut elements
   template <typename F>
   void GetCutElementQuadScheme(
       const F &fphi,
       uvector<double, N> xmin,
       uvector<double, N> xmax,
       const uvector<int, N> &P,
       int q,
       std::vector<uvector<double, N + 1>> &surf,
       std::vector<uvector<double, N + 1>> &phase0) const
   {
      // Construct phi by mapping [0,1] onto bounding box [xmin,xmax]
      xarray<double, N> phi(nullptr, P);
      algoim_spark_alloc(double, phi);
      bernstein::bernsteinInterpolate<N>(
          [&](const uvector<double, N> &x)
          { return fphi(xmin + x * (xmax - xmin)); },
          phi);
      // Build quadrature hierarchy
      ImplicitPolyQuadrature<2> ipquad(phi);
      // Compute quadrature scheme and record the nodes & weights; phase0
      // corresponds to {phi < 0}, and surf corresponds to {phi == 0}.
      ipquad.integrate(AutoMixed,
                       q,
                       [&](const uvector<double, N> &x, double w)
                       {
                          if (bernstein::evalBernsteinPoly(phi, x) < 0)
                          {
                             phase0.push_back(add_component(x, N, w));
                          }
                       });
      ipquad.integrate_surf(AutoMixed,
                            q,
                            [&](const uvector<double, N> &x,
                                double w,
                                const uvector<double, N> &wn)
                            { surf.push_back(add_component(x, N, w)); });
   }

   /// get volume and surface quadrature rule on cut elements
   template <typename F>
   void GetCutInterfaceQuadScheme(const F &fphi,
                                  uvector<double, N - 1> xmin,
                                  uvector<double, N - 1> xmax,
                                  const uvector<int, N - 1> &P,
                                  int q,
                                  std::vector<uvector<double, N>> &phase0) const
   {
      // Construct phi_local by mapping [0,1] onto bounding box [xmin,xmax]
      xarray<double, N - 1> phi_local(nullptr, P);
      algoim_spark_alloc(double, phi_local);
      bernstein::bernsteinInterpolate<N - 1>(
          [&](const uvector<double, N - 1> &x)
          { return fphi(xmin + x * (xmax - xmin)); },
          phi_local);
      // Build quadrature hierarchy
      ImplicitPolyQuadrature<N - 1> ipquad(phi_local);
      // Compute quadrature scheme and record the nodes & weights; phase0
      // corresponds to {phi_local < 0}
      ipquad.integrate(AlwaysGL,
                       q,
                       [&](const uvector<double, N - 1> &x, double w)
                       {
                          if (bernstein::evalBernsteinPoly(phi_local, x) < 0)
                          {
                             phase0.push_back(add_component(x, N - 1, w));
                          }
                       });
   }

   /// get integration rule for cut elements
   void GetCutElementIntRule(
       vector<int> cutelems,
       int order,
       std::map<int, IntegrationRule *> &cutSquareIntRules,
       std::map<int, IntegrationRule *> &cutSegmentIntRules) const
   {
      auto LSF = [&](const uvector<double, 2> &x) { return phi(x); };
      cout << "#cut elements " << cutelems.size() << endl;
      double tol = 1e-14;
      std::vector<uvector<double, N + 1>> qVol, qSurf;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir = NULL;
         IntegrationRule *irSurf = NULL;
         uvector<double, N> xmin;
         uvector<double, N> xmax;
         uvector<double, N> xupper;
         uvector<double, N> xlower;
         // standard reference element
         // xlower = {0, 0};
         // xupper = {1, 1};
         int elemid = cutelems.at(k);
         ElementTransformation *trans = mesh->GetElementTransformation(elemid);
         findBoundingBox(elemid, xmin, xmax);
         double xscale = xmax(0) - xmin(0);
         double yscale = xmax(1) - xmin(1);
         xlower(0) = xmin(0);
         xlower(1) = xmin(1);
         xupper(0) = xmax(0);
         xupper(1) = xmax(1);
         std::vector<uvector<double, N + 1>> surf;
         std::vector<uvector<double, N + 1>> vol;
         GetCutElementQuadScheme(LSF, xmin, xmax, 13, order, surf, vol);
         int i = 0;
         if (vol.size() > 0)
         {
            ir = new IntegrationRule(vol.size());
            for (const auto &pt : vol)
            {
               IntegrationPoint &ip = ir->IntPoint(i);
               // ip.x = (pt.x[0] - xmin[0]) / xscale;
               // ip.y = (pt.x[1] - xmin[1]) / yscale;
               //   ip.weight = pt.w / trans->Weight();
               trans->SetIntPoint(&ip);
               ip.x = pt(0);
               ip.y = pt(1);
               ip.weight = pt(2);
               uvector<double, N + 1> xp;
               xp(0) = (pt(0) * xscale) + xmin(0);
               xp(1) = (pt(1) * yscale) + xmin(1);
               xp(2) = pt(2) * trans->Weight();
               qVol.push_back(xp);
               i = i + 1;
               MFEM_ASSERT(ip.weight > 0,
                           "integration point weight is negative in domain "
                           "integration from Saye's method");
               // MFEM_ASSERT(
               //     (phi(xp) < tol),
               //     " phi = "
               //         << phi(xp) << " : "
               //         << " levelset function positive at the quadrature
               //         point "
               //            "domain integration (Saye's method)");
            }
            cutSquareIntRules[elemid] = ir;
         }
         i = 0;
         if (surf.size() > 0)
         {
            irSurf = new IntegrationRule(surf.size());
            for (const auto &pt : surf)
            {
               IntegrationPoint &ip = irSurf->IntPoint(i);
               // ip.x = (pt.x[0] - xmin[0]) / xscale;
               // ip.y = (pt.x[1] - xmin[1]) / yscale;
               //   ip.weight = pt.w / trans->Weight();
               trans->SetIntPoint(&ip);
               ip.x = pt(0);
               ip.y = pt(1);
               ip.weight = pt(2);
               uvector<double, N + 1> xp;
               xp(0) = (pt(0) * xscale) + xmin(0);
               xp(1) = (pt(1) * yscale) + xmin(1);
               xp(2) = pt(2) * trans->Weight();
               qSurf.push_back(xp);
               i = i + 1;
               MFEM_ASSERT(ip.weight > 0,
                           "integration point weight is negative in domain "
                           "integration from Saye's method");
               // MFEM_ASSERT(
               //     (phi(xp) < tol),
               //     " phi = "
               //         << phi(xp) << " : "
               //         << " levelset function positive at the quadrature
               //         point "
               //            "domain integration (Saye's method)");
            }
            cutSegmentIntRules[elemid] = irSurf;
         }
      }
      std::string fvol = "cut-element-quadrature.vtp";
      outputQuadratureRuleAsVtp(qVol, fvol);
      std::cout << "  scheme.vtp file written for cut cells, containing "
                << qVol.size() << " quadrature points\n";
      std::string fsurf = "cut-segment-quadrature.vtp";
      outputQuadratureRuleAsVtp(qSurf, fsurf);
      std::cout << "  scheme.vtp file written for cut segments, containing "
                << qSurf.size() << " quadrature points\n";
   }
   void GetCutInterfaceIntRule(
       vector<int> cutelems,
       vector<int> cutinteriorFaces,
       int order,
       std::map<int, IntegrationRule *> &cutInteriorFaceIntRules)
   {
      std::vector<uvector<double, N + 1>> qface_all;
      for (int k = 0; k < cutelems.size(); ++k)
      {
         IntegrationRule *ir = NULL;
         uvector<double, N> xmin;
         uvector<double, N> xmax;
         uvector<double, N> xupper;
         uvector<double, N> xlower;
         int side;
         int dir;
         double tol = 1e-14;
         // standard reference element
         // xlower = {0, 0};
         // xupper = {1, 1};
         int elemid = cutelems.at(k);
         findBoundingBox(elemid, xmin, xmax);
         double xscale = xmax(0) - xmin(0);
         double yscale = xmax(1) - xmin(1);
         xlower(0) = xmin(0);
         xlower(1) = xmin(1);
         xupper(0) = xmax(0);
         xupper(1) = xmax(1);
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
                  FaceElementTransformations *trans;
                  trans = mesh->GetInteriorFaceTransformations(fid);
                  mfem::Array<int> v;
                  mesh->GetEdgeVertices(fid, v);
                  double *v1coord, *v2coord;
                  v1coord = mesh->GetVertex(v[0]);
                  v2coord = mesh->GetVertex(v[1]);
                  double xlim, ylim;
                  std::vector<uvector<double, N>> qface;
                  if (abs(v1coord[0] - v2coord[0]) < 1e-15)
                  {
                     dir = 0;
                     // if (v1coord[0] > v2coord[1])
                     if (abs(v1coord[0] - xmax(0)) > 1e-15)
                     {
                        side = 0;
                        xlim = xmin(0);
                     }
                     else
                     {
                        side = 1;
                        xlim = xmax(0);
                     }
                     auto LSF = [&](const uvector<double, 1> &x)
                     {
                        uvector<double, N> xs;
                        xs(0) = xlim;
                        xs(1) = x(0);
                        return phi(xs);
                     };
                     GetCutInterfaceQuadScheme(
                         LSF, xmin(1), xmax(1), 13, order, qface);
                  }
                  else
                  {
                     dir = 1;
                     if (abs(v1coord[1] - xmax(1)) > 1e-15)
                     {
                        side = 0;
                        ylim = xmin(1);
                     }
                     else
                     {
                        side = 1;
                        ylim = xmax(1);
                     }
                     auto LSF = [&](const uvector<double, 1> &x)
                     {
                        uvector<double, N> xs;
                        xs(0) = x(0);
                        xs(1) = ylim;
                        return phi(xs);
                     };
                     GetCutInterfaceQuadScheme(
                         LSF, xmin(0), xmax(0), 15, order, qface);
                  }
                  int i = 0;
                  ir = new IntegrationRule(qface.size());
                  for (const auto &pt : qface)
                  {
                     IntegrationPoint &ip = ir->IntPoint(i);
                     ip.y = 0.0;
                     uvector<double, N + 1> xp;
                     if (dir == 0)
                     {
                        if (v1coord[1] < v2coord[1])
                        {
                           if (-1 == orient[c])
                           {
                              // ip.x = 1 - (pt.x[1] - xmin[1]) / yscale;
                              ip.x = 1 - pt(0);
                           }
                           else
                           {
                              // ip.x = (pt.x[1] - xmin[1]) / yscale;
                              ip.x = pt(0);
                           }
                        }
                        else
                        {
                           if (1 == orient[c])
                           {
                              // ip.x = 1 - (pt.x[1] - xmin[1]) / yscale;
                              ip.x = 1 - pt(0);
                           }
                           else
                           {
                              // ip.x = (pt.x[1] - xmin[1]) / yscale;
                              ip.x = pt(0);
                           }
                        }
                        xp(0) = xlim;
                        xp(1) = (pt(0) * yscale) + xmin(1);
                     }
                     else if (dir == 1)
                     {
                        if (v1coord[0] < v2coord[0])
                        {
                           if (-1 == orient[c])
                           {
                              // ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
                              ip.x = 1 - pt(0);
                              // cout << "pt.x[0] " << pt.x[0] << endl;
                              // cout << "ip.x " << ip.x << endl;
                           }
                           else
                           {
                              ip.x = pt(0);
                              // ip.x = (pt.x[0] - xmin[0]) / xscale;
                           }
                        }
                        else
                        {
                           if (1 == orient[c])
                           {
                              ip.x = 1 - pt(0);
                              // ip.x = 1 - (pt.x[0] - xmin[0]) / xscale;
                           }
                           else
                           {
                              ip.x = pt(0);
                              // ip.x = (pt.x[0] - xmin[0]) / xscale;
                           }
                        }
                        xp(0) = (pt(0) * xscale) + xmin(0);
                        xp(1) = ylim;
                     }
                     trans->SetIntPoint(&ip);
                     ip.weight = pt(1);
                     xp(2) = pt(1) * trans->Weight();
                     i = i + 1;
                     qface_all.push_back(xp);
                     MFEM_ASSERT(
                         ip.weight > 0,
                         " ip.weight = "
                             << ip.weight
                             << "integration point weight is negative from "
                                "Saye's method");
                     // MFEM_ASSERT(
                     //     (phi(pt.x) < tol),
                     //     " phi = " << phi(pt.x) << " : "
                     //               << "levelset function positive at the "
                     //                  "quadrature point (Saye's method)");
                     MFEM_ASSERT(
                         (xp(0) <= (max(v1coord[0], v2coord[0]))) &&
                             (xp(0) >= (min(v1coord[0], v2coord[0]))),
                         "integration point (xcoord) not on element face "
                         "(Saye's rule)");
                     MFEM_ASSERT(
                         (xp(1) <= (max(v1coord[1], v2coord[1]))) &&
                             (xp(1) >= (min(v1coord[1], v2coord[1]))),
                         "integration point (ycoord) not on element face "
                         "(Saye's rule)");
                  }
                  cutInteriorFaceIntRules[fid] = ir;
               }
            }
         }
      }  /// loop over cut elements
      /// quad rule for faces
      std::string face = "cut-interface-quadrature.vtp";
      outputQuadratureRuleAsVtp(qface_all, face);
      std::cout << "  scheme.vtp file written for interior faces, containing "
                << qface_all.size() << " quadrature points\n";
   }

protected:
   mfem::Mesh *mesh;
   // Algoim::LevelSet<N> phi;
   LevelSetF<N> phi;
};
}  // namespace mach

#endif