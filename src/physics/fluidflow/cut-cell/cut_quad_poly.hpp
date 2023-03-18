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
       std::vector<uvector<double, N>> Xc)
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
#if 1
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
         phi_ls.a = 3.0;
         phi_ls.b = 3.0;
      }
      return phi_ls;
   }
#endif
#if 0
   /// construct levelset using given geometry points
   Algoim::LevelSet<2> constructLevelSet() const
   {
      std::vector<TinyVector<double, N>> Xc;
      std::vector<TinyVector<double, N>> nor;
      std::vector<TinyVector<double, N - 1>> kappa;
      int nel = mesh->GetNE();
      int nbnd;
      nbnd =  sqrt(nel) ; //128;
      /// parameters
      double delta = 1e-10;
      double xc = 5.0;
      double yc = 5.0;
      /// radius
      double a, b;
      if (ls == 1)
      {
         a = 0.5;
         b = 0.5;
      }
      else
      {
         a = 3.0;
         b = 3.0;
      }
/// use this if reading from file
#if 0
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
#if 1
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
      xcent(0) = 5.0;
      xcent(1) = 5.0;
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
      // xle(0) = 19.5;
      // xle(1) = 20.0;
      // xte(0) = 19.997592;
      // xte(1) = 20.0;
      xle(0) = 4.5;
      xle(1) = 5.0;
      xte(0) = 5.5;
      xte(1) = 5.0;
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
      auto LSF = [&](const uvector<double, 2> &x)
      {
         return phi.lsign * (((((x(0) * phi.xscale) + phi.min_x - phi.xc) *
                               ((x(0) * phi.xscale) + phi.min_x - phi.xc)) /
                              (phi.a * phi.a)) +
                             ((((x(1) * phi.yscale) + phi.min_y - phi.yc) *
                               ((x(1) * phi.yscale) + phi.min_y - phi.yc)) /
                              (phi.b * phi.b)) -
                             (1.0));
      };
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
         GetCutElementQuadScheme(LSF, xmin, xmax, 15, order, surf, vol);
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
      std::string fvol = "element_quad_rule_ls_bnds_outer.vtp";
      outputQuadratureRuleAsVtp(qVol, fvol);
      std::cout << "  scheme.vtp file written for cut cells, containing "
                << qVol.size() << " quadrature points\n";
      std::string fsurf = "cut_segment_quad_rule_ls_bnds_outer.vtp";
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
                        return phi.lsign *
                               (((((xlim * phi.xscale) + phi.min_x - phi.xc) *
                                  ((xlim * phi.xscale) + phi.min_x - phi.xc)) /
                                 (phi.a * phi.a)) +
                                ((((x(0) * phi.yscale) + phi.min_y - phi.yc) *
                                  ((x(0) * phi.yscale) + phi.min_y - phi.yc)) /
                                 (phi.b * phi.b)) -
                                (1.0));
                     };
                     GetCutInterfaceQuadScheme(
                         LSF, xmin(1), xmax(1), 15, order, qface);
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
                        return phi.lsign *
                               (((((x(0) * phi.xscale) + phi.min_x - phi.xc) *
                                  ((x(0) * phi.xscale) + phi.min_x - phi.xc)) /
                                 (phi.a * phi.a)) +
                                ((((ylim * phi.yscale) + phi.min_y - phi.yc) *
                                  ((ylim * phi.yscale) + phi.min_y - phi.yc)) /
                                 (phi.b * phi.b)) -
                                (1.0));
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
      std::string face = "cut_face_quad_rule.vtp";
      outputQuadratureRuleAsVtp(qface_all, face);
      std::cout << "  scheme.vtp file written for interior faces, containing "
                << qface_all.size() << " quadrature points\n";
   }
#if 0
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
      std::cout << "  scheme.vtp file written for cut segments, containing "
                << qp.nodes.size() << " quadrature points\n";
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
      std::ofstream face("bdr_cut_face_quad_rule_outer.vtp");
      Algoim::outputQuadratureRuleAsVtpXML(qbdrface, face);
      std::cout << "  scheme.vtp file written for boundary faces, containing "
                << qbdrface.nodes.size() << " quadrature points\n";
   }
#endif
protected:
   mfem::Mesh *mesh;
   // mutable circle<N> phi_c;
   // Algoim::LevelSet<N> phi;
   circle<N> phi;
};
}  // namespace mach

#endif