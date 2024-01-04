#include "mfem.hpp"

#include "utils.hpp"
#include "orthopoly.hpp"
#include "sbp_operators.hpp"
#include "sbp_fe.hpp"

namespace mfem
{
using namespace std;

void SBPFiniteElement::multNormMatrix(const DenseMatrix &u,
                                      DenseMatrix &Hu) const
{
   int num_nodes = GetDof();
   MFEM_ASSERT(u.Width() == Hu.Width() && u.Width() == num_nodes, "");
   MFEM_ASSERT(u.Height() == Hu.Height(), "");
   int num_states = u.Height();
   for (int i = 0; i < num_nodes; ++i)
   {
      for (int n = 0; n < num_states; ++n)
      {
         Hu(n, i) = H(i) * u(n, i);
      }
   }
}

void SBPFiniteElement::multNormMatrixInv(const DenseMatrix &u,
                                         DenseMatrix &Hinvu) const
{
   int num_nodes = GetDof();
   MFEM_ASSERT(u.Width() == Hinvu.Width() && u.Width() == num_nodes, "");
   MFEM_ASSERT(u.Height() == Hinvu.Height(), "");
   int num_states = u.Height();
   for (int i = 0; i < num_nodes; ++i)
   {
      double fac = 1.0 / H(i);
      for (int n = 0; n < num_states; ++n)
      {
         Hinvu(n, i) = fac * u(n, i);
      }
   }
}

void SBPFiniteElement::getStrongOperator(int di,
                                         DenseMatrix &D,
                                         bool trans) const
{
   MFEM_ASSERT(di >= 0 && di < GetDim(), "");
   if (trans)
   {
      // Q[di] stores the transposed operator already!!!
      D = Q[di];
      D.InvRightScaling(H);
   }
   else
   {
      D.Transpose(Q[di]);  // this copies Qx^T, etc
      D.InvLeftScaling(H);
   }
}

void SBPFiniteElement::getStrongOperator(int di, int i, Vector &D) const
{
   MFEM_ASSERT(di >= 0 && di < GetDim(), "");
   MFEM_ASSERT(i >= 0 && i < GetDof(), "");
   int num_nodes = GetDof();
   D.SetSize(num_nodes);
   double fac = 1.0 / H(i);
   for (int j = 0; j < num_nodes; ++j)
   {
      D[j] = fac * Q[di](j, i);  // Q[di] stores transposed weak operator
   }
}

void SBPFiniteElement::getWeakOperator(int di,
                                       DenseMatrix &Qdi,
                                       bool trans) const
{
   MFEM_ASSERT(di >= 0 && di < GetDim(), "");
   if (trans)
   {
      // Q[di] stores the transposed operator already!!!
      Qdi = Q[di];  // assignment (deep copy)
   }
   else
   {
      Qdi.Transpose(Q[di]);  // this copies Qx^T, etc
   }
}

void SBPFiniteElement::multWeakOperator(int di,
                                        const DenseMatrix &u,
                                        DenseMatrix &Qu,
                                        bool trans) const
{
   MFEM_ASSERT(di >= 0 && di < GetDim(), "");
   int num_nodes = GetDof();
   MFEM_ASSERT(u.Width() == Qu.Width() && u.Width() == num_nodes, "");
   MFEM_ASSERT(u.Height() == Qu.Height(), "");
   int num_states = u.Height();
   if (trans)
   {
      for (int i = 0; i < num_nodes; ++i)
      {
         for (int j = 0; j < num_nodes; ++j)
         {
            for (int n = 0; n < num_states; ++n)
            {
               // recall that Q[di] stores the transposed operator
               Qu(n, i) -= Q[di](i, j) * u(n, j);
            }
         }
      }
   }
   else  // trans == false
   {
      for (int i = 0; i < num_nodes; ++i)
      {
         for (int j = 0; j < num_nodes; ++j)
         {
            for (int n = 0; n < num_states; ++n)
            {
               // recall that Q[di] stores the transposed operator
               Qu(n, i) += Q[di](j, i) * u(n, j);
            }
         }
      }
   }
}

void SBPFiniteElement::multWeakOperator(int di,
                                        int i,
                                        const DenseMatrix &u,
                                        Vector &Qu) const
{
   MFEM_ASSERT(di >= 0 && di < GetDim(), "");
   int num_nodes = GetDof();
   MFEM_ASSERT(u.Width() == num_nodes, "");
   MFEM_ASSERT(u.Height() == Qu.Size(), "");
   int num_states = u.Height();
   Qu = 0.0;
   for (int j = 0; j < num_nodes; ++j)
   {
      for (int n = 0; n < num_states; ++n)
      {
         // recall that Q[di] stores the transposed operator
         Qu(n) += Q[di](j, i) * u(n, j);
      }
   }
}

void SBPFiniteElement::multStrongOperator(int di,
                                          int i,
                                          const DenseMatrix &u,
                                          Vector &Du) const
{
   multWeakOperator(di, i, u, Du);
   double fac = 1.0 / H(i);
   Du *= fac;
}

double SBPFiniteElement::getQ(int di, int i, int j) const
{
   return Q[di](j, i);  // Recall: Q[di] stores the transposed operator
}

double SBPFiniteElement::getSkewEntry(int di,
                                      int i,
                                      int j,
                                      const mfem::DenseMatrix &adjJ_i,
                                      const mfem::DenseMatrix &adjJ_j) const
{
   double Sij = 0.0;
   for (int k = 0; k < GetDim(); ++k)
   {
      Sij += adjJ_i(k, di) * Q[k](j, i) - adjJ_j(k, di) * Q[k](i, j);
   }
   return Sij;
}

void SBPFiniteElement::getSkewEntryRevDiff(int di,
                                           int i,
                                           int j,
                                           double Sij_bar,
                                           mfem::DenseMatrix &adjJ_i_bar,
                                           mfem::DenseMatrix &adjJ_j_bar) const
{
   for (int k = 0; k < GetDim(); ++k)
   {
      // Sij += adjJ_i(k,di)*Q[k](j,i) - adjJ_j(k,di)*Q[k](i,j);
      adjJ_i_bar(k, di) += Sij_bar * Q[k](j, i);
      adjJ_j_bar(k, di) -= Sij_bar * Q[k](i, j);
   }
}

double SBPFiniteElement::getSymEntry(int di,
                                     int i,
                                     const mfem::DenseMatrix &adjJ_i) const
{
   double Eij = 0.0;
   for (int k = 0; k < GetDim(); ++k)
   {
      Eij += adjJ_i(k, di) * Q[k](i, i);
   }
   return Eij;
}

double SBPFiniteElement::getQEntry(int di,
                                   int i,
                                   int j,
                                   const mfem::DenseMatrix &adjJ_i,
                                   const mfem::DenseMatrix &adjJ_j) const
{
   if (i == j)
   {
      return getSymEntry(di, i, adjJ_i);
   }
   else
   {
      return 0.5 * getSkewEntry(di, i, j, adjJ_i, adjJ_j);
   }
}

void SBPFiniteElement::getProjOperator(DenseMatrix &P) const
{
   MFEM_ASSERT(P.Size() == dof, "");
   // Set lps = I - V*V'*H
   MultAAt(V, P);
   P.RightScaling(H);
   P *= -1.0;
   for (int i = 0; i < dof; ++i)
   {
      P(i, i) += 1.0;
   }
}

double SBPFiniteElement::getProjOperatorEntry(int i, int j) const
{
   MFEM_ASSERT(i < dof, "");
   MFEM_ASSERT(j < dof, "");
   double Pij = (i == j) ? 1.0 : 0.0;
   // loop over the polynomial basis functions
   for (int k = 0; k < V.Width(); ++k)
   {
      Pij -= V(i, k) * V(j, k) * H(j);
   }
   return Pij;
}

void SBPFiniteElement::multProjOperator(const DenseMatrix &u,
                                        DenseMatrix &Pu,
                                        bool trans) const
{
   int num_nodes = GetDof();
   MFEM_ASSERT(u.Width() == Pu.Width() && u.Width() == num_nodes, "");
   MFEM_ASSERT(u.Height() == Pu.Height(), "");
   int num_states = u.Height();
   Vector prod(num_states);  // work vector
   Vector uj;
   Vector Puj;  // For references to existing data
   // Note: DenseMatrix::operator= is not in-place
   Pu = u;
   if (trans)
   {
      // loop over the polynomial basis functions
      for (int i = 0; i < V.Width(); ++i)
      {
         // perform the inner product V(:,i)^T * u
         prod = 0.0;
         for (int j = 0; j < num_nodes; ++j)
         {
            for (int n = 0; n < num_states; ++n)
            {
               prod(n) += V(j, i) * u(n, j);
            }
         }
         // Subtract V(:,i) *(V(:,i)^T H u) from Pu
         for (int j = 0; j < num_nodes; ++j)
         {
            double fac = V(j, i) * H(j);
            for (int n = 0; n < num_states; ++n)
            {
               Pu(n, j) -= fac * prod(n);
            }
         }
      }
   }
   else  // trans != true
   {
      // loop over the polynomial basis functions
      for (int i = 0; i < V.Width(); ++i)
      {
         // perform the inner product V(:,i)^T * H * u
         prod = 0.0;
         for (int j = 0; j < num_nodes; ++j)
         {
            double fac = V(j, i) * H(j);
            for (int n = 0; n < num_states; ++n)
            {
               prod(n) += fac * u(n, j);
            }
         }
         // Subtract V(:,i) *(V(:,i)^T H u) from Pu
         for (int j = 0; j < num_nodes; ++j)
         {
            for (int n = 0; n < num_states; ++n)
            {
               Pu(n, j) -= V(j, i) * prod(n);
            }
         }
      }
   }
}

int SBPFiniteElement::getIntegrationPointIndex(const IntegrationPoint &ip) const
{
   const double tol = 1e-12;
   for (int i = 0; i < GetDof(); ++i)
   {
      double delta = pow(ip.x - x(i, 0), 2);
      if (GetDim() > 1)
      {
         delta += pow(ip.y - x(i, 1), 2);
         if (GetDim() > 2)
         {
            delta += pow(ip.z - x(i, 2), 2);
         }
      }
      delta = sqrt(delta);
      if (delta < tol)
      {
         return i;
      }
   }
   throw mach::MachException(
       "SBPFiniteElement::getIntegrationPointIndex(ip)\n"
       "\tprovided ip is not a node of given element!");
}

/// SBPSegmentElement is a segment element with nodes at Gauss Lobatto
/// points with ordering consistent with SBPTriangleElement's edges.

SBPSegmentElement::SBPSegmentElement(const int degree)
 : SBPFiniteElement(1, Geometry::SEGMENT, degree + 2, degree)
{
   const int num_nodes = degree + 2;
   Q[0].SetSize(num_nodes);
   Vector pts(num_nodes);
   Vector wts(num_nodes);
   mach::getLobattoQuadrature(degree + 2, pts, wts);
   // shift nodes to [0,1] and scale quadrature
   for (int i = 0; i < num_nodes; ++i)
   {
      pts(i) = 0.5 * (pts(i) + 1.0);
      wts(i) *= 0.5;
   }

   Nodes.IntPoint(0).x = pts(0);
   Nodes.IntPoint(0).weight = wts(0);
   Nodes.IntPoint(1).x = pts(num_nodes - 1);
   Nodes.IntPoint(1).weight = wts(num_nodes - 1);
   for (int i = 0; i < (num_nodes - 2) / 2; ++i)
   {
      Nodes.IntPoint(2 * (i + 1)).x = pts(i + 1);
      Nodes.IntPoint(2 * (i + 1)).weight = wts(i + 1);
      Nodes.IntPoint(2 * (i + 1) + 1).x = 1.0 - pts(i + 1);
      Nodes.IntPoint(2 * (i + 1) + 1).weight = wts(i + 1);
   }
   if (num_nodes % 2 == 1)
   {
      // Account for mid-point node
      Nodes.IntPoint(num_nodes - 1).x = pts((num_nodes - 1) / 2);
      Nodes.IntPoint(num_nodes - 1).weight = wts((num_nodes - 1) / 2);
   }

   // Populate the Q[0] matrix
   switch (degree)
   {
   case 0:
      Q[0] = sbp_operators::p0Qx_seg;
      break;
   case 1:
      Q[0] = sbp_operators::p1Qx_seg;
      break;
   case 2:
      Q[0] = sbp_operators::p2Qx_seg;
      break;
   case 3:
      Q[0] = sbp_operators::p3Qx_seg;
      break;
   case 4:
      Q[0] = sbp_operators::p4Qx_seg;
      break;
   default:
      mfem_error(
          "SBP elements are currently only supported for 0 <= order <= 4");
      break;
   }

   // populate unordered_map with mapping from IntPoint address to index
   for (int i = 0; i < num_nodes; ++i)
   {
      ipIdxMap[&(Nodes.IntPoint(i))] = i;
   }
   // set the node and diagonal norm arrays
   for (int i = 0; i < num_nodes; ++i)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      H(i) = ip.weight;
      x(i, 0) = ip.x;
   }

   // Construct the Vandermonde matrix in order to perform LPS projections;
   V.SetSize(num_nodes, degree + 1);
   // First, get node coordinates and shift to segment with vertices (-1), (1)
   Vector xi;
   getNodeCoords(0, xi);
   xi *= 2.0;
   xi -= 1.0;
   mach::getVandermondeForSeg(xi, order, V);
   // scale V to account for the different reference elements
   V *= sqrt(2.0);
}

/// CalcShape outputs ndofx1 vector shape based on Kronecker \delta_{i, ip}
/// where ip is the integration point CalcShape is evaluated at.
void SBPSegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   int ipIdx = -1;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range &oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() (specifically
   // GridFunction::GetValues()), which calls CalcShape() with an
   // `IntegrationPoint` defined by a refined geometry type. Since the
   // IntegrationPoint is not in Nodes, its address is not in the ipIdxMap,
   // and an out_of_range error is thrown.
   {
      // This projects the SBP "basis" onto the degree = order orthogonal polys;
      // Such an approach is fine if LPS is used, but it will eliminate high
      // frequencey modes that may be present in the true solution.  It has
      // the advantage of being fast and not requiring a min-norm solution.
      Vector xvec(1);  // Vector with 1 entry (needed by jacobiPoly)
      Vector poly(1);
      xvec(0) = 2 * ip.x - 1;
      int ptr = 0;
      shape = 0.0;
      for (int i = 0; i <= order; ++i)
      {
         mach::jacobiPoly(xvec, 0.0, 0.0, i, poly);
         poly *= 2.0;  // scale to mfem reference element
         for (int k = 0; k < GetDof(); ++k)
         {
            shape(k) += poly(0) * V(k, ptr) * H(k);
         }
         ++ptr;
      }
      return;
   }
   shape = 0.0;
   shape(ipIdx) = 1.0;
}

/// CalcDShape outputs ndof x 1 DenseMatrix dshape, where the first column
/// is the ith row of Dx, where i is the integration point CalcDShape is
/// evaluated at.
void SBPSegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
   int ipIdx = -1;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range &oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() ->
   // GridFunction::GetValues() which calls CalcShape() with an
   // `IntegrationPoint` defined by a refined geometry type. Since the
   // IntegrationPoint is not in Nodes, its address is not in the ipIdxMap, and
   // an out_of_range error is thrown. This code catches the error and uses
   // float comparisons to determine the IntegrationPoint index.
   {
      double tol = 1e-12;
      for (int i = 0; i < dof; i++)
      {
         double delta_x = ip.x - Nodes.IntPoint(i).x;
         if (fabs(delta_x) < tol)
         {
            ipIdx = i;
            break;
         }
      }
   }
   // TODO: I think we can make tempVec an empty Vector, since it is just a
   // reference
   dshape = 0.0;
   Vector tempVec(dof);
   Q[0].GetColumnReference(ipIdx, tempVec);
   dshape.SetCol(0, tempVec);
   dshape.InvLeftScaling(H);
}

// //////////////////////////////////////////////////////////////////////////
// /// Not currently implemented as collocated SBP type element
// //////////////////////////////////////////////////////////////////////////
// SBPSegmentElement::SBPSegmentElement(const int p)
//    : NodalTensorFiniteElement(1, p+1, BasisType::GaussLobatto, H1_DOF_MAP)
//    //SBPFiniteElement(1, GetTensorProductGeometry(1), p+2, p),
// {
//    const double *cp = poly1d.ClosedPoints(p+1, b_type);

// #ifndef MFEM_THREAD_SAFE
//    shape_x.SetSize(p+2);
//    dshape_x.SetSize(p+2);
// #endif

//    Nodes.IntPoint(0).x = cp[0];
//    Nodes.IntPoint(1).x = cp[p+1];

//    switch (p)
//    {
//       case 1:
//          Nodes.IntPoint(2).x = cp[1];
//          break;
//       case 2:
//          Nodes.IntPoint(2).x = cp[1];
//          Nodes.IntPoint(3).x = cp[2];
//          break;
//       case 3:
//          Nodes.IntPoint(2).x = cp[2];
//          Nodes.IntPoint(3).x = cp[1];
//          Nodes.IntPoint(4).x = cp[3];
//          break;
//       case 4:
//          Nodes.IntPoint(2).x = cp[2];
//          Nodes.IntPoint(3).x = cp[3];
//          Nodes.IntPoint(4).x = cp[1];
//          Nodes.IntPoint(5).x = cp[4];
//          break;
//    }
// }

// void SBPSegmentElement::CalcShape(const IntegrationPoint &ip,
//                                   Vector &shape) const
// {
//    const int p = order;

// #ifdef MFEM_THREAD_SAFE
//    Vector shape_x(p+2);
// #endif

//    basis1d.Eval(ip.x, shape_x);

//    shape(0) = shape_x(0);
//    shape(1) = shape_x(p+1);

//    switch (p)
//    {
//       case 1:
//          shape(2) = shape_x(1);
//          break;
//       case 2:
//          shape(2) = shape_x(1);
//          shape(3) = shape_x(2);
//          break;
//       case 3:
//          shape(2) = shape_x(2);
//          shape(3) = shape_x(1);
//          shape(4) = shape_x(3);
//          break;
//       case 4:
//          shape(2) = shape_x(2);
//          shape(3) = shape_x(3);
//          shape(4) = shape_x(1);
//          shape(5) = shape_x(4);
//          break;
//    }
// }

// void SBPSegmentElement::CalcDShape(const IntegrationPoint &ip,
//                                    DenseMatrix &dshape) const
// {
//    const int p = order;

// #ifdef MFEM_THREAD_SAFE
//    Vector shape_x(p+2), dshape_x(p+2);
// #endif

//    basis1d.Eval(ip.x, shape_x, dshape_x);

//    dshape(0,0) = dshape_x(0);
//    dshape(1,0) = dshape_x(p+1);

//    switch (p)
//    {
//       case 1:
//          dshape(2,0) = dshape_x(1);
//          break;
//       case 2:
//          dshape(2,0) = dshape_x(1);
//          dshape(3,0) = dshape_x(2);
//          break;
//       case 3:
//          dshape(2,0) = dshape_x(2);
//          dshape(3,0) = dshape_x(1);
//          dshape(4,0) = dshape_x(3);
//          break;
//       case 4:
//          dshape(2,0) = dshape_x(2);
//          dshape(3,0) = dshape_x(3);
//          dshape(4,0) = dshape_x(1);
//          dshape(5,0) = dshape_x(4);
//          break;
//    }
// }

// Leftover function from H1_Segment element
// void SBPSegmentElement::ProjectDelta(int vertex, Vector &dofs) const
// {
//    const int p = order;
//    const double *cp = poly1d.ClosedPoints(p, b_type);

//    switch (vertex)
//    {
//       case 0:
//          dofs(0) = poly1d.CalcDelta(p, (1.0 - cp[0]));
//          dofs(1) = poly1d.CalcDelta(p, (1.0 - cp[p]));
//          for (int i = 1; i < p; i++)
//          {
//             dofs(i+1) = poly1d.CalcDelta(p, (1.0 - cp[i]));
//          }
//          break;

//       case 1:
//          dofs(0) = poly1d.CalcDelta(p, cp[0]);
//          dofs(1) = poly1d.CalcDelta(p, cp[p]);
//          for (int i = 1; i < p; i++)
//          {
//             dofs(i+1) = poly1d.CalcDelta(p, cp[i]);
//          }
//          break;
//    }
// }

SBPTriangleElement::SBPTriangleElement(const int degree, const int num_nodes)
 : SBPFiniteElement(2, Geometry::TRIANGLE, num_nodes, degree)
{
   /// Header file including SBP Dx and Dy matrix data
   Q[0].SetSize(num_nodes);
   Q[1].SetSize(num_nodes);

   // Populate the Q[i] matrices and create the element's Nodes
   switch (degree)
   {
   case 0:
      Q[0] = sbp_operators::p0Qx_tri;
      Q[1] = sbp_operators::p0Qy_tri;
      // vertices
      Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.16666666666666666);
      Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.16666666666666666);
      Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.16666666666666666);
      break;
   case 1:
      Q[0] = sbp_operators::p1Qx_tri;
      Q[1] = sbp_operators::p1Qy_tri;
      // vertices
      Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.024999999999999998);
      Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.024999999999999998);
      Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.024999999999999998);
      // edges
      Nodes.IntPoint(3).Set2w(0.5, 0.0, 0.06666666666666667);
      Nodes.IntPoint(4).Set2w(0.5, 0.5, 0.06666666666666667);
      Nodes.IntPoint(5).Set2w(0.0, 0.5, 0.06666666666666667);
      // interior
      Nodes.IntPoint(6).Set2w(
          0.3333333333333333, 0.3333333333333333, 0.22500000000000006);
      break;
   case 2:
      Q[0] = sbp_operators::p2Qx_tri;
      Q[1] = sbp_operators::p2Qy_tri;
      // vertices
      Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.006261126504899741);
      Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.006261126504899741);
      Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.006261126504899741);
      // edges
      Nodes.IntPoint(3).Set2w(0.27639320225002106, 0.0, 0.026823800250389242);
      Nodes.IntPoint(4).Set2w(0.7236067977499789, 0.0, 0.026823800250389242);
      Nodes.IntPoint(5).Set2w(
          0.7236067977499789, 0.27639320225002106, 0.026823800250389242);
      Nodes.IntPoint(6).Set2w(
          0.27639320225002106, 0.7236067977499789, 0.026823800250389242);
      Nodes.IntPoint(7).Set2w(0.0, 0.7236067977499789, 0.026823800250389242);
      Nodes.IntPoint(8).Set2w(0.0, 0.27639320225002106, 0.026823800250389242);
      // interior
      Nodes.IntPoint(9).Set2w(
          0.21285435711180825, 0.5742912857763836, 0.10675793966098839);
      Nodes.IntPoint(10).Set2w(
          0.21285435711180825, 0.21285435711180825, 0.10675793966098839);
      Nodes.IntPoint(11).Set2w(
          0.5742912857763836, 0.21285435711180825, 0.10675793966098839);
      break;
   case 3:
      Q[0] = sbp_operators::p3Qx_tri;
      Q[1] = sbp_operators::p3Qy_tri;
      // vertices
      Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.0022825661430496253);
      Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.0022825661430496253);
      Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.0022825661430496253);
      // edges
      Nodes.IntPoint(3).Set2w(0.5, 0.0, 0.015504052643022513);
      Nodes.IntPoint(4).Set2w(0.17267316464601146, 0.0, 0.011342592592592586);
      Nodes.IntPoint(5).Set2w(0.8273268353539885, 0.0, 0.011342592592592586);
      Nodes.IntPoint(6).Set2w(0.5, 0.5, 0.015504052643022513);
      Nodes.IntPoint(7).Set2w(
          0.8273268353539885, 0.17267316464601146, 0.011342592592592586);
      Nodes.IntPoint(8).Set2w(
          0.17267316464601146, 0.8273268353539885, 0.011342592592592586);
      Nodes.IntPoint(9).Set2w(0.0, 0.5, 0.015504052643022513);
      Nodes.IntPoint(10).Set2w(0.0, 0.8273268353539885, 0.011342592592592586);
      Nodes.IntPoint(11).Set2w(0.0, 0.17267316464601146, 0.011342592592592586);
      // interior
      Nodes.IntPoint(12).Set2w(
          0.4243860251718814, 0.1512279496562372, 0.07467669469983994);
      Nodes.IntPoint(13).Set2w(
          0.4243860251718814, 0.4243860251718814, 0.07467669469983994);
      Nodes.IntPoint(14).Set2w(
          0.1512279496562372, 0.4243860251718814, 0.07467669469983994);
      Nodes.IntPoint(15).Set2w(
          0.14200508409677795, 0.7159898318064442, 0.051518167995569394);
      Nodes.IntPoint(16).Set2w(
          0.14200508409677795, 0.14200508409677795, 0.051518167995569394);
      Nodes.IntPoint(17).Set2w(
          0.7159898318064442, 0.14200508409677795, 0.051518167995569394);
      break;
   case 4:
      Q[0] = sbp_operators::p4Qx_tri;
      Q[1] = sbp_operators::p4Qy_tri;

      // vertices
      Nodes.IntPoint(0).Set2w(
          0.000000000000000000, 0.000000000000000000, 0.001090393904993471);
      Nodes.IntPoint(1).Set2w(
          1.000000000000000000, 0.000000000000000000, 0.001090393904993471);
      Nodes.IntPoint(2).Set2w(
          0.000000000000000000, 1.000000000000000000, 0.001090393904993471);
      // edges
      Nodes.IntPoint(3).Set2w(
          0.357384241759677534, 0.000000000000000000, 0.006966942871463700);
      Nodes.IntPoint(4).Set2w(
          0.642615758240322466, 0.000000000000000000, 0.006966942871463700);
      Nodes.IntPoint(5).Set2w(
          0.117472338035267576, 0.000000000000000000, 0.005519747637357106);
      Nodes.IntPoint(6).Set2w(
          0.882527661964732424, 0.000000000000000000, 0.005519747637357106);
      Nodes.IntPoint(7).Set2w(
          0.642615758240322466, 0.357384241759677534, 0.006966942871463700);
      Nodes.IntPoint(8).Set2w(
          0.357384241759677534, 0.642615758240322466, 0.006966942871463700);
      Nodes.IntPoint(9).Set2w(
          0.882527661964732424, 0.117472338035267576, 0.005519747637357106);
      Nodes.IntPoint(10).Set2w(
          0.117472338035267576, 0.882527661964732424, 0.005519747637357106);
      Nodes.IntPoint(11).Set2w(
          0.000000000000000000, 0.642615758240322466, 0.006966942871463700);
      Nodes.IntPoint(12).Set2w(
          0.000000000000000000, 0.357384241759677534, 0.006966942871463700);
      Nodes.IntPoint(13).Set2w(
          0.000000000000000000, 0.882527661964732424, 0.005519747637357106);
      Nodes.IntPoint(14).Set2w(
          0.000000000000000000, 0.117472338035267576, 0.005519747637357106);
      // interior
      Nodes.IntPoint(15).Set2w(
          0.103677508142805172, 0.792644983714389628, 0.028397190663911491);
      Nodes.IntPoint(16).Set2w(
          0.103677508142805172, 0.103677508142805172, 0.028397190663911491);
      Nodes.IntPoint(17).Set2w(
          0.792644983714389628, 0.103677508142805172, 0.028397190663911491);
      Nodes.IntPoint(18).Set2w(
          0.265331380484209678, 0.469337239031580644, 0.039960048027851809);
      Nodes.IntPoint(19).Set2w(
          0.265331380484209678, 0.265331380484209678, 0.039960048027851809);
      Nodes.IntPoint(20).Set2w(
          0.469337239031580644, 0.265331380484209678, 0.039960048027851809);
      Nodes.IntPoint(21).Set2w(
          0.587085567133367348, 0.088273960601581103, 0.036122826526134168);
      Nodes.IntPoint(22).Set2w(
          0.324640472265051494, 0.088273960601581103, 0.036122826526134168);
      Nodes.IntPoint(23).Set2w(
          0.324640472265051494, 0.587085567133367348, 0.036122826526134168);
      Nodes.IntPoint(24).Set2w(
          0.587085567133367348, 0.324640472265051494, 0.036122826526134168);
      Nodes.IntPoint(25).Set2w(
          0.088273960601581103, 0.324640472265051494, 0.036122826526134168);
      Nodes.IntPoint(26).Set2w(
          0.088273960601581103, 0.587085567133367348, 0.036122826526134168);
      break;
   default:
      mfem_error(
          "SBP elements are currently only supported for 0 <= order <= 4");
      break;
   }

   // populate unordered_map with mapping from IntPoint address to index
   for (int i = 0; i < dof; i++)
   {
      ipIdxMap[&(Nodes.IntPoint(i))] = i;
   }

   for (int i = 0; i < dof; i++)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      H(i) = ip.weight;
      x(i, 0) = ip.x;
      x(i, 1) = ip.y;
   }
   // Construct the Vandermonde matrix in order to perform LPS projections;
   V.SetSize(num_nodes, (degree + 1) * (degree + 2) / 2);
   // First, get node coordinates and shift to triangle with vertices
   // (-1,-1), (1,-1), (-1,1)
   Vector xi;
   Vector eta;
   getNodeCoords(0, xi);
   getNodeCoords(1, eta);
   xi *= 2.0;
   xi -= 1.0;
   eta *= 2.0;
   eta -= 1.0;
   mach::getVandermondeForTri(xi, eta, order, V);
   // scale V to account for the different reference elements
   V *= 2.0;
}

/// CalcShape outputs ndofx1 vector shape based on Kronecker \delta_{i, ip}
/// where ip is the integration point CalcShape is evaluated at.
void SBPTriangleElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   int ipIdx = -1;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range &oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() (specifically
   // GridFunction::GetValues()), which calls CalcShape() with an
   // `IntegrationPoint` defined by a refined geometry type. Since the
   // IntegrationPoint is not in Nodes, its address is not in the ipIdxMap,
   // and an out_of_range error is thrown.
   {
      // This projects the SBP "basis" onto the degree = order orthogonal polys;
      // Such an approach is fine if LPS is used, but it will eliminate high
      // frequencey modes that may be present in the true solution.  It has
      // the advantage of being fast and not requiring a min-norm solution.
      Vector xvec(1);  // Vector with 1 entry (needed by prorioPoly)
      Vector yvec(1);
      Vector poly(1);
      xvec(0) = 2 * ip.x - 1;
      yvec(0) = 2 * ip.y - 1;
      int ptr = 0;
      shape = 0.0;
      for (int r = 0; r <= order; ++r)
      {
         for (int j = 0; j <= r; ++j)
         {
            mach::prorioPoly(xvec, yvec, r - j, j, poly);
            poly *= 2.0;  // scale to mfem reference element
            for (int k = 0; k < GetDof(); ++k)
            {
               shape(k) += poly(0) * V(k, ptr) * H(k);
            }
            ++ptr;
         }
      }
      return;
   }
   shape = 0.0;
   shape(ipIdx) = 1.0;
}

/// CalcDShape outputs ndof x ndim DenseMatrix dshape, where the first column
/// is the ith row of Dx, and the second column is the ith row of Dy, where i
/// is the integration point CalcDShape is evaluated at.
void SBPTriangleElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   int ipIdx = -1;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range &oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() ->
   // GridFunction::GetValues() which calls CalcShape() with an
   // `IntegrationPoint` defined by a refined geometry type. Since the
   // IntegrationPoint is not in Nodes, its address is not in the ipIdxMap, and
   // an out_of_range error is thrown. This code catches the error and uses
   // float comparisons to determine the IntegrationPoint index.
   {
      double tol = 1e-12;
      for (int i = 0; i < dof; i++)
      {
         double delta_x = ip.x - Nodes.IntPoint(i).x;
         double delta_y = ip.y - Nodes.IntPoint(i).y;
         if (delta_x * delta_x + delta_y * delta_y < tol)
         {
            ipIdx = i;
            break;
         }
      }
   }
   dshape = 0.0;

   Vector tempVec(dof);
   Q[0].GetColumnReference(ipIdx, tempVec);
   dshape.SetCol(0, tempVec);
   Q[1].GetColumnReference(ipIdx, tempVec);
   dshape.SetCol(1, tempVec);
   dshape.InvLeftScaling(H);
}

SBPTetrahedronElement::SBPTetrahedronElement(const int degree, const int num_nodes)
 : SBPFiniteElement(3,Geometry::TETRAHEDRON,num_nodes,degree)
 {
   /// Header file including SBP Dx and Dy matrix data
   Q[0].SetSize(num_nodes);
   Q[1].SetSize(num_nodes);
   Q[2].SetSize(num_nodes);
   // Populate the Q[i] matrices and create the element's Nodes   
   switch (degree)
   {
   case 0:
      Q[0] = sbp_operators::p0Qx_tet;
      Q[1] = sbp_operators::p0Qy_tet;
      Q[2] = sbp_operators::p0Qz_tet;
      // vertices
      Nodes.IntPoint(0).Set(0.0,0.0,0.0,0.041666666666666664);
      Nodes.IntPoint(1).Set(1.0,0.0,0.0,0.041666666666666664);
      Nodes.IntPoint(2).Set(0.0,1.0,0.0,0.041666666666666664);
      Nodes.IntPoint(3).Set(0.0,0.0,1.0,0.041666666666666664);
      break;
   case 1:
      Q[0] = sbp_operators::p1Qx_tet;
      Q[1] = sbp_operators::p1Qy_tet;
      Q[2] = sbp_operators::p1Qz_tet;
      // // vertices
      // Nodes.IntPoint(0).Set(0.0,0.0,0.0,0.0026679395344347597);
      // Nodes.IntPoint(1).Set(1.0,0.0,0.0,0.0026679395344347597);
      // Nodes.IntPoint(2).Set(0.0,1.0,0.0,0.0026679395344347597);
      // Nodes.IntPoint(3).Set(0.0,0.0,1.0,0.0026679395344347597);
      // // edges 
      // Nodes.IntPoint(4).Set(0.5,0.0,0.0,0.003996605685951749);
      // Nodes.IntPoint(5).Set(0.5,0.5,0.0,0.003996605685951749);
      // Nodes.IntPoint(6).Set(0.0,0.5,0.0,0.003996605685951749);
      // Nodes.IntPoint(7).Set(0.0,0.0,0.5,0.003996605685951749);
      // Nodes.IntPoint(8).Set(0.5,0.0,0.5,0.003996605685951749);
      // Nodes.IntPoint(9).Set(0.0,0.5,0.5,0.003996605685951749);
      // // faces
      // Nodes.IntPoint(10).Set(0.3333333333333333,0.3333333333333333,0.0,0.03300381860330423);
      // Nodes.IntPoint(11).Set(0.3333333333333333,0.0,0.3333333333333333,0.03300381860330423);
      // Nodes.IntPoint(12).Set(0.3333333333333333,0.3333333333333333,0.3333333333333333,0.03300381860330423);
      // Nodes.IntPoint(13).Set(0.0,0.3333333333333333,0.3333333333333333,0.03300381860330423); 

      // vertices
      Nodes.IntPoint(0).Set(0.0,0.0,0.0,0.0026679395344347597);
      Nodes.IntPoint(1).Set(1.0,0.0,0.0,0.0026679395344347597);
      Nodes.IntPoint(2).Set(0.0,1.0,0.0,0.0026679395344347597);
      Nodes.IntPoint(3).Set(0.0,0.0,1.0,0.0026679395344347597);
      // edges 
      Nodes.IntPoint(4).Set(0.5,0.0,0.0,0.003996605685951749);
      Nodes.IntPoint(5).Set(0.0,0.5,0.0,0.003996605685951749);
      Nodes.IntPoint(6).Set(0.0,0.0,0.5,0.003996605685951749);
      Nodes.IntPoint(7).Set(0.5,0.5,0.0,0.003996605685951749);
      Nodes.IntPoint(8).Set(0.5,0.0,0.5,0.003996605685951749);
      Nodes.IntPoint(9).Set(0.0,0.5,0.5,0.003996605685951749);
      // faces
      Nodes.IntPoint(10).Set(0.3333333333333333,0.3333333333333333,0.3333333333333333,0.03300381860330423);  
      Nodes.IntPoint(11).Set(0.0,0.3333333333333333,0.3333333333333333,0.03300381860330423); 
      Nodes.IntPoint(12).Set(0.3333333333333333,0.0,0.3333333333333333,0.03300381860330423);
      Nodes.IntPoint(13).Set(0.3333333333333333,0.3333333333333333,0.0,0.03300381860330423); 
      break;
      
   default:
      mfem_error(
          "SBP elements are currently only supported for 0 <= order <= 1");
      break;
   }
   // populate unordered_map with mapping from IntPoint address to index
   for (int i = 0; i < dof; ++i)
   {
      ipIdxMap[&(Nodes.IntPoint(i))] = i;
   }

   for (int i = 0; i < dof; ++i)
   {
      const IntegrationPoint &ip = Nodes.IntPoint(i);
      H(i)    = ip.weight;
      x(i, 0) = ip.x;
      x(i, 1) = ip.y;
      x(i, 2) = ip.z;
   }
   // Construct the Vandermonde matrix in order to perform LPS projections;
   V.SetSize(num_nodes, (degree + 1) * (degree + 2) * (degree + 3)/ 6);
   // First, get node coordinates and shift to triangle with vertices
   // (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
   Vector xi;
   Vector eta;
   Vector zeta;
   getNodeCoords(0, xi);
   getNodeCoords(1, eta);
   getNodeCoords(2, zeta);
   xi *= 2.0;
   xi -= 1.0;
   eta *= 2.0;
   eta -= 1.0;
   zeta *= 2.0;
   zeta -= 1.0;
   mach::getVandermondeForTet(xi, eta, zeta, order, V);
   // scale V to account for the different reference elements
   V *= 2.0*sqrt(2.0);
 }

/// CalcShape outputs ndofx1 vector shape based on Kronecker \delta_{i, ip}
/// where ip is the integration point CalcShape is evaluated at.
void SBPTetrahedronElement::CalcShape(const IntegrationPoint &ip, Vector &shape) const
{
   int ipIdx = -1;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range &oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() (specifically
   // GridFunction::GetValues()), which calls CalcShape() with an
   // `IntegrationPoint` defined by a refined geometry type. Since the
   // IntegrationPoint is not in Nodes, its address is not in the ipIdxMap,
   // and an out_of_range error is thrown.
   {
      // This projects the SBP "basis" onto the degree = order orthogonal polys;
      // Such an approach is fine if LPS is used, but it will eliminate high
      // frequencey modes that may be present in the true solution.  It has
      // the advantage of being fast and not requiring a min-norm solution.
      Vector xvec(1);  // Vector with 1 entry (needed by prorioPoly)
      Vector yvec(1);
      Vector zvec(1);
      Vector poly(1);
      xvec(0) = 2 * ip.x - 1;
      yvec(0) = 2 * ip.y - 1;
      zvec(0) = 2 * ip.z - 1;
      int ptr = 0;
      shape = 0.0;
      for (int r = 0; r <= order; ++r)
      {
         for (int j = 0; j <= r; ++j)
         {  
            for (int k = 0; k <= r-j; ++k)
            {
               mach::prorioPoly(xvec, yvec, zvec, r-j-k, k, j, poly);
               poly *= 2.0*sqrt(2.0); // scale to mfem reference element

               for (int l = 0; l < GetDof(); ++l)
               {
                  shape(l) += poly(0) * V(l, ptr) * H(l);
               }
               ++ptr;               
            }
         }
      }
      return;
   }
   shape = 0.0;
   shape(ipIdx) = 1.0;
}

/// CalcDShape outputs ndof x ndim DenseMatrix dshape, where the first column
/// is the ith row of Dx, the second column is the ith row of Dy, and the  
/// third column is the ith row of Dz, where i is the integration point CalcDShape is evaluated at.
void SBPTetrahedronElement::CalcDShape(const IntegrationPoint &ip, DenseMatrix &dshape) const
{
   int ipIdx = -1;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range &oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() ->
   // GridFunction::GetValues() which calls CalcShape() with an
   // `IntegrationPoint` defined by a refined geometry type. Since the
   // IntegrationPoint is not in Nodes, its address is not in the ipIdxMap, and
   // an out_of_range error is thrown. This code catches the error and uses
   // float comparisons to determine the IntegrationPoint index.
   {
      double tol = 1e-12;
      for (int i = 0; i < dof; ++i)
      {
         double delta_x = ip.x - Nodes.IntPoint(i).x;
         double delta_y = ip.y - Nodes.IntPoint(i).y;
         double delta_z = ip.z - Nodes.IntPoint(i).z;
         if (delta_x * delta_x + delta_y * delta_y + delta_z * delta_z < tol)
         {
            ipIdx = i;
            break;
         }
      }
   }
   dshape = 0.0;

   Vector tempVec(dof);
   Q[0].GetColumnReference(ipIdx, tempVec);
   dshape.SetCol(0, tempVec);
   Q[1].GetColumnReference(ipIdx, tempVec);
   dshape.SetCol(1, tempVec);
   Q[2].GetColumnReference(ipIdx, tempVec);
   dshape.SetCol(2, tempVec);
   dshape.InvLeftScaling(H);
}

SBPCollection::SBPCollection(const int p, const int dim)
 : FiniteElementCollection(p)
{
   MFEM_VERIFY(p >= 0 && p <= 4, "SBPCollection requires 0 <= order <= 4.");
   MFEM_VERIFY(dim >=0 && dim <= 3, "SBPCollection requires 0 <= dim <= 3.");

   snprintf(SBPname, 32, "SBP_%dD_P%d", dim, p);

   for (int g = 0; g < Geometry::NumGeom; ++g)
   {
      SBPdof[g] = 0;
      SBPElements[g] = nullptr;
   }
   for (auto &i : SegDofOrd)
   {
      i = nullptr;
   }
   for (auto &i : TriDofOrd)
   {
      i = nullptr;
   }

   SBPdof[Geometry::POINT] = 1;
   SBPElements[Geometry::POINT] = new PointFiniteElement;

   if (dim >= 1)
   {
      SBPdof[Geometry::SEGMENT] = p;

      SBPElements[Geometry::SEGMENT] = new SBPSegmentElement(p);

      int nodeOrder0[] = {};
      int nodeOrder1[1] = {0};
      int nodeOrder2[2] = {0, 1};
      int nodeOrder3[3] = {0, 1, 2};
      int nodeOrder4[4] = {0, 1, 2, 3};

      int revNodeOrder0[] = {};
      int revNodeOrder1[1] = {0};
      int revNodeOrder2[2] = {1, 0};
      int revNodeOrder3[3] = {0, 2, 1};     // {1, 0, 2};    // {0, 2, 1};
      int revNodeOrder4[4] = {1, 0, 3, 2};  // {1, 0, 3, 2};

      switch (p)
      {
      case 0:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder0[i];
            SegDofOrd[1][i] = revNodeOrder0[i];
         }
         break;
      case 1:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder1[i];
            SegDofOrd[1][i] = revNodeOrder1[i];
         }
         break;
      case 2:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder2[i];
            SegDofOrd[1][i] = revNodeOrder2[i];
         }
         break;
      case 3:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder3[i];
            SegDofOrd[1][i] = revNodeOrder3[i];
         }
         break;
      case 4:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder4[i];
            SegDofOrd[1][i] = revNodeOrder4[i];
         }
         break;
      default:
         mfem_error(
             "SBP elements are currently only supported for 0 <= order <= 4");
         break;
      }
   }

   if (dim >= 2)
   {  
      switch (p)
      {
      case 0:
         SBPdof[Geometry::TRIANGLE] = 3 - 3 - 3 * p;
         break;
      case 1:
         SBPdof[Geometry::TRIANGLE] = 7 - 3 - 3 * p;
         break;
      case 2:
         SBPdof[Geometry::TRIANGLE] = 12 - 3 - 3 * p;
         break;
      case 3:
         SBPdof[Geometry::TRIANGLE] = 18 - 3 - 3 * p;
         break;
      case 4:
         SBPdof[Geometry::TRIANGLE] = 27 - 3 - 3 * p;
         break;
      default:
         mfem_error(
             "SBP elements are currently only supported for 0 <= order <= 4");
         break;
      }

      const int &TriDof = SBPdof[Geometry::TRIANGLE] +
                          3 * SBPdof[Geometry::POINT] +
                          3 * SBPdof[Geometry::SEGMENT];
      const int TriNodes = SBPdof[Geometry::TRIANGLE];
      if (p >=1)
      {
         TriDofOrd[0] = new int[6*TriNodes];
         for (int i = 1; i < 6; ++i)
         {
            TriDofOrd[i] = TriDofOrd[i-1] + TriNodes;
         }
         if (p==1)
         {
            TriDofOrd[0][0] = {0};
            TriDofOrd[1][0] = {0};
            TriDofOrd[2][0] = {0};
            TriDofOrd[3][0] = {0};
            TriDofOrd[4][0] = {0};
            TriDofOrd[5][0] = {0};            
         }
      }

      SBPElements[Geometry::TRIANGLE] = new SBPTriangleElement(p, TriDof);
   }

   if (dim >= 3)
   {
      switch (p)
      {
         case 0:
            SBPdof[Geometry::TETRAHEDRON] =  4 - 4 - (6 * p) - (4 * (3-3-3*p));
            break;
         case 1:
            SBPdof[Geometry::TETRAHEDRON] = 14 - 4 - (6 * p) - (4 * (7-3-3*p));
            break;
         default:
            mfem_error(
                  "SBP elements are currently only supported for 0 <= order <= 1");
            break;
      }

      const int &TetDof = SBPdof[Geometry::TETRAHEDRON] + 
                          4 * SBPdof[Geometry::POINT] + 
                          4 * SBPdof[Geometry::TRIANGLE] + 
                          6 * SBPdof[Geometry::SEGMENT];

      SBPElements[Geometry::TETRAHEDRON] = new SBPTetrahedronElement(p, TetDof);
   }
}

const FiniteElement *SBPCollection::FiniteElementForGeometry(
    Geometry::Type GeomType) const
{
   if (GeomType == Geometry::TRIANGLE || GeomType == Geometry::SEGMENT ||
       GeomType == Geometry::POINT || GeomType == Geometry::TETRAHEDRON)
   {
      return SBPElements[GeomType];
   }
   else
   {
      MFEM_ABORT("Unsupported geometry type " << GeomType);
      return nullptr;
   }
}

const int *SBPCollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                 int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   if (GeomType == Geometry::TRIANGLE)
   {
      return TriDofOrd[Or%6];
   }
   return nullptr;
}

SBPCollection::~SBPCollection()
{
   delete[] SegDofOrd[0];
   delete[] SegDofOrd[1];
   for (auto &SBPElement : SBPElements)
   {
      delete SBPElement;
   }
}

// From here thee DSBPCollection class
DSBPCollection::DSBPCollection(const int p, const int dim)
 : FiniteElementCollection(p)
{
   MFEM_VERIFY(p >= 0 && p <= 4, "SBPCollection requires 0 <= order <= 4.");
   MFEM_VERIFY(dim == 2, "SBPCollection requires dim == 2.");
   snprintf(DSBPname, 32, "DSBP_%dD_P%d", dim, p);
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      DSBPElements[g] = nullptr;
      Tr_SBPElements[g] = nullptr;
   }
   for (auto &i : SegDofOrd)
   {
      i = nullptr;
   }
   if (dim >= 1)
   {
      DSBPdof[Geometry::POINT] = 0;
      DSBPdof[Geometry::SEGMENT] = 0;

      DSBPElements[Geometry::POINT] = new PointFiniteElement;
      DSBPElements[Geometry::SEGMENT] = new SBPSegmentElement(p);
      Tr_SBPElements[Geometry::POINT] = new PointFiniteElement;
      int nodeOrder0[] = {};
      int nodeOrder1[1] = {0};
      int nodeOrder2[2] = {0, 1};
      int nodeOrder3[3] = {0, 1, 2};
      int nodeOrder4[4] = {0, 1, 2, 3};

      int revNodeOrder0[] = {};
      int revNodeOrder1[1] = {0};
      int revNodeOrder2[2] = {1, 0};
      int revNodeOrder3[3] = {0, 2, 1};     //{1, 0, 2};    // {0, 2, 1};
      int revNodeOrder4[4] = {1, 0, 3, 2};  // {1, 0, 3, 2};
      // set the dof order
      switch (p)
      {
      case 0:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder0[i];
            SegDofOrd[1][i] = revNodeOrder0[i];
         }
         break;
      case 1:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder1[i];
            SegDofOrd[1][i] = revNodeOrder1[i];
         }
         break;
      case 2:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder2[i];
            SegDofOrd[1][i] = revNodeOrder2[i];
         }
         break;
      case 3:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder3[i];
            SegDofOrd[1][i] = revNodeOrder3[i];
         }
         break;
      case 4:
         SegDofOrd[0] = new int[p];
         SegDofOrd[1] = new int[p];
         for (int i = 0; i < p; i++)
         {
            SegDofOrd[0][i] = nodeOrder4[i];
            SegDofOrd[1][i] = revNodeOrder4[i];
         }
         break;
      default:
         mfem_error(
             "SBP elements are currently only supported for 0 <= order <= 4");
         break;
      }
   }

   // two dimensional sbp triangle element
   if (dim >= 2)
   {
      switch (p)
      {
      case 0:
         DSBPdof[Geometry::TRIANGLE] = 3;
         break;
      case 1:
         DSBPdof[Geometry::TRIANGLE] = 7;
         break;
      case 2:
         DSBPdof[Geometry::TRIANGLE] = 12;
         break;
      case 3:
         DSBPdof[Geometry::TRIANGLE] = 18;
         break;
      case 4:
         DSBPdof[Geometry::TRIANGLE] = 27;
         break;
      default:
         mfem_error(
             "SBP elements are currently only supported for 0 <= order <= 4");
         break;
      }
      const int &TriDof = DSBPdof[Geometry::TRIANGLE] +
                          3 * DSBPdof[Geometry::POINT] +
                          3 * DSBPdof[Geometry::SEGMENT];
      DSBPElements[Geometry::TRIANGLE] = new SBPTriangleElement(p, TriDof);
      Tr_SBPElements[Geometry::SEGMENT] = new SBPSegmentElement(p);
   }
}

const int *DSBPCollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                  int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   return nullptr;
}

DSBPCollection::~DSBPCollection()
{
   delete[] SegDofOrd[0];
   delete[] SegDofOrd[1];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete DSBPElements[g];
      delete Tr_SBPElements[g];
   }
}
}  // namespace mfem
