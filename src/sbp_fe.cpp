#include "mfem.hpp"
#include "sbp_fe.hpp"

namespace mfem
{

using namespace std;

/// SBPSegmentElement is a segment element with nodes at Gauss Lobatto
/// points with ordering consistent with SBPTriangleElement's edges.

//////////////////////////////////////////////////////////////////////////
/// Not currently implemented as collocated SBP type element
//////////////////////////////////////////////////////////////////////////
SBPSegmentElement::SBPSegmentElement(const int p)
   : NodalTensorFiniteElement(1, p+1, BasisType::GaussLobatto, H1_DOF_MAP) //SBPFiniteElement(1, GetTensorProductGeometry(1), p+2, p),
{
   const double *cp = poly1d.ClosedPoints(p+1, b_type);

#ifndef MFEM_THREAD_SAFE
   shape_x.SetSize(p+2);
   dshape_x.SetSize(p+2);
#endif

   Nodes.IntPoint(0).x = cp[0];
   Nodes.IntPoint(1).x = cp[p+1];

   switch (p)
   {
      case 1:
         Nodes.IntPoint(2).x = cp[1];
         break;
      case 2:
         Nodes.IntPoint(2).x = cp[1];
         Nodes.IntPoint(3).x = cp[2];
         break;
      case 3:
         Nodes.IntPoint(2).x = cp[2];
         Nodes.IntPoint(3).x = cp[1];
         Nodes.IntPoint(4).x = cp[3];
         break;
      case 4:
         Nodes.IntPoint(2).x = cp[2];
         Nodes.IntPoint(3).x = cp[3];
         Nodes.IntPoint(4).x = cp[1];
         Nodes.IntPoint(5).x = cp[4];
         break;
   }
}

void SBPSegmentElement::CalcShape(const IntegrationPoint &ip,
                                  Vector &shape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+2);
#endif

   basis1d.Eval(ip.x, shape_x);

   shape(0) = shape_x(0);
   shape(1) = shape_x(p+1);

   switch (p)
   {
      case 1:
         shape(2) = shape_x(1);
         break;
      case 2:
         shape(2) = shape_x(1);
         shape(3) = shape_x(2);
         break;
      case 3:
         shape(2) = shape_x(2);
         shape(3) = shape_x(1);
         shape(4) = shape_x(3);
         break;
      case 4:
         shape(2) = shape_x(2);
         shape(3) = shape_x(3);
         shape(4) = shape_x(1);
         shape(5) = shape_x(4);
         break;
   }
}

void SBPSegmentElement::CalcDShape(const IntegrationPoint &ip,
                                   DenseMatrix &dshape) const
{
   const int p = Order;

#ifdef MFEM_THREAD_SAFE
   Vector shape_x(p+2), dshape_x(p+2);
#endif

   basis1d.Eval(ip.x, shape_x, dshape_x);

   dshape(0,0) = dshape_x(0);
   dshape(1,0) = dshape_x(p+1);

   switch (p)
   {
      case 1:
         dshape(2,0) = dshape_x(1);
         break;
      case 2:
         dshape(2,0) = dshape_x(1);
         dshape(3,0) = dshape_x(2);
         break;
      case 3:
         dshape(2,0) = dshape_x(2);
         dshape(3,0) = dshape_x(1);
         dshape(4,0) = dshape_x(3);
         break;
      case 4:
         dshape(2,0) = dshape_x(2);
         dshape(3,0) = dshape_x(3);
         dshape(4,0) = dshape_x(1);
         dshape(5,0) = dshape_x(4);
         break;
   }
}

// Leftover function from H1_Segment element
// void SBPSegmentElement::ProjectDelta(int vertex, Vector &dofs) const
// {
//    const int p = Order;
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
   : SBPFiniteElement(2, Geometry::TRIANGLE, num_nodes, degree),
   Dx(num_nodes), Dy(num_nodes)
{
   /// Header file including SBP Dx and Dy matrix data
   #include "sbp_operators.hpp"

   // Create Dx and Dy matrixes
   //Dx = new DenseMatrix(Dof);
   //Dy = new DenseMatrix(Dof);
   
   // Populate the Dx and Dy matrices and create the element's Nodes
   switch (degree)
   {
      case 0:
         Dx=p0Dx;
         Dy=p0Dy;

         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.16666666666666666);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.16666666666666666);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.16666666666666666);
         break;
      case 1:
         Dx=p1Dx;
         Dy=p1Dy;

         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.024999999999999998);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.024999999999999998);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.024999999999999998);
         Nodes.IntPoint(3).Set2w(0.5, 0.0, 0.06666666666666667);
         Nodes.IntPoint(4).Set2w(0.5, 0.5, 0.06666666666666667);
         Nodes.IntPoint(5).Set2w(0.0, 0.5, 0.06666666666666667);
         Nodes.IntPoint(6).Set2w(0.3333333333333333, 0.3333333333333333, 0.22500000000000006);
         break;
      case 2:
         Dx=p2Dx;
         Dy=p2Dy;
  
         // vertices
         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.006261126504899741);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.006261126504899741);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.006261126504899741);

         // edges
         Nodes.IntPoint(3).Set2w(0.27639320225002106, 0.0, 0.026823800250389242);
         Nodes.IntPoint(4).Set2w(0.7236067977499789, 0.0, 0.026823800250389242);
         Nodes.IntPoint(5).Set2w(0.7236067977499789, 0.27639320225002106, 0.026823800250389242);
         Nodes.IntPoint(6).Set2w(0.27639320225002106, 0.7236067977499789, 0.026823800250389242);
         Nodes.IntPoint(7).Set2w(0.0, 0.7236067977499789, 0.026823800250389242);
         Nodes.IntPoint(8).Set2w(0.0, 0.27639320225002106, 0.026823800250389242);

         // interior
         Nodes.IntPoint(9).Set2w(0.21285435711180825, 0.5742912857763836, 0.10675793966098839);
         Nodes.IntPoint(10).Set2w(0.21285435711180825, 0.21285435711180825, 0.10675793966098839);
         Nodes.IntPoint(11).Set2w(0.5742912857763836, 0.21285435711180825, 0.10675793966098839);
         break;
      case 3:
         Dx=p3Dx;
         Dy=p3Dy;
   
         // vertices
         Nodes.IntPoint(0).Set2w(0.0, 0.0, 0.0022825661430496253);
         Nodes.IntPoint(1).Set2w(1.0, 0.0, 0.0022825661430496253);
         Nodes.IntPoint(2).Set2w(0.0, 1.0, 0.0022825661430496253);

         // edges
         Nodes.IntPoint(3).Set2w(0.5, 0.0, 0.015504052643022513);
         Nodes.IntPoint(4).Set2w(0.17267316464601146, 0.0, 0.011342592592592586);
         Nodes.IntPoint(5).Set2w(0.8273268353539885, 0.0, 0.011342592592592586);

         Nodes.IntPoint(6).Set2w(0.5, 0.5, 0.015504052643022513);
         Nodes.IntPoint(7).Set2w(0.8273268353539885, 0.17267316464601146, 0.011342592592592586);
         Nodes.IntPoint(8).Set2w(0.17267316464601146, 0.8273268353539885, 0.011342592592592586);

         Nodes.IntPoint(9).Set2w(0.0, 0.5, 0.015504052643022513);
         Nodes.IntPoint(10).Set2w(0.0, 0.8273268353539885, 0.011342592592592586);
         Nodes.IntPoint(11).Set2w(0.0, 0.17267316464601146, 0.011342592592592586);

         // interior
         Nodes.IntPoint(12).Set2w(0.4243860251718814, 0.1512279496562372, 0.07467669469983994);
         Nodes.IntPoint(13).Set2w(0.4243860251718814, 0.4243860251718814, 0.07467669469983994);
         Nodes.IntPoint(14).Set2w(0.1512279496562372, 0.4243860251718814, 0.07467669469983994);

         Nodes.IntPoint(15).Set2w(0.14200508409677795, 0.7159898318064442, 0.051518167995569394);
         Nodes.IntPoint(16).Set2w(0.14200508409677795, 0.14200508409677795, 0.051518167995569394);
         Nodes.IntPoint(17).Set2w(0.7159898318064442, 0.14200508409677795, 0.051518167995569394);

         break;
      case 4:
         Dx=p4Dx;
         Dy=p4Dy; 

         // vertices
         Nodes.IntPoint(0).Set2w(0.000000000000000000,0.000000000000000000,0.001090393904993471);
         Nodes.IntPoint(1).Set2w(1.000000000000000000,0.000000000000000000,0.001090393904993471);
         Nodes.IntPoint(2).Set2w(0.000000000000000000,1.000000000000000000,0.001090393904993471);

         // edges
         Nodes.IntPoint(3).Set2w(0.357384241759677534,0.000000000000000000,0.006966942871463700);
         Nodes.IntPoint(4).Set2w(0.642615758240322466,0.000000000000000000,0.006966942871463700);
         Nodes.IntPoint(5).Set2w(0.117472338035267576,0.000000000000000000,0.005519747637357106);
         Nodes.IntPoint(6).Set2w(0.882527661964732424,0.000000000000000000,0.005519747637357106);

         Nodes.IntPoint(7).Set2w(0.642615758240322466,0.357384241759677534,0.006966942871463700);
         Nodes.IntPoint(8).Set2w(0.357384241759677534,0.642615758240322466,0.006966942871463700);
         Nodes.IntPoint(9).Set2w(0.882527661964732424,0.117472338035267576,0.005519747637357106);
         Nodes.IntPoint(10).Set2w(0.117472338035267576,0.882527661964732424,0.005519747637357106);

         Nodes.IntPoint(11).Set2w(0.000000000000000000,0.642615758240322466,0.006966942871463700);
         Nodes.IntPoint(12).Set2w(0.000000000000000000,0.357384241759677534,0.006966942871463700);
         Nodes.IntPoint(13).Set2w(0.000000000000000000,0.882527661964732424,0.005519747637357106);
         Nodes.IntPoint(14).Set2w(0.000000000000000000,0.117472338035267576,0.005519747637357106);

         // interior
         Nodes.IntPoint(15).Set2w(0.103677508142805172,0.792644983714389628,0.028397190663911491);
         Nodes.IntPoint(16).Set2w(0.103677508142805172,0.103677508142805172,0.028397190663911491);
         Nodes.IntPoint(17).Set2w(0.792644983714389628,0.103677508142805172,0.028397190663911491);
         Nodes.IntPoint(18).Set2w(0.265331380484209678,0.469337239031580644,0.039960048027851809);
         Nodes.IntPoint(19).Set2w(0.265331380484209678,0.265331380484209678,0.039960048027851809);
         Nodes.IntPoint(20).Set2w(0.469337239031580644,0.265331380484209678,0.039960048027851809);
         Nodes.IntPoint(21).Set2w(0.587085567133367348,0.088273960601581103,0.036122826526134168);
         Nodes.IntPoint(22).Set2w(0.324640472265051494,0.088273960601581103,0.036122826526134168);
         Nodes.IntPoint(23).Set2w(0.324640472265051494,0.587085567133367348,0.036122826526134168);
         Nodes.IntPoint(24).Set2w(0.587085567133367348,0.324640472265051494,0.036122826526134168);
         Nodes.IntPoint(25).Set2w(0.088273960601581103,0.324640472265051494,0.036122826526134168);
         Nodes.IntPoint(26).Set2w(0.088273960601581103,0.587085567133367348,0.036122826526134168);

         break;
      default:
         mfem_error("SBP elements are currently only supported for 0 <= order <= 4");
         break;
   }

   // populate unordered_map with mapping from IntPoint address to index
   for (int i = 0; i < Dof; i++)
   {
      ipIdxMap[&(Nodes.IntPoint(i))] = i;
   }
}

/// CalcShape outputs ndofx1 vector shape based on Kronecker \delta_{i, ip}
/// where ip is the integration point CalcShape is evaluated at. 
void SBPTriangleElement::CalcShape(const IntegrationPoint &ip,
                                   Vector &shape) const
{
   int ipIdx;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range& oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() ->  GridFunction::GetValues()
   // which calls CalcShape() with an `IntegrationPoint` defined by a refined
   // geometry type. Since the IntegrationPoint is not in Nodes, its address is
   // not in the ipIdxMap, and an out_of_range error is thrown. This code catches
   // the error and uses float comparisons to determine the IntegrationPoint
   // index.
   {
      double tol = 1e-12;
      for (int i = 0; i < Dof; i++)
      {
         double delta_x = ip.x - Nodes.IntPoint(i).x;
         double delta_y = ip.y - Nodes.IntPoint(i).y;
         if (delta_x*delta_x + delta_y*delta_y < tol)
         {
            ipIdx = i;
            break;
         }
      }
   }
   shape = 0.0;
   shape(ipIdx) = 1.0;
}

/// CalcDShape outputs ndof x ndim DenseMatrix dshape, where the first column
/// is the ith row of Dx, and the second column is the ith row of Dy, where i
/// is the integration point CalcDShape is evaluated at. Since DenseMatrices 
/// are stored a column major we should store the transpose so accessing a row
/// is faster, but this is not done here.
void SBPTriangleElement::CalcDShape(const IntegrationPoint &ip,
                                    DenseMatrix &dshape) const
{
   int ipIdx;
   try
   {
      ipIdx = ipIdxMap.at(&ip);
   }
   catch (const std::out_of_range& oor)
   // error handling code to handle cases where the pointer to ip is not
   // in the map. Problems arise in GridFunction::SaveVTK() ->  GridFunction::GetValues()
   // which calls CalcShape() with an `IntegrationPoint` defined by a refined
   // geometry type. Since the IntegrationPoint is not in Nodes, its address is
   // not in the ipIdxMap, and an out_of_range error is thrown. This code catches 
   // the error and uses float comparisons to determine the IntegrationPoint
   // index.
   {
      double tol = 1e-12;
      for (int i = 0; i < Dof; i++)
      {
         double delta_x = ip.x - Nodes.IntPoint(i).x;
         double delta_y = ip.y - Nodes.IntPoint(i).y;
         if (delta_x*delta_x + delta_y*delta_y < tol)
         {
            ipIdx = i;
            break;
         }
      }
   }
   dshape = 0.0;

   Vector tempVec(Dof);

   // when we switch to storing Dx and Dy transpose so that access to the row we want
   // is faster Dx->GetRow() will be replaced with Dx->GetColumnReference()
   Dx.GetRow(ipIdx, tempVec);
   dshape.SetCol(0, tempVec);
   Dy.GetRow(ipIdx, tempVec);
   dshape.SetCol(1, tempVec);
}

void SBPTriangleElement::GetOperator(int di, DenseMatrix &D, bool trans) const
{
   MFEM_ASSERT(di >= 0 && di <= 1, "");
   if (trans)
   {
      di == 0 ? D.Transpose(Dx) : D.Transpose(Dy); // this copies Dx^T, etc
   }
   else
   {
      di == 0 ? D = Dx : D = Dy; // assignment (deep copy)
   }
}

void SBPTriangleElement::GetDiagNorm(Vector &H) const
{
   /// TODO: if we decide to store H as a Vector, this needs to change
   H.SetSize(Dof);
   for (int i = 0; i < Dof; i++)
      H[i] = Nodes.IntPoint(i).weight;
}

SBPCollection::SBPCollection(const int p, const int dim)
{
   MFEM_VERIFY(p >= 0 && p <= 4, "SBPCollection requires 0 <= order <= 4.");
   MFEM_VERIFY(dim == 2, "SBPCollection requires dim == 2.");

   snprintf(SBPname, 32, "SBP_%dD_P%d", dim, p);

   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      SBPdof[g] = 0;
      SBPElements[g] = NULL;
   }
   for (int i = 0; i < 2; i++)
   {
      SegDofOrd[i] = NULL;
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
      int revNodeOrder3[3] = {0, 2, 1};
      int revNodeOrder4[4] = {1, 0, 3, 2};

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
            mfem_error("SBP elements are currently only supported for 0 <= order <= 4");
            break;

      }
   }

   if (dim >= 2)
   {
      switch (p)
      {
         case 0:
            SBPdof[Geometry::TRIANGLE] = 3 - 3 - 3*p;
            break;
         case 1:
            SBPdof[Geometry::TRIANGLE] = 7 - 3 - 3*p;
            break;
         case 2:
            SBPdof[Geometry::TRIANGLE] = 12 - 3 - 3*p;
            break;
         case 3:
            SBPdof[Geometry::TRIANGLE] = 18 - 3 - 3*p;
            break;
         case 4:
            SBPdof[Geometry::TRIANGLE] = 27 - 3 - 3*p;
            break;
         default:
            mfem_error("SBP elements are currently only supported for 0 <= order <= 4");
            break;
      }

      const int &TriDof = SBPdof[Geometry::TRIANGLE] + 3*SBPdof[Geometry::POINT] + 3*SBPdof[Geometry::SEGMENT];

      SBPElements[Geometry::TRIANGLE] = new SBPTriangleElement(p, TriDof);
   }
}

const FiniteElement *SBPCollection::FiniteElementForGeometry(
      Geometry::Type GeomType) const
{
   if (GeomType == Geometry::TRIANGLE || GeomType == Geometry::SEGMENT || GeomType == Geometry::POINT)
   {

   }
   else
   {
      MFEM_ABORT("Unsupported geometry type " << GeomType);
   }
   return SBPElements[GeomType]; 
}

const int *SBPCollection::DofOrderForOrientation(Geometry::Type GeomType,
                                                   int Or) const
{
   if (GeomType == Geometry::SEGMENT)
   {
      return (Or > 0) ? SegDofOrd[0] : SegDofOrd[1];
   }
   return NULL;
}

SBPCollection::~SBPCollection()
{
   delete [] SegDofOrd[0];
   for (int g = 0; g < Geometry::NumGeom; g++)
   {
      delete SBPElements[g];
   }
}

} // namespace mfem