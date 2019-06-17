#ifndef MFEM_SBP_FE
#define MFEM_SBP_FE

#include <unordered_map> // TODO: delete when we re-implement SBP elements?
#include "mfem.hpp"

// We add to the mfem namespace, since this seems appropriate; it will also
// simplify things if we ever port the SBP stuff to MFEM
namespace mfem
{

/// Abstract class for diaognal norm summation-by-parts (SBP) elements
class SBPFiniteElement : public NodalFiniteElement
{
public:
   /// Constructor
   /// \param[in] dim - element reference dimension
   /// \param[in] geo - reference geometry type (`SEGMENT`, `TRIANGLE`, etc.)
   /// \param[in] num_nodes - number of nodal degrees of freedom on element
   /// \param[in] degree - maximum degree for which the operator is exact
   /// \param[in] fspace - needed by `FiniteElement` base class only
   SBPFiniteElement(int dim, Geometry::Type geo, int num_nodes, int degree,
                    int fspace = FunctionSpace::Pk)
       : NodalFiniteElement(dim, geo, num_nodes, degree, fspace),
         H(num_nodes), x(num_nodes, dim) {}

   /// Returns the diagonal norm/mass matrix as a vector.
   const Vector &returnNormMatrix() const { return H; }

   /// Returns the nodes (in reference space) as a Vector.
   /// \param[in] di - reference coordinate desired
   /// This should be contrasted with `FiniteElement`'s GetNodes, which returns
   /// an IntegrationRule.
   void returnNodes(int di, mfem::Vector &x_di) 
   {
      // TODO: would like returnNodes to be const member, but cannot ...
      return x.GetColumnReference(di, x_di);
   }

   /// Sets `D` to be the derivative operator in direction `di`.
   /// \param[in] di - desired reference direction for operator
   /// \param[in,out] D - to store the operator
   /// \param[in] trans - if true, return \f$ D^T \f$.
   virtual void getStrongOperator(int di, DenseMatrix &D,
                                  bool trans = false) const = 0;

   /// Sets `Q` to be the weak derivative operator in direction `di`.
   /// \param[in] di - desired reference direction for operator
   /// \param[in,out] Q - to store the operator
   /// \param[in] trans - if true, return \f$ Q^T \f$.
   virtual void getWeakOperator(int di, DenseMatrix &Q,
                                bool trans = false) const = 0;

protected:
   /// the diagonal norm matrix stored as a vector
   Vector H;
   /// node coordinates of the reference element
   DenseMatrix x;
};

/// Class for summation-by-parts operator on interval
class SBPSegmentElement : public NodalTensorFiniteElement //, public SBPFiniteElement
{
public:
   SBPSegmentElement(const int p);
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   // TODO: tempoarily just an empty function to compile
   virtual void getStrongOperator(int di, DenseMatrix &D,
                                  bool trans = false) const {}
   // TODO: tempoarily just an empty function to compile
   virtual void getWeakOperator(int di, DenseMatrix &Q,
                                bool trans = false) const {}

private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, dshape_x;
#endif
};

/// Class for (diagonal-norm) summation-by-parts operator on triangles
class SBPTriangleElement : public SBPFiniteElement
{
public:
   /// Constructor for SBP operator on triangles (so-called "diagonal E")
   /// \param[in] degree - maximum poly degree for which operator is exact
   /// \param[in] num_nodes - the number of nodes the operator has
   SBPTriangleElement(const int degree, const int num_nodes);
   
   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;

   /// Get the derivative operator in the direction di; transposed if trans=true
   void GetOperator(int di, DenseMatrix &D, bool trans=false) const;

   virtual void getStrongOperator(int di, DenseMatrix &D,
                                  bool trans = false) const;
                                  
   virtual void getWeakOperator(int di, DenseMatrix &Q,
                                bool trans = false) const;

private:
#ifndef MFEM_THREAD_SAFE
   mutable Vector shape_x, shape_y, shape_l, dshape_x, dshape_y, dshape_l, u;
   mutable Vector ddshape_x, ddshape_y, ddshape_l;
   mutable DenseMatrix du, ddu;
#endif
   mutable DenseMatrix Qx, Qy;
   std::unordered_map<const IntegrationPoint*, int> ipIdxMap;
};

/// High order H1-conforming (continuous) Summation By Parts
/// operators.
/// Todo: members do not follow our naming convention
class SBPCollection : public FiniteElementCollection
{

protected:
   char SBPname[32];
   FiniteElement *SBPElements[Geometry::NumGeom];
   int  SBPdof[Geometry::NumGeom];
   int *SegDofOrd[2];

public:
   explicit SBPCollection(const int p, const int dim = 2);

   virtual const FiniteElement *FiniteElementForGeometry(
      Geometry::Type GeomType) const;
   virtual int DofForGeometry(Geometry::Type GeomType) const
   { return SBPdof[GeomType]; }
   virtual const int *DofOrderForOrientation(Geometry::Type GeomType,
                                             int Or) const;
   virtual const char *Name() const { return SBPname; }
   virtual ~SBPCollection();

};

} // namespace mfem

#endif