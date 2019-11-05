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
         H(num_nodes), x(num_nodes, dim), Q(dim) {}

   /// Returns the diagonal norm/mass matrix as a vector.
   const Vector &returnDiagNorm() const { return H; }

   double getDiagNormEntry(int i) const { return H(i); }

   /// Apply the norm matrix to given vector
   /// \param[in] u - `num_state` x `num_node` matrix of data being multiplied
   /// \param[out] Hu - result of applying `H` to `u`
   /// \note This is implemented in-place
   void multNormMatrix(const DenseMatrix &u, DenseMatrix &Hu) const;

   /// Apply the inverse norm matrix to given vector
   /// \param[in] u - `num_state` x `num_node` matrix of data being multiplied
   /// \param[out] Hinvu - result of applying `H^{-1}` to `u`
   /// \note This is implemented in-place
   void multNormMatrixInv(const DenseMatrix &u, DenseMatrix &Hinvu) const;

   /// Get a reference to the nodes coordinates as an `mfem::Vector`.
   /// \param[in] di - reference coordinate desired
   /// \param[out] x_di - vector reference to node coordinates
   /// This should be contrasted with `FiniteElement`'s GetNodes, which returns
   /// an IntegrationRule.
   void getNodeCoords(int di, mfem::Vector &x_di) const
   {
      // TODO: would like getNodeCoords to be const member, but cannot ...
      x.GetColumn(di, x_di);
   }

   /// Sets `D` to be the derivative operator in direction `di`.
   /// \param[in] di - desired reference direction for operator
   /// \param[in,out] D - to store the operator
   /// \param[in] trans - if true, return \f$ D^T \f$.
   void getStrongOperator(int di, DenseMatrix &D, bool trans = false) const;

   /// Sets `Q` to be the weak derivative operator in direction `di`.
   /// \param[in] di - desired reference direction for operator
   /// \param[in,out] Q - to store the operator
   /// \param[in] trans - if true, return \f$ Q^T \f$.
   void getWeakOperator(int di, DenseMatrix &Q, bool trans = false) const;

   /// Applies the weak derivative, `Q` or `Q^T`, to the given vector
   /// \param[in] di - desired reference space direction for operator
   /// \param[in] num_state - number of state variables at each node
   /// \param[in] u - vector that is being multiplied
   /// \param[out] Qu - result of applying `Q` or `Q^T` to `u`
   /// \param[in] trans - if `true` applies `Q^T`, otherwise applies `Q`
   //void multWeakOperator(int di, int num_state, const Vector &u, Vector &Qu,
   //                      bool trans = false) const;

   /// Applies the weak derivative, `Q` or `-Q^T`, to the given data
   /// \param[in] di - desired reference space direction for operator
   /// \param[in] u - `num_state` x `num_node` matrix of data being multiplied
   /// \param[out] Qu - result of applying `Q` or `-Q^T` to `u` is added here
   /// \param[in] trans - if `true` applies `-Q^T`, otherwise applies `Q`
   /// \warning The result of the operation is **added** to `Qu`, so the user is
   /// responsible for initializing `Qu`.
   void multWeakOperator(int di, const DenseMatrix &u, DenseMatrix &Qu,
                         bool trans = false) const;

   /// Sets `P` to be the operator that removes polynomials of degree `order`
   /// \param[in,out] P - to store the operator
   void getProjOperator(DenseMatrix &P) const;

   /// Returns the `i`th row `j`th column entry of the projection operator
   /// \param[in] i - desired row
   /// \param[in] j - desired column
   /// \returns \f$ P_{ij} \f$
   double getProjOperatorEntry(int i, int j) const;

   /// Applies the local projection operator, `P` or `P^T`, to the given data
   /// \param[in] u - `num_state` x `num_node` matrix of data being multiplied
   /// \param[out] Pu - result of applying `P` or `P^T` to `u` is stored here
   /// \param[in] trans - if `true` applies `P^T`, otherwise applies `P`
   /// \warning Pu is overwritten and possibly resized.
   void multProjOperator(const DenseMatrix &u, DenseMatrix &Pu,
                         bool trans = false) const;

   /// Returns the `(i,j)`th entry of the weak derivative in direction `di`
   /// \param[in] di - desired reference space direction of operator entry
   /// \param[in] i - row index
   /// \param[in] j - column index
   /// \returns \f$ (Q_{di})_{i,j} \f$ in reference space
   double getQ(int di, int i, int j) const;

   /// `(i,j)`th entry of skew-symmetric matrix \f$ S_{di} \f$ in physical space
   /// \param[in] di - desired physical space coordinate direction
   /// \param[in] i - row index for \f$ S_{di} \f$
   /// \param[in] j - column index for \f$ S_{di} \f$
   /// \param[in] adjJ_i - adjugate of the mapping Jacobian at node `i`
   /// \param[in] adjJ_j - adjugate of the mapping Jacobian at node `j`
   /// \returns \f$ (S_{di})_{i,j} \f$ in physical space
   /// \note The factor of 1/2 is missing, because there is a factor of 2 in
   /// two-point fluxes that cancel. 
   double getSkewEntry(int di, int i, int j, const mfem::DenseMatrix &adjJ_i,
                       const mfem::DenseMatrix &adjJ_j) const;

   /// Attempts to find the index corresponding to a given IntegrationPoint
   /// \param[in] ip - try to match the coordinates of this point
   /// \returns index - the index of the node corresponding to `ip`
   /// \note If `ip` is not a node, this function throws an exception.
   int getIntegrationPointIndex(const IntegrationPoint &ip) const;

protected:
   /// maps from integration points to integer index
   std::unordered_map<const IntegrationPoint*, int> ipIdxMap;
   /// the diagonal norm matrix stored as a vector
   Vector H;
   /// node coordinates of the reference element (0,0), (1,0), (0,1)
   mutable DenseMatrix x;
   /// difference operator(s); the transposed operators are stored in practice
   mutable Array<DenseMatrix> Q;
   /// generalized Vandermonde matrix; used for the projection operator in LPS
   mutable DenseMatrix V;
};

// /// Class for summation-by-parts operator on interval
// class SBPSegmentElement : public NodalTensorFiniteElement //, public SBPFiniteElement
// {
// public:
//    SBPSegmentElement(const int p);
//    virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
//    virtual void CalcDShape(const IntegrationPoint &ip,
//                            DenseMatrix &dshape) const;

//    // TODO: tempoarily just an empty function to compile
//    virtual void getStrongOperator(int di, DenseMatrix &D,
//                                   bool trans = false) const {}
//    // TODO: tempoarily just an empty function to compile
//    virtual void getWeakOperator(int di, DenseMatrix &Q,
//                                 bool trans = false) const {}

// private:
// #ifndef MFEM_THREAD_SAFE
//    mutable Vector shape_x, dshape_x;
// #endif
// };

/// Class for summation-by-parts operator on interval
class SBPSegmentElement : public SBPFiniteElement
{
public:
   /// Constructor for SBP operator on segments (so-called "diagonal E")
   /// \param[in] degree - maximum poly degree for which operator is exact
   /// \note a degree p "diagonal E" SBP segment is equivalent to a degree
   /// p+1 LGL collocation element
   SBPSegmentElement(const int degree);

   virtual void CalcShape(const IntegrationPoint &ip, Vector &shape) const;
   virtual void CalcDShape(const IntegrationPoint &ip,
                           DenseMatrix &dshape) const;
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