#ifndef MFEM_RBFSPACE
#define MFEM_RBFSPACE

#include "mfem.hpp"

namespace mfem
{

/// class for declaration for the Radial basis function space
/// The first version have radial basis function centers at
///  element centers
class RBFSpace : public mfem::FiniteElementSpace
{

public:
   /// The default constuctor
   /// \param[in] m - mfem mesh object pointer
   /// \param[in] f - pointer to FiniteElementCollection
   /// \param[in] nb - number of Radial basis functions
   /// \param[in] vdim - number of unknows per node
   /// \param[in] span_coef - the range that radial basis function span
   /// \param[in] ordering - orderding method for the state variables
   RBFSpace(mfem::Mesh *m, const mfem::FiniteElementCollection *f,
            int nb, int vdim = 1,
            double span_coef = 1.0, int ordering = mfem::Ordering::byVDIM);
   
   /// build the radial basis function prolongtaion
   void BuildRBFProlongation() const;

   /// Select the elements that are within the effective range
   /// \param[in] id - element id
   /// \param[in/out] basis_id - basis that are within the effective range
   /// \param[in/out] basis_coord - coordinates of selected basis
   /// \param[in/out] lam_selected - the selected lambda coefficient matrix
   void SelectElementBasis(const int id, mfem::Array<int> &basis_id,
                           mfem::Array<mfem::Vector> &basis_coord,
                           mfem::Array<mfem::Vector> &lam_selected) const;

   /// extract the element interpolation points
   /// \param[in] id - element id
   /// \param[in/out] inter_points - interpolation points on element id
   void GetElementInterPoints(const int id, mfem::Array<mfem::Vector> &inter_points) const;

   /// Assemble the local prolongation matrix back to the global matrix
   /// \param[in] id - element id
   /// \param[in] basis_selected - id of selected basis
   /// \param[in] local_prolong - local prolongation matrix
   void AssembleProlongationMatrix(const int id,
                                   const mfem::Array<int> &basis_selected,
                                   const mfem::DenseMatrix &local_prolong) const;
   
   virtual const Operator *GetProlongationMatrix() const
   { 
      if (!cP)
      {
         BuildRBFProlongation();
         return cP;
      }
      else
      {
         return cP; 
      }
   }
protected:
   /// problem dimension
   int dim;

   /// number of Radial functions basis
   int num_basis;

   /// Array that hold the basis centers
   //std::unique_ptr<mfem::Array<mfem::Vector>> basis_center;
   mfem::Array<mfem::Vector> basis_center;

   /// range that RBF span
   /// This could be an array that holding difference values
   double span;

   /// Shape parameter
   mfem::Array<mfem::Vector> lam;
   // std::unique_prt<mfem::Array<mfem::Vector>> lam;
   
   std::unique_ptr<mfem::Array<mfem::Vector>> a_test;
};

} // end of namespace mfem 
#endif
