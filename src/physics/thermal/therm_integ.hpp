// #ifndef MISO_THERM_INTEG
// #define MISO_THERM_INTEG

// #ifdef MFEM_USE_PUMI
// #include <unordered_set>

// #include "mfem.hpp"
// #include "apfMDS.h"

// namespace miso
// {

// class InteriorBoundaryOutFluxInteg : public mfem::BilinearFormIntegrator
// {
// protected:
//    mfem::Coefficient &K;
//    mfem::Coefficient &H;
//    const int source_attr;
//    const double ambient_temp;
//    const std::unordered_set<int> faces;
//    apf::Mesh2 *pumi_mesh;

//    // these are not thread-safe!
//    mfem::Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh;
//    mfem::DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

// public:
//    InteriorBoundaryOutFluxInteg(mfem::Coefficient &k,
//                                 mfem::Coefficient &h,
//                                 int source_region,
//                                 double ambient_temp,
//                                 std::unordered_set<int> faces,
//                                 apf::Mesh2 *pumi_mesh)
//       : K(k), H(h), source_attr(source_region), ambient_temp(ambient_temp),
//       faces(faces), pumi_mesh(pumi_mesh)
//    { }

//    void AssembleFaceVector(const mfem::FiniteElement &el1,
//                            const mfem::FiniteElement &el2,
//                            mfem::FaceElementTransformations &face_trans,
//                            const mfem::Vector &elfun,
//                            mfem::Vector &elvect) override;

//    void AssembleFaceMatrix(const mfem::FiniteElement &el1,
//                            const mfem::FiniteElement &el2,
//                            mfem::FaceElementTransformations &face_trans,
//                            mfem::DenseMatrix &elmat) override;
// };

// } // namespace miso

// #endif

// #endif
