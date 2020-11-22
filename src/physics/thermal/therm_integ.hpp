// #ifndef MACH_THERM_INTEG
// #define MACH_THERM_INTEG

// #include "mfem.hpp"

// #include "coefficient.hpp"

// namespace mach
// {

// /// Integrator for the form:
// /// {w, (K grad(u)).n}_\Gamma to 
// class InteriorBoundaryOutFluxInteg : public BilinearFormIntegrator
// {
// protected:
//    Coefficient *K;
//    const int source_attr;

//    // these are not thread-safe!
//    Vector shape1, shape2, dshape1dn, dshape2dn, nor, nh, ni;
//    DenseMatrix jmat, dshape1, dshape2, mq, adjJ;

// public:
//    InteriorBoundaryOutFluxInteg(Coefficient &k, int source_region)
//       : K(&k) source_attr(source_region) { }

//    // void AssembleFaceVector(const FiniteElement &el1,
//    //                         const FiniteElement &el2,
//    //                         FaceElementTransformations &Trans,
//    //                         DenseMatrix &elmat) override;

//    void AssembleFaceMatrix(const FiniteElement &el1,
//                            const FiniteElement &el2,
//                            FaceElementTransformations &Trans,
//                            DenseMatrix &elmat) override;
// };

// } // namespace mach

// #endif
