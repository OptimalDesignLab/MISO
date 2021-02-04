cmake .. \
 -DCMAKE_BUILD_TYPE=Debug \
 -DADEPT_DIR="/users/babcot/Developer/adept-install/" \
 -DMFEM_DIR="/users/bedong/Builds/mfem-install/" \
 -DPUMI_DIR="/lore/babcot/core/build/install/" \
 -DMFEM_USE_EGADS=YES \
 -DEGADS_DIR="/lore/babcot/MeshGen/EngSketchPad/" \
# -DEGADS_INCLUDE_DIR=/users/bedong/Builds/EngSketchPad/include \
# -DEGADS_LIBRARY=/users/bedong/Builds/EngSketchPad/lib/libegads.so \
 -DBUILD_TESTING="YES" 
