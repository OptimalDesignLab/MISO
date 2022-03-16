target_sources(pyMach
   PUBLIC
      "pyMach.cpp"
      "field.cpp"
      "machSolver.cpp"
      "vector.cpp"
      "mesh.cpp"
      # "meshMotion.cpp"
      "mpi4py_comm.hpp"
      "pybind11_json.hpp"
)


