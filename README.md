# ODL Mach #

Mach is a C++ library for multi-physics finite-element simulations based on LLNL's [MFEM](https://github.com/mfem/mfem).

## Build Instructions ##

### Dependencies: ###

* Required: [MFEM](https://github.com/mfem/mfem), [Adept](https://github.com/rjhogan/Adept-2), and [CMake](https://cmake.org) version 3.13 or newer
* Optional: [PUMI](https://github.com/scorec/core)

### Configuration: ###

CMake is used to configure and build. An example config file located in the `build/` directory looks like:

```cmake
cmake .. \
 -DADEPT_DIR="/path/to/adept/installation/" \
 -DMFEM_DIR="/path/to/mfem/installation/" \
 -DPUMI_DIR="/path/to/pumi/installation/" \ # if MFEM built with PUMI
```

Source this config file from the build directory to let CMake configure the build. If MFEM was built with MPI, CMake will find the MPI compilers on your system. If MFEM was built with PUMI, the `PUMI_DIR` must be specified. You can use a front-end like `ccmake` to see a full list of options.

### Build: ###

Once configured, execute `make` from the build directory to build Mach. As usual, you can also use the `-j` argument to build in parallel.

If using a build system other than GNU `make`, you can let CMake pick the computer's default build system with a command like `cmake --build . -j 4 --target install`. This tells CMake to `build` the project in the `.` (build) directory, in parallel using 4 processes, with the `install` target.

### Tests: ###

The test/ subdirectory has unit and regression tests. These are not included in the default `make` target, but can be built and run by executing `make tests`.

### Installation: ###

To install Mach in a specific location, add `-DCMAKE_INSTALL_PREFIX="/path/to/mach/install"` to the configuration file. If not specified, the install directory will be the root directory, and the library will be installed to `lib/`, and the header files will be copied to `include/`. Use `make install` to install the library.

TODO: Install pkg-config files so it is easier to use Mach in other CMake projects.

### Documentation: ###

Use the build target `doc` to build the doxygen documentaion for Mach.


### Sandbox: ###

Use the build target `sandbox` to build the sandbox. The executables are built in the `/build/sandbox` directory, and the options files are copied over.
