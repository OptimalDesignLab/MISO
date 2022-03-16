#[=======================================================================[.rst:
FindAdept
---------

Find Adept include dirs and libraries

Use this module by invoking :command:`find_package` with the form:

.. code-block:: cmake

  find_package(Adept
    [version] [EXACT]      # Minimum or EXACT version e.g. 2.1.0
    [REQUIRED]             # Fail with error if Adept is not found
    [COMPONENTS <libs>...] # Adept libraries by their canonical name
                           # e.g. "adept" for "libadept"
    [OPTIONAL_COMPONENTS <libs>...]
                           # Optional Adept libraries by their canonical name
  )                        # e.g. "adept" for "libadept"

This module finds headers and requested component libraries from Adept

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Adept_FOUND``
  True if headers and requested libraries were found.

``Adept_INCLUDE_DIRS``
  Adept include directories.

``Adept_LIBRARY_DIRS``
  Link directories for Adept libraries.

``Adept_LIBRARIES``
  Adept component libraries to be linked.

``Adept_<COMPONENT>_FOUND``
  True if component ``<COMPONENT>`` was found.

``Adept_<COMPONENT>_LIBRARY``
  Libraries to link for component ``<COMPONENT>`` (may include
  :command:`target_link_libraries` debug/optimized keywords).

Cache variables
^^^^^^^^^^^^^^^

Search results are saved persistently in CMake cache entries:

``Adept_INCLUDE_DIR``
  Directory containing Adept headers.

``Adept_LIBRARY_DIR``
  Directory containing Adept libraries.

Hints
^^^^^

This module reads hints about search locations from variables:

``Adept_ROOT``, ``AdeptROOT``
  Preferred installation prefix.

``Adept_INCLUDEDIR``
  Preferred include directory e.g. ``<prefix>/include``.

``Adept_LIBRARYDIR``
  Preferred library directory e.g. ``<prefix>/lib``.

``Adept_NO_SYSTEM_PATHS``
  Set to ``ON`` to disable searching in locations not
  specified by these hint variables. Default is ``OFF``.

``Adept_ADDITIONAL_VERSIONS``
  List of Adept versions not known to this module.
  (Adept install locations may contain the version).

Users may set these hints or results as ``CACHE`` entries.  Projects
should not read these entries directly but instead use the above
result variables.  Note that some hint names start in upper-case
``Adept``.  One may specify these as environment variables if they are
not specified as CMake variables or cache entries.

This module first searches for the Adept header files using the above
hint variables (excluding ``Adept_LIBRARYDIR``) and saves the result in
``Adept_INCLUDE_DIR``.  Then it searches for requested component libraries
using the above hints (excluding ``Adept_INCLUDEDIR``), "lib" directories 
near ``Adept_INCLUDE_DIR``, and the library name configuration settings below. 
It saves the library directories in ``Adept_LIBRARY_DIR`` and individual library
locations in ``Adept_<COMPONENT>_LIBRARY``.
When one changes settings used by previous searches in the same build
tree (excluding environment variables) this module discards previous
search results affected by the changes and searches again.

Imported Targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``Adept::adept``
  Target for adept (shared or static library).

It is important to note that the imported targets behave differently
than variables created by this module: multiple calls to
:command:`find_package(Adept)` in the same directory or sub-directories with
different options (e.g. static or shared) will not override the
values of the targets created by the first call.

Examples
^^^^^^^^

Find Adept libraries and use imported targets:

.. code-block:: cmake

  find_package(Adept REQUIRED)
  add_executable(foo foo.cc)
  target_link_libraries(foo Adept::adept)

#]=======================================================================]

include(GNUInstallDirs)

set(quiet "")
if(Adept_FIND_QUIETLY)
  set(quiet QUIET)
endif()

# ------------------------------------------------------------------------
# Find Adept include dir
# ------------------------------------------------------------------------
# message(STATUS "adept root (env): $ENV{Adept_ROOT}")
if(NOT Adept_INCLUDE_DIR)

  set(_Adept_INCLUDE_SEARCH_DIRS "")
  if(Adept_INCLUDEDIR)
    list(APPEND _Adept_INCLUDE_SEARCH_DIRS ${Adept_INCLUDEDIR})
  endif()

  if(DEFINED ENV{Adept_ROOT})
    list(APPEND _Adept_INCLUDE_SEARCH_DIRS $ENV{Adept_ROOT}/include $ENV{Adept_ROOT})
  endif()

  if(DEFINED ENV{AdeptROOT})
    list(APPEND _Adept_INCLUDE_SEARCH_DIRS $ENV{AdeptROOT}/include $ENV{AdeptROOT})
  endif()

  if(DEFINED Adept_ROOT)
    list(APPEND _Adept_INCLUDE_SEARCH_DIRS ${Adept_ROOT}/include $ENV{Adept_ROOT})
  endif()

  if(DEFINED AdeptROOT)
    list(APPEND _Adept_INCLUDE_SEARCH_DIRS ${AdeptROOT}/include $ENV{AdeptROOT})
  endif()


  find_path(Adept_INCLUDE_DIR NAMES adept.h HINTS ${_Adept_INCLUDE_SEARCH_DIRS})
endif()

# ------------------------------------------------------------------------
#  Begin finding Adept libraries
# ------------------------------------------------------------------------

# all potential Adept components
set(Adept_COMPONENTS adept)

# if not explicitly asking for any component, find all of them
if(NOT Adept_FIND_COMPONENTS)
  set(Adept_FIND_COMPONENTS ${Adept_COMPONENTS})
endif()

foreach(component ${Adept_FIND_COMPONENTS})
  
  find_library(${component}_LIBRARY NAMES ${component})

  if(${component}_LIBRARY)
    set(Adept_${component}_FOUND True)
  else()
    set(Adept_${component}_FOUND False)
  endif()

  # Create a library target only if the above checks passed
  if(Adept_${component}_FOUND AND NOT TARGET Adept::${component})
    # Can't easily tell how Adept was compiled, so just default to UNKNOWN
    # library type and CMake will make a best effort guess
    add_library(Adept::${component} UNKNOWN IMPORTED)

    set_property(
        TARGET Adept::${component} PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${Adept_INCLUDE_DIR}"
    )
    if(EXISTS "${${component}_LIBRARY}")
      set_property(
          TARGET Adept::${component} PROPERTY
          IMPORTED_LOCATION "${${component}_LIBRARY}"
      )
    endif()
  endif()
endforeach()

# Use CMake provided module to check the variables
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Adept
  REQUIRED_VARS Adept_INCLUDE_DIR
  # VERSION_VAR Adept_VERSION_STRING
  HANDLE_COMPONENTS
)

# # FindAdept.cmake
# #
# # Finds the Adept library
# #
# # This will define the following variables
# #
# #    ADEPT_FOUND
# #    ADEPT_LIBRARIES
# #    ADEPT_INCLUDE_DIRS
# #

# # look for Adept static library
# find_path(ADEPT_INCLUDE_DIR adept.h PATHS "${ADEPT_DIR}/include")
# find_library(ADEPT_LIBRARY NAMES libadept.so adept PATHS "${ADEPT_DIR}/lib")


# set(ADEPT_INCLUDE_DIRS "${ADEPT_INCLUDE_DIR}")
# set(ADEPT_LIBRARIES "${ADEPT_LIBRARY}")

# # cmake function to handle error if ADEPT_INCLUDE_DIR or ADEPT_LIBRARY were not found
# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(Adept
#    "Must specify Adept installation directory in cmake config file \ 
#       (-DADEPT_DIR=\"path/to/adept\")"
#    ADEPT_INCLUDE_DIR ADEPT_LIBRARY
# )

# mark_as_advanced(ADEPT_INCLUDE_DIR ADEPT_LIBRARY)
