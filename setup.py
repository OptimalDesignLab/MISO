from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('mach/__init__.py').read(),
)[0]

setup(name='mach',
      version=__version__,

      description="""Mach is a C++ library for multi-physics finite-element simulations developed by the Optimal Design Lab at Rensselaer Polytechnic Institute based on Lawrence Livermore National Lab\'s MFEM code.""",
      long_description="""Mach is a C++ library for multi-physics finite-element simulations developed by the Optimal Design Lab at Rensselaer Polytechnic Institute based on Lawrence Livermore National Lab\'s MFEM code.
      For fluids, it solves the compressible Euler and Navier-Stokes equations. Mach also solves magnetostatic and thermal equations for electromagnetic and thermal simulations.
      Mach has implemented the discrete adjoint allowing for analytical derivative evaluation.""",

      long_description_content_type="text/markdown",
      keywords='adjoint optimization',
      author='',
      author_email='',
      url='https://github.com/optimaldesignlab/mach',
      license='',
      packages=[
          'mach',
      ],
      package_data={
          'mach': ['*.so']
      },
      install_requires=[
            'mpi4py>=3.1.3',
      ],
      classifiers=[
        "Programming Language :: Python, C++"]
      )
