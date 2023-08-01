from setuptools import setup
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open('miso/__init__.py').read(),
)[0]

setup(name='miso',
      version=__version__,

      description="""MISO is a C++ library for multi-physics finite-element simulations developed by the Optimal Design Lab at Rensselaer Polytechnic Institute based on Lawrence Livermore National Lab\'s MFEM code.""",
      long_description="""MISO is a C++ library for multi-physics finite-element simulations developed by the Optimal Design Lab at Rensselaer Polytechnic Institute based on Lawrence Livermore National Lab\'s MFEM code.
      For fluids, it solves the compressible Euler and Navier-Stokes equations. MISO also solves magnetostatic and thermal equations for electromagnetic and thermal simulations.
      MISO has implemented the discrete adjoint allowing for analytical derivative evaluation.""",

      long_description_content_type="text/markdown",
      keywords='adjoint optimization',
      author='',
      author_email='',
      url='https://github.com/optimaldesignlab/miso',
      license='',
      packages=[
          'miso',
      ],
      package_data={
          'miso': ['*.so']
      },
      install_requires=[
            'mpi4py>=3.0.2',
      ],
      classifiers=[
        "Programming Language :: Python, C++"]
      )
