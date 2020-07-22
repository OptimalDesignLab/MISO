from _pyMach import *
from mpi4py import MPI


# """Class that wraps the C++ Abstract Solver"""
# class machSolver:
#     """Class that wraps the C++ Abstract Solver"""

#     def setMeshCoordinates(self, vol_mesh_coords):
#         pass

#     def calcResidual(self, state, residual):
#         pass

#     def calcState(self, state):
#         pass

#     def multStateJacTranspose(self, x, y):
#         """
#         Multiply :math:`\\mathbf{x}` by the transpose of :math:`\\frac{\\partial \\mathbf{R_{u}}}{\\partial \\mathbf{u}}`.
        
#         .. math:: \\mathbf{y} = \\frac{\\partial \\mathbf{R_{u}}}{\\partial \\mathbf{u}}^T \\mathbf{x}
#         """
#         pass

#     def multMeshJacTranspose(self, x, y):
#         """
#         Multiply :math:`\\mathbf{x}` by the transpose of :math:`\\frac{\\partial \\mathbf{R_{u}}}{\\partial \\mathbf{X}}`.
        
#         .. math:: \\mathbf{y} = \\frac{\\partial \\mathbf{R_{u}}}{\\partial \\mathbf{X}}^T \\mathbf{x}
#         """
#         pass

#     def invertStateJacTranspose(self, x, y):
#         """
#         Multiply :math:`\\mathbf{x}` by the inverse transpose of :math:`\\frac{\\partial \\mathbf{R_{u}}}{\\partial \\mathbf{u}}`.
        
#         .. math:: \\mathbf{y} = \\frac{\\partial \\mathbf{R_{u}}}{\\partial \\mathbf{u}}^{-T} \\mathbf{x}
#         """
#         pass

#     def getLocalMeshSize(self):
#         return 1
    
#     def getLocalStateSize(self):
#         return 1