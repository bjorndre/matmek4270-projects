import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sympy.utilities.lambdify import implemented_function
from poisson import Poisson

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, Lx] x [0, Ly]

    with homogeneous Dirichlet boundary conditions.
    """

    def __init__(self, Lx, Ly, Nx=None, Ny=None):
        self.px = Poisson(Lx, Nx) # we can reuse some of the code from the 1D case
        self.py = Poisson(Ly, Ny)

    def create_mesh(self, Nx, Ny):
        """Return a 2D Cartesian mesh
        """
        
        x = self.px.create_mesh(Nx)
        y = self.py.create_mesh(Ny)
        self.Nx = self.px.N
        self.Ny = self.py.N
        self.xij, self.yij = np.meshgrid(x,y, indexing="ij", sparse=True)
        return self.xij, self.yij

    def laplace(self):
        """Return a vectorized Laplace operator"""
        D2x = self.px.D2()
        D2y = self.py.D2()
        return (sparse.kron(D2x, sparse.eye(self.Ny+1)) + sparse.kron(sparse.eye(self.Nx+1), D2y))

    def assemble(self, f, xij, yij):
        """Return assemble coefficient matrix A and right hand side vector b"""
        B = np.ones((self.Nx+1, self.Ny+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel()==1)[0]
        A = self.laplace()
        
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        F = sp.lambdify((x, y), f)(xij, yij)
        b = F.ravel()
        b[bnds] = 0
        return A, b

    def l2_error(self, u, ue):
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy function
            The analytical solution
        """
        uj = sp.lambdify((x,y) , ue)(self.xij,self.yij)
        print(self.px.dx, self.py.dx)
        return np.sqrt(self.px.dx*self.py.dx*np.sum((u-uj)**2))

    def __call__(self, Nx, Ny, f=implemented_function('f', lambda x, y: 2)(x, y)):
        """Solve Poisson's equation with a given righ hand side function

        Parameters
        ----------
        N : int
            The number of uniform intervals
        f : Sympy function
            The right hand side function f(x, y)

        Returns
        -------
        The solution as a Numpy array

        """
        xij, yij = self.create_mesh(Nx, Ny)
        A, b = self.assemble(f, xij, yij)
        U = sparse.linalg.spsolve(A, b.ravel()).reshape((self.px.N+1, self.py.N+1))
        return U

def test_poisson2d():
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    Lx, Ly = 1, 1
    f = ue.diff(x, 2) + ue.diff(y, 2)

    sol = Poisson2D(Lx, Ly)
    u = sol(100, 100, f)
    tol = 1/1000
    assert sol.l2_error(u,ue) < tol


if __name__ == '__main__':
    ue = x*(1-x)*y*(1-y)*sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    Lx, Ly = 1, 1
    f = ue.diff(x, 2) + ue.diff(y, 2)

    sol = Poisson2D(Lx, Ly)
    u = sol(100, 100, f)

    print(sol.l2_error(u,ue))