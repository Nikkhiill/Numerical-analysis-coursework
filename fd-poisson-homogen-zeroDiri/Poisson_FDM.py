# Solving a 2D homogeneous Poisson's equation with a given source function
# and Zero & NON-ZERO Dirichlet B.C. using Finite Difference Method (FDM)

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la


def generate_grid(LeftX, LeftY, RightX, RightY, Nx, Ny):
    
    """ Inputs: X and Y co-ordinates of the domain and the no. of intervals in each direction 
    Returns arrays x and y of the grid points and dx,dy - the mesh size """

    dx = (RightX-LeftX)/Nx
    dy = (RightY-LeftY)/Ny

    x,y = np.mgrid[LeftX+dx:RightX:dx,LeftY+dy:RightY:dy]

    return x,y,dx,dy


def FDLaplacian2D(Nx,Ny,dx,dy):
    
    """ Inputs: no. of intervals Nx and Ny, and the resulting mesh size 
    Returns A - sparse csc format of the FD matrix """

    Dx = (1/dx)*(sp.diags([1,-1],[0,-1],shape=(Nx,Nx-1)))

    Dy = (1/dy)*(sp.diags([1,-1],[0,-1],shape=(Ny,Ny-1)))

    DxT = Dx.transpose()
    DyT = Dy.transpose()

    Lxx = DxT.dot(Dx)
    Lyy = DyT.dot(Dy)

    Ix = sp.eye(Nx-1,Nx-1)
    Iy = sp.eye(Ny-1,Ny-1)

    A = (sp.kron(Iy,Lxx) + sp.kron(Lyy,Ix)).tocsc()

    return A


def sourcefunc(x,y,Nx,Ny):
    
    """ Evaluates the given source function f on the generated grid points 
    f(x,y) = sum_{i=1..9} sum_{j=1..4} exp(alpha*(x-i)^2 + alpha*(y-j)^2) 
    Inputs: grid points x and y, and the no. of intervals Nx and Ny 
    Return an array f of the source function evaluated at grid points """

    alpha = -40

    f = np.zeros(shape=(Nx-1,Ny-1))

    for m in range(Nx-1):
        for n in range(Ny-1):

            for i in range(1,10):
                for j in range(1,5):

                    f[m,n] += np.exp(alpha*(x[m,n]-i)**2 +alpha*(y[m,n]-j)**2)

    return f


def boundary_value(x,y):

    """
    Example non-zero Dirichlet BC
    """

    return 1*np.ones_like(x)


def apply_nonzero_dirichlet(f,Nx,Ny,dx,dy,LeftX,RightX,LeftY,RightY):
    
    """ Modifies the RHS vector to include non zero dirichlet BC """
    
    f_mod = f.copy()

    x = np.linspace(LeftX,RightX,Nx+1)
    y = np.linspace(LeftY,RightY,Ny+1)

    # LEFT boundary
    for j in range(Ny-1):

        bc = boundary_value(LeftX,y[j+1])

        f_mod[0,j] += bc/(dx*dx)

    # RIGHT boundary
    for j in range(Ny-1):

        bc = boundary_value(RightX,y[j+1])

        f_mod[Nx-2,j] += bc/(dx*dx)

    # BOTTOM boundary
    for i in range(Nx-1):

        bc = boundary_value(x[i+1],LeftY)

        f_mod[i,0] += bc/(dy*dy)

    # TOP boundary
    for i in range(Nx-1):

        bc = boundary_value(x[i+1],RightY)

        f_mod[i,Ny-2] += bc/(dy*dy)

    return f_mod


def solve(A,f,Nx,Ny,LeftX,RightX,LeftY,RightY,order="f"):
    
    """ Given the system matrix A and the source function f, 
    return the solution u on each grid point 
    order = 'f' corresponds to lexicographic ordering """

    # reshape RHS
    fLX = f.reshape(-1, order=order)

    # solve linear system
    u = la.spsolve(A,fLX)

    u2d = u.reshape(f.shape, order=order)

    uArr = np.zeros((Nx+1,Ny+1))

    # interior
    uArr[1:-1,1:-1] = u2d

    x = np.linspace(LeftX,RightX,Nx+1)
    y = np.linspace(LeftY,RightY,Ny+1)

    # left/right boundaries
    for j in range(Ny+1):

        uArr[0,j] = boundary_value(LeftX,y[j])

        uArr[-1,j] = boundary_value(RightX,y[j])

    # bottom/top boundaries
    for i in range(Nx+1):

        uArr[i,0] = boundary_value(x[i],LeftY)

        uArr[i,-1] = boundary_value(x[i],RightY)

    return uArr


def plot_field(field,LeftX,RightX,LeftY,RightY,title):

    plt.figure()

    plt.imshow(field.transpose(),origin='lower',extent=[LeftX,RightX,LeftY,RightY])

    plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.colorbar(shrink=0.55)

    plt.show(block=True)


def main():

    # Problem parameters
    LeftX = 0.0
    RightX = 10.0

    LeftY = 0.0
    RightY = 5.0

    Nx = 100
    Ny = 80

    # generate mesh grid
    x,y,dx,dy = generate_grid(LeftX, LeftY,RightX, RightY,Nx, Ny)

    # FD matrix
    A = FDLaplacian2D(Nx,Ny,dx,dy)

    # source function
    f = sourcefunc(x,y,Nx,Ny)

    # modify RHS vector f
    f_mod = apply_nonzero_dirichlet(f,Nx,Ny,dx,dy,LeftX,RightX,LeftY,RightY)

    # solve
    u = solve(A, f_mod, Nx,Ny,LeftX,RightX,LeftY,RightY,order='f')

    # plots
    plot_field(f, LeftX,RightX, LeftY,RightY, 'Source function f')

    plot_field(u, LeftX,RightX, LeftY,RightY, 'Solution u')


if __name__ == "__main__":
    main()