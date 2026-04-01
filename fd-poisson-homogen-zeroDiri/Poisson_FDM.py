# Solving a 2D homogeneous Poisson's equation with a given source function
# and zero Dirichlet B.C. using Finite Difference Method (FDM)

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la


def generate_grid(LeftX, LeftY, RightX, RightY, Nx, Ny):
    
    """
    Inputs: X and Y co-ordinates of the domain and the no. of intervals in each direction
    Returns arrays x and y of the grid points and dx,dy - the mesh size
    """
    
    
    dx = (RightX-LeftX)/Nx
    dy = (RightY-LeftY)/Ny
    
    x,y = np.mgrid[LeftX+dx:RightX:dx,LeftY+dy:RightY:dy]
    
    return x,y,dx,dy


#Function for generating system matrix A
def FDLaplacian2D(Nx,Ny,dx,dy):
    
    """
    For a homogeneous Poisson's equation -del(u) = f, the sparse FD system matrix A is of the form
    -del(u_{i,j}) = [-u_{i-1,j} + 2u_{i,j} - u_{i+1,j}]/[d_x ^2] + [-u_{i,j-1} + 2u_{i,j} - u_{i,j+1}]/[d_y ^2]
    and taking the known values u_{i,j} to RHS we get the final form described in the Report
    
    Inputs: no. of intervals Nx and Ny, and the resulting mesh size
    Returns A - sparse csc format of the FD matrix
    """
    
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

#Function for generating the source function f - 
def sourcefunc(x,y,Nx,Ny):
    
    """
    Evaluates the given source function f on the generated grid points
    
    f(x,y) = sum_{i=1..9} sum_{j=1..4} exp(alpha*(x-i)^2 + alpha*(y-j)^2)
    
    Inputs: grid points x and y, and the no. of intervals Nx and Ny
    Return an array f of the source function evaluated at grid points
    """
    
    alpha = -40
    f = np.zeros(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            for i in range(1,10):
                for j in range(1,5):
                    f[m,n] = f[m,n]+np.exp(alpha*np.power(x[m,n]-i,2)+alpha*np.power(y[m,n]-j,2))

    return f

def homPoisson_zeroDiri_solve(A, f, order = "f"):
    
    """
    Given the system matrix A and the source function f,
    return the solution u on each grid point
    
    order = 'f' corresponds to lexicographic ordering
    """
    
    # Reshape 2d array of source function into a 1d array following lexicograhic ordering (f)
    fLX = f.reshape(-1, order=order)
    
    # Solve u = A^{-1} * f
    u = la.spsolve(A,fLX)
    u2d = u.reshape(f.shape, order=order)
    
    # Convert 1d solution array u to a 2d array and append 0's corresponding to zero Dirichlet BC
    uArr = np.pad(u2d, pad_width=((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    
    return uArr

def plot_field(field, LeftX,RightX,LeftY,RightY, title):
    
    plt.figure()
    plt.imshow(field.transpose(), origin='lower',extent=[LeftX,RightX,LeftY,RightY])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(shrink=0.55)
    plt.show(block = True)

def main():
    
    #Problem parameters - 
    LeftX = 0.0
    RightX = 10.0
    LeftY = 0.0
    RightY = 5.0
    Nx = 100
    Ny = 80
    
    # generate mesh grid
    x,y,dx,dy = generate_grid(LeftX, LeftY, RightX, RightY, Nx, Ny)
    
    # Get the sparse FD matrirx
    A = FDLaplacian2D(Nx,Ny,dx,dy)
    
    # specify the source function f (modify the source function accordingly)
    f = sourcefunc(x,y,Nx,Ny)
    
    # main solve
    u = homPoisson_zeroDiri_solve(A, f, order='f')
    
    # plotting the desired fields
    plot_field(f, LeftX,RightX,LeftY,RightY, 'Source function f')
    plot_field(u, LeftX,RightX,LeftY,RightY, 'Solution u')
    
if __name__ == "__main__":
    main()
    