# Solving a 2D homogeneous & non-homogeneous Poisson's equation with a given source function
# and zero Dirichlet B.C. using Finite Volume Method (FDM)

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la


# Generate grid for FV mesh
def generate_FVgrid(LeftX, LeftY, RightX, RightY, Nx, Ny):
    
    """
    Inputs: X and Y co-ordinates of the domain and the no. of intervals in each direction
    Returns arrays x and y of the grid points of the Finite Volume and dx,dy - the mesh size
    """
    
    
    dx = (RightX-LeftX)/Nx
    dy = (RightY-LeftY)/Ny
    
    x,y = np.mgrid[LeftX+dx:RightX:dx,LeftY+dy:RightY:dy]
    
    return x,y,dx,dy

# Generate grid for coefficient K
def generate_Kgrid(LeftX, LeftY, RightX, RightY, Nx, Ny):
    
    """
    Inputs: X and Y co-ordinates of the domain and the no. of intervals in each direction
    Returns arrays x and y of the grid points on which the coefficient K is evaluated
    """
    
    
    dx = (RightX-LeftX)/Nx
    dy = (RightY-LeftY)/Ny
    
    x,y = np.mgrid[LeftX:RightX+dx:dx,LeftY:RightY+dy:dy]
    
    return x,y

#Function for generating the source function f - 
def sourcefunc(x,y,Nx,Ny):
    alpha = -40
    f = np.zeros(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            for i in range(1,10):
                for j in range(1,5):
                    f[m,n] = f[m,n]+np.exp(alpha*np.power(x[m,n]-i,2)+alpha*np.power(y[m,n]-j,2))

    return f

#Defining the coefficient functions - 
def coeffK1(x,y,Nx,Ny):
    
    """ Homogeneous coefficient K = 1
    """
    
    K=np.ones(shape=(Nx+1,Ny+1))
    
    return K

def coeffK2(x,y,Nx,Ny):
    
    """ Non homogeneous coefficient example - 
    K = 1 + 0.1*(x + y + xy)
    """
    
    K=1+0.1*(x+y+np.multiply(x,y))
    
    return K


def create2DLFVM(Nx,Ny,dx,dy,K):
    
    """ Creates a 2D FVM matrix for homogeneous and non-homogeneous Poisson
    equation with zero Dirichlet boundary condition
    """
    
    diag1 = np.empty(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            diag1[m,n] = (K[m,n+1]+K[m+1,n+1])/(2*np.power(dx,2)) + (K[m+1,n]+K[m+1,n+1])/(2*np.power(dy,2)) + (K[m+1,n+1]+K[m+2,n+1])/(2*np.power(dx,2)) + (K[m+1,n+1]+K[m+1,n+2])/(2*np.power(dy,2))
    diag1 = np.reshape(diag1,((Nx-1)*(Ny-1)),order='F')

    diag2 = np.zeros(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            diag2[m,n] = -1*(K[m,n+1]+K[m+1,n+1])/(2*np.power(dx,2))
    diag2[0,:] = 0
    diag2 = np.reshape(diag2,((Nx-1)*(Ny-1)),order='F')
    diag2 = np.delete(diag2,0)

    diag3 = np.zeros(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            diag3[m,n] = -1*(K[m+1,n+1]+K[m+1,n+2])/(2*np.power(dy,2))
    diag3 = np.delete(diag3,Ny-2,1)
    diag3 = np.reshape(diag3,((Nx-1)*(Ny-2)),order='F')

    diagonals = [diag1,diag2,diag2,diag3,diag3]
    A = sp.diags(diagonals,[0,-1,1,Nx-1,-(Nx-1)],format='csc')

    return A


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
    Nx = 200
    Ny = 100

    # Generate grid for FV mesh
    x,y,dx,dy = generate_FVgrid(LeftX, LeftY, RightX, RightY, Nx, Ny)
    
    # Generate grid for coefficient K
    x1, y1 = generate_Kgrid(LeftX, LeftY, RightX, RightY, Nx, Ny)
    
    # specify the source function f (modify the source function accordingly)
    f = sourcefunc(x,y,Nx,Ny)
    
    # Generate coefficient K (modify/add this coefficient accordingly)
    K = coeffK2(x1,y1,Nx,Ny)
    
    # Get the sparse 2D FVM matrix A
    A = create2DLFVM(Nx,Ny,dx,dy,K)
    
    # Main solve
    u = homPoisson_zeroDiri_solve(A, f, order = "f")
    
    # plotting the desired fields
    plot_field(f, LeftX,RightX,LeftY,RightY, 'Source function f')
    plot_field(u, LeftX,RightX,LeftY,RightY, 'Solution u')
    plot_field(K, LeftX,RightX,LeftY,RightY, 'Coefficient function K')
    
if __name__ == "__main__":
    main()

