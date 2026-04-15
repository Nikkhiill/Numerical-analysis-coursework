import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from matplotlib.animation import FuncAnimation

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
def create2DLFDM(Nx,Ny,dx,dy):
    diag1 = np.empty(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            diag1[m,n] = 4/(dx*dx)
            if (m-1==-1 and n-1==-1) or (m+2==Nx and n+2==Ny) or (m+2==Nx and n-1==-1) or (m-1 == -1 and n+2==Ny):
                diag1[m,n] = 2/(dx*dx)
            elif m-1==-1 or n-1==-1 or m+2==Nx or n+2==Ny:
                diag1[m,n] = 3/(dx*dx)
    diag1 = np.reshape(diag1,((Nx-1)*(Ny-1)),order='F')

    diag2 = np.zeros(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            diag2[m,n] = -1/(dx*dx)
    diag2[0,:] = 0
    diag2 = np.reshape(diag2,((Nx-1)*(Ny-1)),order='F')
    diag2 = np.delete(diag2,0)

    diag3 = np.zeros(shape=(Nx-1,Ny-1))
    for m in range(Nx-1):
        for n in range(Ny-1):
            diag3[m,n] = -1/(dx*dx)
    diag3 = np.delete(diag3,Ny-2,1)
    diag3 = np.reshape(diag3,((Nx-1)*(Ny-2)),order='F')

    diagonals = [diag1,diag2,diag2,diag3,diag3]
    A = sp.diags(diagonals,[0,-1,1,Nx-1,-(Nx-1)],format='csr')

    return A

def u_initial(Nx,Ny,a,b):
    
    r = 0.01*(a+b)*np.random.rand((Nx-1),(Ny-1))
    u0 = a*np.ones([(Nx-1),(Ny-1)]) + b*np.ones([(Nx-1),(Ny-1)]) + r
    
    return u0

def v_initial(Nx,Ny,a,b):
    
    v0 = (b/np.power(a+b,2))*np.ones([(Nx-1),(Ny-1)])
    
    return v0

def forward_euler(Du,Dv,A,u0,v0,k,T,Nx,Ny,Nt,a,b):
    
    dt = T/Nt
    
    save_every = 200
    frames_u = []
    frames_v = []
    
    one_mat = np.ones([(Nx-1)*(Ny-1),1])
    a_vec = a * one_mat
    b_vec = b * one_mat 
    
    u0 = np.reshape(u0,((Nx-1)*(Ny-1),1),order='F')
    v0 = np.reshape(v0,((Nx-1)*(Ny-1),1),order='F')
    
    unow = u0.copy()
    vnow = v0.copy()
    
    for step in range(Nt):
    
        Au = A @ unow
        Av = A @ vnow
        
        u2 = unow * unow
        
        fu = -Du*Au + k*(a_vec - unow + u2*vnow)
        fv = -Dv*Av + k*(b_vec - u2*vnow)
        
        unow += dt * fu
        vnow += dt * fv
        
        # Save frames occasionally
        if step % save_every == 0:
            
            frames_u.append(np.reshape(unow,(Nx-1,Ny-1),order='F').copy())
            frames_v.append(np.reshape(vnow,(Nx-1,Ny-1),order='F').copy())
    
        
    #Getting and reshaping u and v
    u = np.reshape(unow,(Nx-1,Ny-1),order='F')
    v = np.reshape(vnow,(Nx-1,Ny-1),order='F')
    
    
    return u,v,frames_u,frames_v

def plot_field(field, LeftX,RightX,LeftY,RightY, title):
    
    plt.figure()
    plt.imshow(field.transpose(), origin='lower',extent=[LeftX,RightX,LeftY,RightY])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(shrink=0.55)
    plt.show(block = True)
    
def animate_solution(frames, LeftX, RightX, LeftY, RightY, title, filename):
    
    fig, ax = plt.subplots()
    
    vmin = min(np.min(f) for f in frames)
    vmax = max(np.max(f) for f in frames)
    
    im = ax.imshow(frames[0].T, origin='lower',
                   extent=[LeftX,RightX,LeftY,RightY],
                   vmin=vmin, vmax=vmax)
    
    def update(i):
        im.set_array(frames[i].T)
        ax.set_title(f"{title} (frame {i})")
        return [im]
    
    ani = FuncAnimation(fig, update, frames=len(frames), interval=50)
    
    ani.save(filename, writer='pillow')
    plt.close()

def main():
    
    ## Problem parameters - 
    
    # Domain size
    LeftX = 0.0
    RightX = 4.0
    LeftY = 0.0
    RightY = 4.0
    
    # Discretization
    Nx = 100
    Ny = 100
    
    # Schnakenberg model parameters
    Du = 0.05
    Dv = 1.0
    k = 5
    a = 0.1305
    b = 0.7695
    
    # Time discretization
    T = 20
    Nt = 51000
    
    # generate mesh grid
    x,y,dx,dy = generate_grid(LeftX, LeftY, RightX, RightY, Nx, Ny)
    
    # Get the sparse FD matrix
    A = create2DLFDM(Nx,Ny,dx,dy)
    
    # initial conditions (T=0)
    u0 = u_initial(Nx,Ny,a,b)
    v0 = v_initial(Nx,Ny,a,b)
    
    # plotting the desired fields
    plot_field(u0, LeftX,RightX,LeftY,RightY, 'Forward Euler u(x,y,T) at T = 0')
    plot_field(v0, LeftX,RightX,LeftY,RightY, 'Forward Euler v(x,y,T) at T = 0')
    
    # solve
    u,v,frames_u, frames_v = forward_euler(Du,Dv,A,u0,v0,k,T,Nx,Ny,Nt,a,b)
    
    # plotting the desired fields
    plot_field(u, LeftX,RightX,LeftY,RightY, 'Forward Euler u(x,y,T) at T = 20')
    plot_field(v, LeftX,RightX,LeftY,RightY, 'Forward Euler v(x,y,T) at T = 20')
    
    animate_solution(frames_u, LeftX, RightX, LeftY, RightY,
                 "Activator u evolution", "u_animation.gif")

    animate_solution(frames_v, LeftX, RightX, LeftY, RightY,
                 "Inhibitor v evolution", "v_animation.gif")
    
if __name__ == "__main__":
    main()
    
    
    
    
    