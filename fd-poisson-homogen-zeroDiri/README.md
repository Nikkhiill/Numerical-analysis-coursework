# 2D Poisson Solver (Finite Difference Method, Dirichlet BC)

This project solves the 2D Poisson equation on a unifor rectangular domain using a
finite difference discretization and sparse linear algebra.

## Problem
We solve:

$$
-\Delta u(x,y) = f(x,y), \quad (x,y)\in \Omega = [L_x, R_x]\times[L_y, R_y]
$$

with homogeneous Dirichlet boundary conditions:

$$
u(x,y) = 0, \quad (x,y)\in \partial\Omega
$$

The unknowns are defined on the interior grid points. The discrete system is:

$$
A\ \mathbf{u}=\mathbf{f}
$$

where \(A\) is the finite-difference Laplacian matrix assembled via Kronecker
products.

## Method
- Uniform grid with `Nx` and `Ny` intervals in the \(x\) and \(y\) directions  
- Second-order 5-point stencil for $-\Delta$ on interior nodes  
- Sparse matrix assembly using:
\[
A = I_y \otimes L_{xx} + L_{yy} \otimes I_x
\]

where

$$
L_{xx} = D_x^{\mathsf T} D_x,\qquad L_{yy} = D_y^{\mathsf T} D_y.
$$
- Solve using `scipy.sparse.linalg.spsolve`

## Inputs you can change
In the Python file:
- Domain limits: `LeftX, RightX, LeftY, RightY`
- Grid: `Nx, Ny`
- Source term: `sourcefunc(x, y)`

## How to run
Install dependencies:

```bash
pip install numpy scipy matplotlib
python Poisson_FDM.py
```
## Output
- Heat map of the source function
- Heat map of the solution $u(x,y)$

Example figures showed in figures folder. The source function considered for the example is as follows:
<img width="287" height="97" alt="image" src="https://github.com/user-attachments/assets/bde9a393-5112-4ee7-b75a-8c1be459d235" />

## Assumptions:
- Uniform rectangular grid
- Homogeneous Poisson's equation ($k=1$)
- Zero dirichlet boundary condition
