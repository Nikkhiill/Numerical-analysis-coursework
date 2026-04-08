# 2D Poisson Solver (Finite Volume Method, Variable Coefficient)

This project solves the **2D Poisson equation with both homogeneous and non-homogeneous coefficients** on a rectangular domain using the **Finite Volume Method (FVM)** and sparse linear algebra.

---

## Problem

We solve the generalized Poisson equation:

\[
-\nabla \cdot \left( K(x,y)\nabla u(x,y) \right) = f(x,y), \quad (x,y)\in \Omega
\]

where:

- \(\Omega = [L_x, R_x] \times [L_y, R_y]\)  
- \(K(x,y)\) is a spatially varying diffusion coefficient  
- \(f(x,y)\) is the source term  

### Boundary Conditions

Homogeneous Dirichlet boundary conditions:

\[
u(x,y) = 0, \quad (x,y)\in \partial\Omega
\]

---

## Method

- Domain discretized into a **uniform Cartesian finite volume mesh**  
- Unknowns are defined at **cell centers**  
- Fluxes computed at control volume faces  
- Second-order discretization using central differences  

### Discrete System

The system is written as:

\[
A \mathbf{u} = \mathbf{f}
\]

where:

- \(A\) is a sparse matrix assembled from flux balances  
- \(\mathbf{u}\) contains cell-centered unknowns  
- \(\mathbf{f}\) is the source term  

---

## Key Features

- Supports:  
  - ✅ Homogeneous coefficient \(K = 1\)  
  - ✅ Non-homogeneous coefficient \(K(x,y)\)  
- Sparse matrix assembly using `scipy.sparse`  
- Efficient linear solve using `scipy.sparse.linalg.spsolve`  
- Modular structure for:  
  - grid generation  
  - coefficient definition  
  - system assembly  

---

## Grid

- Uniform grid with:

```text
Nx, Ny = number of control volumes in x and y directions
